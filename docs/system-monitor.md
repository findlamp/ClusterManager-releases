# System Monitor Guide (`cm_sdk.sys`)

In **System Monitor**, hardware metrics (**GPU utilization, VRAM, power, CPU, memory**) are collected automatically by `cm-node` and aggregated on each Group's Rank-0 — no code changes required.

To report **produce / consume / throughput rates, queue depth, inter-group link bandwidth (GB/s & cumulative TB)**, and to gate **topological startup** until **all nodes (`N/N`)** in an upstream Group are ready, add 2–4 lines of `cm_sdk.sys`:

- **Decoupled from `cm_sdk.track`**: never writes to experiment directories or pollutes the `Experiments` panel.
- **Zero dependencies**: Python stdlib only; silently no-ops when launched outside ClusterManager.
- **Zero configuration**: `cm-node` injects `CM_GROUP_ID`, `CM_LAUNCH_ID`, `CM_RANK`, and `CM_PERF_INGEST_ADDR` automatically.

---

## 1. Data Service (DataGen / Rollout / Preprocessing)

The data service does **not** need to know which downstream Group it connects to. Simply report outbound sample counts and payload bytes — whichever Group you wire to it on the canvas automatically receives live bandwidth (`GB/s`, `Gbps`) and cumulative traffic (`TB`):

```python
from cm_sdk import sys as cmsys

# 1) Optional warmup status
cmsys.starting("Initializing data service...")
start_data_server(port=9100)

# 2) Signal readiness once accepting requests (call on every node;
#    downstream Groups start automatically once N/N nodes report ready)
cmsys.ready("Data service listening on :9100")

# 3) 1 Hz sliding-window meter (no downstream target name required)
meter = cmsys.meter("produce", unit="samples/s", scope="node")

for batch, payload_bytes in stream_batches():
    send_to_trainer(payload_bytes)
    # Passing `size_bytes` updates both the node's produce rate / queue
    # and the connected edge's live bandwidth / total bytes
    meter.record(
        count=len(batch),
        size_bytes=len(payload_bytes),
        queue=f"{buf_queue.qsize()}/{buf_queue.maxsize}",
    )
```

> - **1-to-N fan-out (optional `target`)**: Only when a single data Group fans out to **2+ distinct downstream edges** (e.g., `train` and `eval`) with separate streams do you need `cmsys.link(target="eval")` or `cmsys.meter("produce", target="eval")` (matched against the downstream Group name or the edge label in the Inspector).
> - **Multi-process / Ray actors**: `scope="node"` deduplicates by machine (`node_uid`), so multiple processes on the same host calling `scope="node"` will overwrite each other. When running multiple sender processes or Ray actors, aggregate in a single driver/coordinator and report once with `scope="group"`.

---

## 2. Training Cluster

For multi-GPU distributed training (e.g. `torchrun` with 8 GPUs per node):
- Call `cmsys.ready(...)` once per node (`LOCAL_RANK == 0`).
- Report global throughput from global `RANK == 0` with `scope="group"` to avoid double-counting across GPUs/nodes.

```python
import os
from cm_sdk import sys as cmsys

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))

init_distributed_and_model()

# 1) One ready signal per node
if local_rank == 0:
    cmsys.ready("Trainer initialized")

# 2) Global Rank 0 reports cluster-wide consume rate (samples/s, tok/s, or steps/s)
meter = (
    cmsys.meter("consume", unit="samples/s", scope="group")
    if global_rank == 0
    else None
)

for step, batch in enumerate(dataloader):
    train_step(batch)
    if meter is not None:
        meter.record(count=global_batch_size, status=f"step {step}")
```

If your training framework already computes global `tokens_per_sec` or `steps_per_sec`, you can call `cmsys.report(...)` directly on global Rank 0:

```python
if global_rank == 0:
    cmsys.report(
        consume_rate=tokens_per_sec,
        consume_unit="tok/s",
        secondary_label="Progress",
        secondary_value=f"step {step}",
        scope="group",
    )
```

> Note: `cmsys.report(...)` sends immediately on each call (unthrottled), whereas `meter.record(...)` throttles automatically to at most 1 Hz (`interval_s=1.0`). The UI refreshes from Rank-0 every 2 seconds.

---

## 3. Inference / Evaluation Cluster

```python
from cm_sdk import sys as cmsys

cmsys.starting("Loading model weights...")
engine = start_inference_server(port=8000)
cmsys.ready("Inference ready on :8000")

meter = cmsys.meter("throughput", unit="tok/s", scope="node")

def on_request_done(num_tokens: int, latency_ms: float):
    meter.record(count=num_tokens, status=f"{latency_ms:.0f}ms")
```

> **Zero-code readiness alternative** (e.g. off-the-shelf `vLLM` / `SGLang`): In the right-hand Inspector, set readiness mode to **`http_get`** (e.g. port `8000`, path `/health`), **`tcp_port`**, or **`log_regex`** to participate in topological startup without editing server code.

---

## 4. API Quick Reference

### Python SDK (`from cm_sdk import sys as cmsys`)

| API | Description |
| --- | --- |
| `cmsys.starting(msg)` | Mark this node as warming up (`ready=False`) |
| `cmsys.ready(msg)` | Mark this node as ready; topological startup advances once `N/N` nodes are ready |
| `cmsys.not_ready(msg)` | Clear this node's ready flag |
| `cmsys.meter(role="consume", unit="samples/s", scope="node")` | Create a 1 Hz sliding-window `RateMeter` (`role`: `"produce"` / `"consume"` / `"throughput"`); passing `size_bytes` automatically feeds outbound link traffic |
| `cmsys.link(target="default", unit="samples/s", scope="node")` | Explicit stream-targeted `RateMeter` for 1-to-N fan-out |
| `meter.record(count=0, size_bytes=0, queue=None, status=None, flush=False)` | Record processed count and/or bytes; automatically throttled to `interval_s` (default `1.0s`) |
| `cmsys.report(produce_rate=..., consume_rate=..., throughput=..., queue=..., secondary_label=..., secondary_value=..., scope="node")` | Push pre-computed instantaneous rates immediately |
| `cmsys.report_link(target="default", bytes_per_s=..., total_bytes=..., items_per_s=..., scope="node")` | Push instantaneous link bandwidth and cumulative bytes immediately |
| `cmsys.check_readiness(server_url, cluster_id, group_id, mode="sdk_ready")` | Check all-nodes (`N/N`) readiness of a Group from an external script |
| `cmsys.wait_ready(server_url, cluster_id, group_id, mode="sdk_ready", timeout_s=300)` | Block until all nodes in a Group are ready (raises on error/timeout) |

### Choosing `scope`

| `scope` | Rank-0 Aggregation | When to Use |
| --- | --- | --- |
| `"node"` (default) | Summed across nodes | One reporting process per host, each reporting local rate/bytes |
| `"group"` | Used directly as cluster total | Single coordinator already knows the global total (e.g. training Rank 0 or Ray driver) |

### Control-Plane HTTP API (`cm-server`, default `127.0.0.1:50052`)

```bash
# Check all-nodes (N/N) service readiness for a Group
curl -s -X POST http://127.0.0.1:50052/api/clusters/<cluster_id>/groups/<group_id>/readiness \
  -H 'Content-Type: application/json' \
  -d '{"mode":"sdk_ready","rank0_only":false}'

# Start in topological order / stop in reverse order / cancel orchestration
curl -s -X POST http://127.0.0.1:50052/api/system/blueprints/<id>/start
curl -s -X POST http://127.0.0.1:50052/api/system/blueprints/<id>/stop
curl -s -X POST http://127.0.0.1:50052/api/system/blueprints/<id>/cancel
```
