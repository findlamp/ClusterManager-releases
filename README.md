# ClusterManager

Distributed GPU training cluster manager, live multi-rank debugger, and performance monitor.

Download the latest binaries from **[Releases](https://github.com/findlamp/ClusterManager-releases/releases/latest)**.

## Install

### 1. macOS Client (Apple Silicon)

Download `ClusterManager-0.1.0-arm64.dmg` from [Releases](https://github.com/findlamp/ClusterManager-releases/releases/latest), drag `ClusterManager.app` into `/Applications`, then run once before first launch:

```bash
xattr -cr /Applications/ClusterManager.app
```

### 2. Linux Server (`x86_64`, Python 3.8+)

```bash
pip install https://github.com/findlamp/ClusterManager-releases/releases/download/v0.1.0/clustermanager-0.1.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Quick Start

### Single Node

```bash
cm-node --port=50351
```

### Multi-Node Cluster (Seed / Master Auto-Discovery)

Start all nodes pointing to the same seed (`--master`). Each node registers itself automatically:

```bash
# Node 0 (seed/master, e.g. 10.0.0.1)
cm-node --uid=node-0 --advertise-url=10.0.0.1:50351 --port=50351 --rank=0 --world-size=2 --master=10.0.0.1

# Node 1 (e.g. 10.0.0.2; repeat for rank 2, 3, ...)
cm-node --uid=node-1 --advertise-url=10.0.0.2:50351 --port=50351 --rank=1 --world-size=2 --master=10.0.0.1
```

If your cluster launcher already sets `RANK`, `WORLD_SIZE`, and `MASTER_ADDR` (e.g. Kubernetes / Slurm / TorchElastic), they are picked up automatically:

```bash
cm-node --uid="node-${RANK}" --advertise-url="$(hostname -I | awk '{print $1}'):50351" --port=50351
```

Open **ClusterManager.app** on macOS and attach to the single seed address (`10.0.0.1:50351`) — all nodes are discovered automatically.

> On shared networks, optionally add `--token=<secret>` (or `CM_NODE_TOKEN`) to all nodes and enter the same token when attaching.

## Training Script Example (Breakpoint + Performance Monitor)

```python
from cm_sdk import dbg

dbg.init()
monitor = dbg.PerfMonitorClient()

for step, batch in enumerate(loader):
    monitor.record("data")
    with monitor.duration("forward"):
        loss = model(batch)
    with monitor.duration("backward"):
        loss.backward()
    with monitor.duration("optim"):
        optimizer.step()

    monitor.flush(step, enable=(step % 10 == 0), metadata={"loss": float(loss)})
    dbg.breakpoint("after-step")
```
