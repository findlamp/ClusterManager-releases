# System Monitor 接入指南（`cm_sdk.sys`）

在 **System Monitor** 画布中，各集群节点的 **GPU 利用率 / 显存 / 功耗 / CPU / 内存** 会由 `cm-node` 自动采集并汇总到本组 Rank-0，无需改代码。

如果需要展示 **生产 / 消费 / 吞吐速率、队列深度、组间连线带宽（GB/s & 累计 TB）**，以及在按拓扑启动时实现 **全节点（N/N）服务就绪后再拉起下游**，只需在业务代码里加入 2~4 行 `cm_sdk.sys` 调用：

- 与 `cm_sdk.track` **完全隔离**，不会污染 `Experiments` 曲线；
- 零额外依赖（仅 Python 标准库），在非 ClusterManager 环境下自动静默 no-op；
- 无需配置地址或 Group ID，`cm-node` 会自动注入 `CM_GROUP_ID`、`CM_LAUNCH_ID`、`CM_RANK`、`CM_PERF_INGEST_ADDR`。

---

## 1. 数据集群（DataGen / 数据服务）

使用 `cmsys.link(target)` 可用 **1 个对象** 同时上报「生产速度 + 缓冲队列 + 连线实时带宽（GB/s）与累计流量（TB）」：

```python
from cm_sdk import sys as cmsys

# 1) 启动阶段（可选）
cmsys.starting("Initializing data service...")
start_data_server(port=9100)

# 2) 服务可对外接收请求时调用（每台节点都调用；集齐 N/N 自动拉起下游训练组）
cmsys.ready("Data service listening on :9100")

# 3) 创建 1 Hz 自动滑窗计量器：target 填画布中下游 Group 名称（单出边时也可自动匹配）
link = cmsys.link(target="train-group", unit="samples/s", scope="node")

for batch, payload_bytes in stream_batches():
    send_to_trainer(payload_bytes)
    # 线程安全，内部每 1 秒自动汇总上报一次，不阻塞热路径
    link.add(
        items=len(batch),
        nbytes=len(payload_bytes),
        queue=f"{buf_queue.qsize()}/{buf_queue.maxsize}",
    )
```

> **多进程 / Ray Actor 注意**：`scope="node"` 按机器（`node_uid`）聚合，**同机多个进程各自调 `scope="node"` 会互相覆盖**。如果单机跑了多个发包进程或 Ray Actor，建议汇总到单个中心进程 / Driver 后用 `scope="group"` 统一上报一次。

---

## 2. 训练集群（Training）

分布式训练通常每机多卡（如 8 卡 `torchrun`）：
- 每台机器仅 `LOCAL_RANK == 0` 上报一次 `ready`；
- 仅全局 `RANK == 0` 使用 `scope="group"` 上报全局吞吐，避免 8 卡 / 多机重复累加。

```python
import os
from cm_sdk import sys as cmsys

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))

init_distributed_and_model()

# 1) 每台节点 local_rank == 0 报就绪
if local_rank == 0:
    cmsys.ready("Trainer initialized")

# 2) 全局 Rank 0 用 scope="group" 上报整组消费速度（samples/s、tok/s 或 steps/s）
meter = (
    cmsys.meter("consume", unit="samples/s", scope="group")
    if global_rank == 0
    else None
)

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    if meter is not None:
        meter.add(items=global_batch_size, status=f"step {step} · loss {loss:.3f}")
```

如果训练框架内部已经算好全局 `tokens_per_second` 或 `steps_per_second`，也可以在全局 Rank 0 直接调用 `cmsys.report(...)`：

```python
if global_rank == 0:
    cmsys.report(
        consume_rate=tokens_per_sec,
        consume_unit="tok/s",
        secondary_label="训练进度",
        secondary_value=f"step {step} · loss {loss:.3f}",
        scope="group",
    )
```

---

## 3. 推理 / 评测集群（Inference / Eval）

```python
from cm_sdk import sys as cmsys

cmsys.starting("Loading model weights...")
engine = start_inference_server(port=8000)
cmsys.ready("Inference ready on :8000")

meter = cmsys.meter("throughput", unit="tok/s", scope="node")

def on_request_done(num_tokens: int, latency_ms: float):
    meter.add(items=num_tokens, status=f"{latency_ms:.0f}ms")
```

> **不改代码的替代方案**（如直接跑 `vLLM` / `SGLang` 命令行）：在 System Monitor 右侧 Inspector 中把就绪检测改为 **`HTTP 健康检查 (GET)`**（如端口 `8000` + 路径 `/health`）、**`TCP 端口监听`** 或 **`终端日志关键字匹配`**，即可零改动参与拓扑启动。

---

## 4. API 速查表

### Python SDK（`from cm_sdk import sys as cmsys`）

| API | 用途 |
| --- | --- |
| `cmsys.starting(msg)` | 标记本节点正在初始化（`ready=False`） |
| `cmsys.ready(msg)` | 标记本节点服务已就绪；Rank-0 集齐 `N/N` 后放行下游 |
| `cmsys.not_ready(msg)` | 清除本节点就绪标记 |
| `cmsys.link(target, unit="samples/s", scope="node")` | 返回 1 Hz 滑窗 `RateMeter`，同时上报卡片生产速度与连线带宽/累计流量 |
| `cmsys.meter(role="consume", unit="samples/s", scope="node")` | 返回 1 Hz 滑窗 `RateMeter`，仅上报当前卡片速率（`role`: `"produce"` / `"consume"` / `"throughput"`） |
| `meter.add(items=0, nbytes=0, queue=None, status=None, flush=False)` | 累计本次样本数与字节数，自动 1 Hz 限频上报 |
| `cmsys.report(produce_rate=..., consume_rate=..., throughput=..., queue=..., secondary_label=..., secondary_value=..., scope="node")` | 直接推送已计算好的瞬时速率 |
| `cmsys.report_link(target, bytes_per_s=..., total_bytes=..., items_per_s=..., scope="node")` | 直接推送连线实时速率与累计字节数 |
| `cmsys.check_readiness(server_url, cluster_id, group_id, mode="sdk_ready")` | 外部脚本单次查询某 Group 全节点就绪状态 |
| `cmsys.wait_ready(server_url, cluster_id, group_id, mode="sdk_ready", timeout_s=300)` | 外部脚本阻塞等待某 Group 全节点就绪 |

### `scope` 选择规则

| `scope` | Rank-0 聚合行为 | 适用场景 |
| --- | --- | --- |
| `"node"`（默认） | 按节点求和 | 每台机器有 1 个上报进程，各报本机速率/流量 |
| `"group"` | 直接使用该值 | 单点已知全局汇总值（如训练 Rank 0 或 Ray Driver） |

### HTTP 就绪检测与拓扑控制接口（`cm-server`，默认 `127.0.0.1:50052`）

```bash
# 检测某 Group 全节点（N/N）服务是否就绪
curl -s -X POST http://127.0.0.1:50052/api/clusters/<cluster_id>/groups/<group_id>/readiness \
  -H 'Content-Type: application/json' \
  -d '{"mode":"sdk_ready","rank0_only":false}'

# 后端按拓扑顺序启动 / 逆拓扑停止 / 取消编排
curl -s -X POST http://127.0.0.1:50052/api/system/blueprints/<id>/start
curl -s -X POST http://127.0.0.1:50052/api/system/blueprints/<id>/stop
curl -s -X POST http://127.0.0.1:50052/api/system/blueprints/<id>/cancel
```
