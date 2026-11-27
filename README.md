# ClusterManager

Distributed GPU training cluster manager, live multi-rank debugger, and performance monitor.

Download the latest binaries from **[Releases](https://github.com/findlamp/ClusterManager-releases/releases/latest)**.

## Install

### 1. macOS Client (Apple Silicon)

Download `ClusterManager-0.2.4-arm64.dmg` from [Releases](https://github.com/findlamp/ClusterManager-releases/releases/latest), drag `ClusterManager.app` into `/Applications`, then run once before first launch:

```bash
xattr -cr /Applications/ClusterManager.app
```

### 2. Linux Server (`x86_64`, Python 3.8+)

```bash
pip install https://github.com/findlamp/ClusterManager-releases/releases/download/v0.2.4/clustermanager-0.2.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Quick Start

### Single Node

```bash
cm-node
```

### Multi-Node Cluster (Seed Auto-Discovery)

If your launcher already sets `RANK`, `WORLD_SIZE`, and `MASTER_ADDR` (Kubernetes / Slurm / TorchElastic), just run `cm-node` on every node.

Or specify them manually:

```bash
# Node 0 (seed/master, e.g. 10.0.0.1)
cm-node --rank=0 --world-size=2 --master=10.0.0.1

# Node 1 (e.g. 10.0.0.2; repeat for rank 2, 3, ...)
cm-node --rank=1 --world-size=2 --master=10.0.0.1
```

Open **ClusterManager.app** on macOS and attach to the seed address (`10.0.0.1:50351`) — all nodes are discovered automatically.

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

## Documentation

- [System Monitor Guide (Data / Training / Inference Telemetry & Readiness)](docs/system-monitor.md)
