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

Start `cm-node` on your Linux GPU server:

```bash
cm-node --uid=node-0 --bind-host=0.0.0.0 --port=50351 --token="your-token"
```

Then open **ClusterManager.app** on macOS and attach to `<server-ip>:50351` with the same token.

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
