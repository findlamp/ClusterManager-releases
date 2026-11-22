# ClusterManager

Distributed GPU training cluster manager, live multi-rank debugger, `py-spy` stack profiler, and integrated AI agent workspace.

> **Note**: This repository is for **binary release distribution and issue tracking only**. Source code is maintained privately.

---

## Downloads (`v0.1.0`)

Go to **[Releases](https://github.com/findlamp/ClusterManager-releases/releases/latest)** to download the latest builds:

| Component | Platform | Artifact |
| --- | --- | --- |
| **Desktop App** (macOS GUI + bundled local server & Paseo) | macOS Apple Silicon (`arm64`) | `ClusterManager-0.1.0-arm64.dmg` |
| **Server / Node Runtime + `cm_sdk`** (static pure-Rust `cm-node`, `cm-server`, `cm-mcp` + Python `cm_sdk`) | Linux `x86_64` (`manylinux_2_17` / static musl) | `clustermanager-0.1.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |

---

## 1. Install Desktop App (macOS Apple Silicon)

1. Download `ClusterManager-0.1.0-arm64.dmg` from [Releases](https://github.com/findlamp/ClusterManager-releases/releases/latest).
2. Open the `.dmg` and drag **ClusterManager.app** into `/Applications`.
3. Because this build is ad-hoc signed (not notarized by Apple), macOS Gatekeeper may block the first launch with *"app is damaged"* or *"unverified developer"*. Run once in Terminal to clear the quarantine flag:

```bash
xattr -cr /Applications/ClusterManager.app
```

4. Launch **ClusterManager** from `/Applications`.

---

## 2. Install Server & Node Runtime (Linux `x86_64`)

The Linux wheel bundles statically linked pure-Rust binaries (`cm-node`, `cm-server`, `cm-mcp`) plus the zero-dependency `cm_sdk` Python package, and automatically installs `py-spy`. It works on Python 3.8+ across Ubuntu, Debian, CentOS, Rocky, and Docker containers without glibc version issues.

### Direct install from GitHub Releases

```bash
pip install https://github.com/findlamp/ClusterManager-releases/releases/download/v0.1.0/clustermanager-0.1.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

For offline / air-gapped servers (after transferring the `.whl` file):

```bash
pip install --no-deps ./clustermanager-0.1.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### Start a Cluster Node (`cm-node`)

```bash
cm-node \
  --uid=node-0 \
  --bind-host=0.0.0.0 \
  --port=50351 \
  --token="your-cluster-token"
```

### Optional: Use `cm_sdk` in PyTorch Training Scripts

```python
from cm_sdk import track, dbg

track.init(name="my-run", config={"lr": 3e-4})
dbg.init(enabled=True)

for step in range(100):
    with track.span("forward", cuda=True):
        ...
    track.log({"train/loss": loss}, step=step)
    track.step_end(step, samples=64)
    dbg.breakpoint("after-step")

track.finish("ok")
```

---

## Checksums

Every release includes `SHA256SUMS.txt`. Verify downloaded files with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```
