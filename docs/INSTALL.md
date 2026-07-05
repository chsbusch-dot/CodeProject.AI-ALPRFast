# Adding an ALPR (License-Plate-Recognition) module to CodeProject.AI Server 2.9.7

A hands-on install manual for a running **CodeProject.AI Server 2.9.7** whose dashboard
and **Install modules** menu are already available.

---

## Read this first: you install exactly ONE ALPR reader

Every ALPR module for CodeProject.AI (CPAI) serves the **same** endpoint:

```
POST /v1/vision/alpr
```

Internally CPAI routes that endpoint through a single work queue named **`alpr_queue`**.
**Only one module may own that route/queue at a time.** If two ALPR modules are installed
and running at once, CPAI **round-robins** requests between them — so half your reads come
from one reader and half from the other, giving mixed and often garbage results.

> **Rule:** install and run **one** ALPR reader. Disable or uninstall the others.
> (See ["Switching readers / avoiding the two-ALPR conflict"](#switching-readers--avoiding-the-two-alpr-conflict).)

### The three options and how to pick

| Option | What it is | Install path | Best for |
|---|---|---|---|
| **A. Stock "License Plate Reader" (ALPR)** | Official CPAI module, PaddleOCR-based, by Mike Lud | **One click** from the dashboard's module registry | Easiest. Windows GPU or any-platform CPU. The default choice. |
| **B. ALPR (YOLO11)** | Mike Lud's newer YOLO11/ONNX ALPR (plate + char + state + vehicle) | Dashboard **if listed**, otherwise **side-load from GitHub** | Newer pipeline; ONNX. GPU via **DirectML on Windows** (stock); Linux/NVIDIA needs the CUDA fork. |
| **C. ALPRFast** | Third-party `fast-alpr` (YOLOv9 plate detector + `fast-plate-ocr` CCT OCR) with a CCTV accuracy stack | **Manual / side-load only** (not in the registry) | **Linux + NVIDIA CUDA**, wide 4K CCTV overview cameras. Manual install. |

Quick guidance:

- **Windows, or you just want it working:** use **A (stock ALPR)**. One click.
- **You want Mike Lud's newer YOLO11 reader:** use **B**. On Windows it GPU-accelerates
  with DirectML out of the box; on **Linux + NVIDIA** you need the CUDA-enabled fork
  (plain upstream YOLO11 is `onnxruntime-directml`, i.e. Windows/DirectML + Apple/MPS only).
- **Linux + NVIDIA GPU, wide CCTV, and you care about distant/angled plates:** use **C
  (ALPRFast)**. It is the option purpose-built for that case, at the cost of a manual install.

---

## Prerequisites

1. **CodeProject.AI Server 2.9.7 running**, dashboard reachable (default
   `http://<host>:32168`), and the **Install modules** menu visible.

2. **An object-detection module installed** (e.g. **ObjectDetectionYOLOv5**, any of the
   YOLOv5/v8/v11 object-detection modules). ALPR modules read plates; they need something
   to find *vehicles* first:
   - Stock ALPR and ALPRFast use an object detector for their **crop-first** step (find
     the car in a wide frame, crop it, then read the plate on the crop).
   - Install an object-detection module from the same **Install modules** menu **before**
     (or alongside) the ALPR module.

3. **Know your platform** — the module directory location and side-load procedure differ:
   - **Windows (native install)**
   - **Linux (native install)**
   - **Docker** (Linux container, Windows/macOS host)

   Where the `modules` directory lives per platform:

   | Platform | Modules directory |
   |---|---|
   | **Windows (native)** | `C:\Program Files\CodeProject\AI\modules` |
   | **Linux (native)** | under the server install tree, e.g. `.../CodeProject.AI-Server/src/modules` (dashboard-installed modules land here; confirm against your install) |
   | **Docker (inside container)** | `/app/modules` (dynamically installed/side-loaded) and `/app/preinstalled-modules` (baked into the image) |

   For **Docker**, `/app/modules` is normally **bind-mounted to a host folder** so modules
   survive container restarts/updates. The default host mappings are:

   | Docker host OS | Host folder mapped to `/app/modules` |
   |---|---|
   | **Windows** | `C:\ProgramData\CodeProject\AI\docker\modules` |
   | **Linux** | `/opt/codeproject/ai` (modules subtree) |
   | **macOS** | `/Library/Application Support/CodeProject/AI/docker/modules` |

   > If your container does **not** have `/app/modules` mounted to a host folder, add that
   > mount (see the Docker note in section **C**) before side-loading — otherwise your
   > module is lost on the next container recreate.

4. **GPU / driver notes:**
   - A healthy **NVIDIA driver** on the host is required for any CUDA path. For **Docker
     + GPU**, the host also needs the **NVIDIA Container Toolkit** and the container must
     be started with `--gpus all` (or `--runtime nvidia`), and you should use a
     **CUDA-tagged** CPAI image (e.g. `codeproject/ai-server:cuda*`).
   - CPAI exposes per-module GPU settings in `modulesettings.json` under `GpuOptions`:
     `InstallGPU` (install GPU libs during setup), `EnableGPU` (use the GPU at runtime),
     and `HalfPrecision` (`enable`/`force`/`disable`). The dashboard also exposes
     GPU/half-precision controls per module — the exact toggle labels vary by build and
     are noted per-module below.

> **Note on 2.9.7 dashboard labels.** The general flow below is correct, but the *exact*
> button/link wording ("Install", "Available modules", the GPU checkbox text) can differ
> slightly between 2.9.x point builds. Where this manual can't confirm the 2.9.7-exact
> label, it says so — trust the described location over a specific quoted word.

---

## A. Stock "License Plate Reader" (ALPR) — dashboard, one click

This is the official module (module id **`ALPR`**, name **"License Plate Reader"**,
PaddleOCR-based, by Mike Lud). It is in the default CPAI module registry, so no files to
download by hand.

### A.1 Install from the dashboard

1. Open the dashboard: `http://<host>:32168`.
2. Go to the **module install** view (the **Install modules** menu / the list of
   available modules — in 2.9.x this is the section that lists modules with an
   **Install** action next to each).
3. Find **"License Plate Reader"** (category *Computer Vision*). The registry only shows
   modules compatible with your server version, so on 2.9.7 the current ALPR release is
   what you'll see.
4. Click **Install** next to it. CPAI downloads the module, creates its per-module Python
   virtual environment, and installs dependencies (PaddleOCR / PaddlePaddle). This takes
   several minutes on first install.
5. When it finishes, the module appears under the **running/installed** modules on the
   main dashboard with a **Started** status.

Make sure an **object-detection** module is also installed (prerequisite #2).

### A.2 GPU

- The stock ALPR module ships with **`InstallGPU: false`** by default, because
  PaddlePaddle-GPU installs cleanly essentially only on **Windows**. On **Windows + NVIDIA**
  you can enable GPU for it in the module's settings (the dashboard's GPU / half-precision
  controls, backed by `GpuOptions.EnableGPU` / `HalfPrecision`).
- On **Linux + NVIDIA**, the **stock** (one-click) PaddleOCR module runs OCR on **CPU** —
  upstream doesn't ship a working Linux/CUDA PaddlePaddle path. To run PaddleOCR **on the
  GPU on Linux**, side-load the
  **[PaddleOCR CUDA-12 fork](https://github.com/codeproject/CodeProject.AI-ALPR/pull/25)**
  (a contributed patch that GPU-enables it on Linux/CUDA 12); or use **option C (ALPRFast)**.
  Just don't expect the *stock* one-click module to use an NVIDIA GPU on Linux without that fork.

### A.3 Verify it's serving `/v1/vision/alpr`

From any machine that can reach the server (replace host/port and image path):

```bash
curl -X POST "http://<host>:32168/v1/vision/alpr" \
  -F "image=@/path/to/car.jpg"
```

A healthy response is JSON with `"success": true` and a `predictions` array (each entry a
plate with `label`/`plate`, `confidence`, and a bounding box). First call after install
may be slow while the module warms up.

### A.4 Platform differences

- **Windows / Linux native:** the one-click install above is all you need; the module
  lands in the platform `modules` directory automatically.
- **Docker:** the one-click install writes the module into `/app/modules` inside the
  container. If `/app/modules` is **not** bind-mounted to a host folder, the install is
  **not persistent** — it disappears when the container is recreated. Mount it first
  (see section C's Docker note).

---

## B. ALPR (YOLO11) — Mike Lud's YOLO11 reader

Mike Lud publishes two YOLO11 ALPR modules (from his module manifest), and **both are
Windows-only**:

| Module id | Name | Ver | Platform |
|---|---|---|---|
| **`ALPRYOLO11`** | License Plate Reader (YOLO11-ONNX) | 1.0.4 | **Windows only** |
| **`ALPRYOLO11NET`** | License Plate Recognition (YOLO11 .NET) | 1.1.1 | **Windows only** (.NET 9 + DirectML) |

Both own the same `alpr_queue` / `vision/alpr` route and require **server 2.9.4+**. On
**Linux + NVIDIA** neither is offered — you'd use the CUDA fork (side-load, B.2) or
ALPRFast (section C).

> **How Mike distributes them — this is the actual install (Windows).** His modules are
> **not** in the default CodeProject.AI registry. Instead he ships a **module feed** on his
> own GitHub, and you enable it by editing the server's **`appsettings.json`** so the
> dashboard's **Install modules** menu lists them and downloads them from his GitHub
> releases:
>
> 1. On the CPAI machine, open **`appsettings.json`** (server root, next to the server
>    binary / `server.dll`) and set, under `ModuleOptions`:
>    ```jsonc
>    "ModuleStorageUrl": "https://github.com/MikeLud/MikeLud-CodeProject.AI-Modules/releases/download/2.9.5/",
>    "ModuleListUrl":    "https://github.com/MikeLud/MikeLud-CodeProject.AI-Modules/raw/main/modules/modules.json",
>    ```
>    Keep `"AllowInternetAccess": true`. (For API-based installs, also set an
>    `"InstallPassword"` — it's `null` by default, which blocks them. On Docker, `docker cp`
>    the file into the container / its settings volume, or bind-mount it.)
> 2. **Restart CodeProject.AI** so it re-reads `appsettings.json`.
> 3. Dashboard → **Install modules** → **ALPRYOLO11** now appears → click **Install** (it
>    downloads from his GitHub release and builds the venv).
> 4. Its **AutoStart defaults to `false`** → **Start** it after install (B.4).
>
> This is the "SCP a file onto the CPAI machine" step people describe: the file is
> **`appsettings.json`** (the feed pointer), *not* the module zip — the server then fetches
> the module itself.

### B.0 GPU reality check (read before installing)

- **Windows:** upstream YOLO11 ALPR uses **`onnxruntime-directml`** → GPU via **DirectML**
  on Windows (and MPS on Apple Silicon). This works out of the box on a Windows GPU.
- **Linux + NVIDIA:** plain upstream is **DirectML-only** and will **not** use your NVIDIA
  GPU. To get CUDA on Linux you need the **Linux/CUDA-enabled fork**
  (`MikeLud/CodeProject.AI-ALPR-YOLO11` with the Linux/CUDA patch — it swaps in
  `onnxruntime-gpu`, sets `USE_CUDA=True`/`USE_DIRECTML=False`, and ships a
  `modulesettings.linux.json`). Use that fork's branch/zip when side-loading on Linux.

### B.1 If it's in the dashboard registry

1. Dashboard → **Install modules** → find **"License Plate Recognition (YOLO11)"** →
   **Install**.
2. Note its **AutoStart defaults to `false`**, so after install you may need to **Start**
   it explicitly from the dashboard (see B.4).
3. On Windows, enable GPU in its settings (DirectML). Skip to **B.3 (verify)**.

### B.2 Side-load from GitHub (if not listed, or for the Linux/CUDA fork)

**Where it goes:** a folder named `ALPRYOLO11` inside your platform's `modules` directory
(the prerequisites table). The folder must contain `modulesettings.json`, `install.sh`
(Linux/Docker) / `install.bat` (Windows), the `alpr_adapter.py` entry point, and the
`alpr/` and `models/` code — i.e. the repo contents.

1. **Get the code** (use the Linux/CUDA fork/branch on Linux+NVIDIA):

   ```bash
   git clone https://github.com/MikeLud/CodeProject.AI-ALPR-YOLO11.git ALPRYOLO11
   # (or clone your Linux/CUDA fork/branch instead)
   ```

2. **Put the folder in the modules directory:**

   - **Windows (native):**
     copy the `ALPRYOLO11` folder into `C:\Program Files\CodeProject\AI\modules\`.
   - **Linux (native):**
     copy it into the server's `modules` directory (e.g.
     `.../CodeProject.AI-Server/src/modules/ALPRYOLO11`).
   - **Docker:** drop it into the **host folder mapped to `/app/modules`** (see the Docker
     host-path table) so that inside the container it appears at
     `/app/modules/ALPRYOLO11`. Example on a Linux Docker host:
     ```bash
     git clone https://github.com/MikeLud/CodeProject.AI-ALPR-YOLO11.git \
       /opt/codeproject/ai/ALPRYOLO11
     ```

3. **Run the module's setup** (builds the per-module venv and downloads the ONNX models):

   - **Windows:** from the module folder, run its setup via the server's setup script
     (`...\CodeProject.AI-Server\src\setup.bat`), or trigger install from the dashboard if
     the folder is detected.
   - **Linux native:** from the module folder:
     ```bash
     bash ../../CodeProject.AI-Server/src/setup.sh
     ```
     (the module's own `install.sh` is invoked by that setup script; it must be run *via*
     setup, not directly).
   - **Docker:** the venv must be built **inside the container**. Exec in and run setup,
     e.g.:
     ```bash
     docker exec -it <container> bash
     cd /app/modules/ALPRYOLO11
     bash /app/setup.sh          # path to the server setup script inside the image
     ```
     (Confirm the setup-script path inside your image; on some images it's under
     `/app/`.)

4. **Restart the server** (or the module) so CPAI re-reads the new
   `modulesettings.json`/env. On Docker: `docker restart <container>`.

### B.3 Enable GPU

- YOLO11 ALPR's `GpuOptions` default to `InstallGPU: true` / `EnableGPU: true` with
  `HalfPrecision: enable`.
- **Windows:** GPU acceleration is **DirectML** — it selects it automatically; the
  dashboard's GPU toggle for the module should show it using the GPU.
- **Linux + NVIDIA:** only the **CUDA fork** will use the GPU. Confirm in the module log
  that the ONNX Runtime provider is **`CUDAExecutionProvider`** (see Troubleshooting). If
  you see `CPUExecutionProvider`, you're on the DirectML-only upstream or CUDA didn't load.

### B.4 Start it and verify `/v1/vision/alpr`

Because **AutoStart defaults to `false`**, after install/side-load open the dashboard and
**Start** the ALPRYOLO11 module (and set it to start on boot if you want it persistent).
Then:

```bash
curl -X POST "http://<host>:32168/v1/vision/alpr" \
  -F "image=@/path/to/car.jpg" \
  -F "operation=full"
```

`operation` may be `plate`, `vehicle`, or `full` (default `full`). A healthy response has
`"success": true` and a `plates` array (and `vehicles`/`analysis` for `full`).

### B.5 Platform differences (summary)

| | Windows | Linux native | Docker |
|---|---|---|---|
| Registry one-click | if listed | if listed | if listed (into `/app/modules`) |
| Side-load target | `C:\Program Files\CodeProject\AI\modules\ALPRYOLO11` | `.../src/modules/ALPRYOLO11` | host folder mapped to `/app/modules` → `/app/modules/ALPRYOLO11` |
| GPU | DirectML (works) | **CUDA fork only** | CUDA fork + `--gpus all` + CUDA image |
| venv builds | on host | on host | **inside the container** |

---

## C. ALPRFast — third-party `fast-alpr`, Linux + CUDA (manual only)

Module id **`ALPRFast`** (GitHub: **`chsbusch-dot/CodeProject.AI-ALPRFast`**). It packages
[`fast-alpr`](https://github.com/ankandrew/fast-alpr) — a **YOLOv9** plate detector plus a
**CCT** (`fast-plate-ocr`) OCR model, all ONNX / `onnxruntime-gpu` — and adds a CCTV
accuracy stack (crop-first vehicle detection, Lanczos super-resolution, US/CA format +
confidence gating, and multi-frame voting). It serves the same `vision/alpr` route on
`alpr_queue`.

**It is NOT in the CPAI registry — install is manual/side-load only.**

### C.0 Requirements

- **Linux** (`linux` or `linux-arm64`) with a working **NVIDIA driver** (CUDA 12).
- **CodeProject.AI Server 2.9+** (2.9.7 is fine).
- **An object-detection module** running on the same server (e.g. ObjectDetectionYOLOv5) —
  ALPRFast's crop-first step calls it over HTTP. Without one, ALPRFast still runs
  full-frame-only but misses small/distant plates.
- CUDA is bundled into the module's venv: setup installs
  `fast-alpr[onnx-gpu]` and **`onnxruntime-gpu[cuda,cudnn]==1.23.0`**, whose extras pull
  the matching **CUDA 12 runtime + cuDNN 9** into the venv — so you do **not** need a
  system-wide CUDA install, only the driver.

### C.1 Get the module folder

```bash
git clone https://github.com/chsbusch-dot/CodeProject.AI-ALPRFast.git ALPRFast
```

The folder must contain (it does, out of the box): `modulesettings.json`, `install.sh`,
`alprfast_adapter.py`, the `alprfast/` package, and `requirements.linux.txt`.

### C.2 Put it in the modules directory

- **Linux native:** copy `ALPRFast` into the server's `modules` directory, e.g.
  `.../CodeProject.AI-Server/src/modules/ALPRFast`.
- **Docker (the tested path):** get the folder to **`/app/modules/ALPRFast`** inside the
  container by placing it in the **host folder mapped to `/app/modules`**. On a Linux
  Docker host that default mapping is under `/opt/codeproject/ai`:
  ```bash
  git clone https://github.com/chsbusch-dot/CodeProject.AI-ALPRFast.git \
    /opt/codeproject/ai/ALPRFast
  # -> visible inside the container as /app/modules/ALPRFast
  ```

  > **Docker note — mount `/app/modules` first.** If your container was started without a
  > host mount for `/app/modules`, add one, or the module (and its venv) is lost on
  > recreate. A GPU-enabled run looks like:
  > ```bash
  > docker run -d --name codeproject-ai --gpus all -p 32168:32168 \
  >   --mount type=bind,source=/opt/codeproject/ai/data,target=/etc/codeproject/ai \
  >   --mount type=bind,source=/opt/codeproject/ai/modules,target=/app/modules \
  >   codeproject/ai-server:cuda12
  > ```
  > (Match the image tag to your CUDA/driver; `--gpus all` requires the NVIDIA Container
  > Toolkit on the host.)

### C.3 Run setup (builds the venv, wires up model download)

The venv must be built where the server runs — **inside the container** on Docker.

- **Linux native**, from the module folder:
  ```bash
  bash ../../CodeProject.AI-Server/src/setup.sh
  ```
- **Docker:**
  ```bash
  docker exec -it codeproject-ai bash
  cd /app/modules/ALPRFast
  bash /app/setup.sh        # server setup script inside the image; confirm the path
  ```

Setup installs the Python deps into `ALPRFast/bin/<os>/<pyver>/venv` and (on bare-metal
Linux) the OpenCV system libs. **Model weights are not bundled** — `fast-alpr` downloads
the YOLOv9 detector and CCT OCR model **from HuggingFace on first use** and caches them
under `~/.cache`, so the **first `/v1/vision/alpr` request after install is slower** while
models download. (Ensure the container/host has outbound network access on first run.)

### C.4 Enable GPU

- ALPRFast ships `GpuOptions`: `InstallGPU: true`, `EnableGPU: true`,
  `HalfPrecision: enable`, and env `USE_CUDA: "True"`.
- It calls **`ort.preload_dlls()`** at startup so ONNX Runtime's bundled CUDA/cuDNN load
  correctly — without that, ORT ≥1.19 silently falls back to CPU. You don't have to do
  anything; just verify the provider list (next).

### C.5 Start and verify `/v1/vision/alpr`

Start the **ALPRFast** module from the dashboard (its `AutoStart` is `true` in the shipped
`modulesettings.json`, but confirm it's **Started**), then:

```bash
curl -X POST "http://<host>:32168/v1/vision/alpr" \
  -F "image=@/path/to/car.jpg" \
  -F "min_confidence=0.4"
```

Healthy response: `"success": true`, a `predictions` array (each: `label`/plate text,
`confidence`, bbox `x_min/y_min/x_max/y_max`, and `votes`), plus `count`, `inferenceMs`,
`processMs`.

**Confirm it's on the GPU:** open the ALPRFast **module log** in the dashboard and check
the ONNX Runtime provider list shows **`CUDAExecutionProvider`** (not just
`CPUExecutionProvider`). See Troubleshooting.

### C.6 Key config (env vars, editable in the dashboard module settings)

| Var | Default | Purpose |
|---|---|---|
| `USE_CUDA` | `True` | Use GPU (`CUDAExecutionProvider`); `False` forces CPU |
| `PLATE_DETECTOR_CONFIDENCE` | `0.35` | Min detector confidence for a plate box |
| `MIN_CHAR_CONFIDENCE` | `0.55` | Per-character OCR floor (junk reads rejected) |
| `ENABLE_VEHICLE_CROP_FALLBACK` | `True` | Crop-first on a full-frame miss |
| `OBJECT_DETECTION_URL` | `http://localhost:32168/v1/vision/detection` | The object-detection endpoint crop-first calls |
| `MAX_VEHICLE_CROPS` | `3` | Cap on vehicle crops re-run per frame |
| `ENABLE_VOTING` / `VOTE_WINDOW_SECS` | `True` / `8.0` | Multi-frame majority voting window |
| `SR_TARGET_HEIGHT` | `64` | Lanczos-upscale small plate crops to this height |
| `SAVE_CAPTURES` / `CAPTURE_DIR` | `True` / `/app/modules/ALPRFast/captures` | Debug capture harness — **the shipped file sets `True`; set `False` for production** to avoid heavy disk use |

> The repo's `modulesettings.json` ships with `SAVE_CAPTURES: True` (dev corpus building).
> For production set it to `False` — then restart the module (see the AutoStart caveat in
> Troubleshooting when you edit `modulesettings.json`).

---

## Switching readers / avoiding the two-ALPR conflict

Because all ALPR modules share `/v1/vision/alpr` + `alpr_queue`, **run exactly one**. To
switch from reader X to reader Y:

1. **Stop / disable the old reader.** In the dashboard, **Stop** the currently-running
   ALPR module and turn **off** its start-on-boot. To be safe (and to avoid it silently
   re-registering on the queue), **uninstall** it or set its `AutoStart` to `false`.
   - Symptom you're avoiding: two ALPR modules both **Started** → CPAI **round-robins**
     `/v1/vision/alpr` between them → intermittently wrong/garbage plate reads that look
     like a flaky model. If reads are inconsistent, **check the dashboard for a second
     running ALPR module first.**
2. **Install/enable the new reader** (sections A/B/C) and confirm only it is **Started**.
3. **Restart CodeProject.AI** so the route/queue ownership is clean
   (Docker: `docker restart <container>`).
4. **Then restart Blue Iris** (next section) — **BI drops ALPR after any CPAI restart and
   only re-arms it after a Blue Iris restart.**

At the end, the dashboard should show **exactly one** module bound to the ALPR route.

---

## Point Blue Iris at it

No client change is needed when you swap readers — Blue Iris just talks to whatever module
owns `/v1/vision/alpr`:

1. In Blue Iris, set the **CodeProject.AI / AI server** to your CPAI host and port
   (default `http://<host>:32168`).
2. **Enable ALPR** on the camera(s) you want. BI posts to `/v1/vision/alpr`, which your
   one installed ALPR module now serves.
3. **After any CPAI restart, restart Blue Iris** to re-arm ALPR (BI drops ALPR on a CPAI
   restart and only reconnects it on a BI restart).

Optional (for wide overview cameras, esp. with ALPRFast): in BI's **Alert Confirmation**
settings, enable pre/post-trigger images and "analyze one image each" so more frames reach
the ALPR endpoint — more frames in the window means better multi-frame voting.

---

## Troubleshooting

> **CUDA silently running on CPU (slow reads, no error).**
> ONNX-Runtime GPU builds ≥1.19 ship CUDA/cuDNN as pip extras the Linux loader does **not**
> auto-find; without preloading, ORT falls back to `CPUExecutionProvider` — full speed, no
> error, just CPU. ALPRFast calls `ort.preload_dlls()` to prevent this.
> **Diagnose:** open the module's **log** in the dashboard and read the **provider list**.
> You want **`CUDAExecutionProvider`**; if you only see `CPUExecutionProvider`, then either
> the NVIDIA driver isn't healthy, the container wasn't started with `--gpus all`
> (Docker), or (YOLO11) you're on the DirectML-only upstream instead of the CUDA fork.

> **First request is slow / "no models" on a fresh ALPRFast (or model-downloading modules).**
> `fast-alpr` downloads its YOLOv9 + CCT models from **HuggingFace on first use** and caches
> them under `~/.cache`. The first `/v1/vision/alpr` call after install pays that download
> cost. Ensure the host/container has **outbound internet** on first run; subsequent calls
> are fast.

> **AutoStart reset / env changes ignored after editing `modulesettings.json`.**
> Editing or replacing a module's `modulesettings.json` can **reset `AutoStart` to the
> file default**, and CPAI won't pick up new settings/env until the module (or the server)
> is **restarted**. Also: CPAI persists an **enabled/disabled** state you chose in the
> dashboard that can **override** the file's `AutoStart`. After editing the file: restart
> the module (or `docker restart <container>`), then re-check in the dashboard that it's
> **Started** and set to start on boot if you want it persistent.

> **Docker: module disappears after a container recreate.**
> The module lives in `/app/modules` inside the container. If that isn't bind-mounted to a
> host folder, it's ephemeral. Mount `/app/modules` to a host directory (host-path table
> in Prerequisites) and re-side-load there.

> **Two ALPR modules installed → intermittent wrong reads.**
> See ["Switching readers"](#switching-readers--avoiding-the-two-alpr-conflict). Only one
> ALPR module may be **Started** at a time.

> **Stock ALPR won't use the GPU on Linux.**
> Expected — the stock PaddleOCR module has no working Linux/CUDA path (`InstallGPU: false`
> by default; GPU is effectively Windows-only for it). Use **ALPRFast** (or a Linux/CUDA
> PaddleOCR fork) for GPU on Linux + NVIDIA.

---

### Sources

- [CodeProject.AI Server docs (v2.9.5)](https://codeproject.github.io/codeproject.ai/)
- [Using Docker — CodeProject.AI Server (v2.9.5)](https://codeproject.github.io/codeproject.ai/faq/docker.html)
- [Installation Guide — DeepWiki](https://deepwiki.com/codeproject/CodeProject.AI-Server/3-installation-guide)
- [Adding a New Module to CodeProject.AI Server](https://www.codeproject.com/Articles/5332075/Adding-a-New-Module-to-CodeProject-AI-Server)
- [CodeProject.AI-ALPR (stock module) — GitHub](https://github.com/codeproject/CodeProject.AI-ALPR)
- Module ground truth: local `CodeProject.AI-ALPR`, `CodeProject.AI-ALPR-YOLO11`, and `CodeProject.AI-ALPRFast` repos (`modulesettings.json`, `install.sh`, adapters, READMEs).
