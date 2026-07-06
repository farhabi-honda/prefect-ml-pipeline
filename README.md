# prefect-ml-training

Prefect-based baseline for training multiple ML models, triggered via a
REST API call. First model wired up: **segmentation-train** (mmsegmentation).
Runs locally now; the same image/deployment pattern carries over to AWS
(ECS/EC2) later with minimal changes.

---

## 1. Repo structure

```
.
├── Dockerfile                     # builds segenv + flowenv into a single image
├── setup_env/
│   ├── install_mmseg_cu124/cu118.sh     # bakes mmsegmentation + CUDA 12.4 / CUDA 11.8 torch into segenv
│   └── requirements.txt           # orchestration deps (Prefect, FastAPI, boto3/awscli) into flowenv
├── docker-compose.yml             # local: prefect server + worker + trigger API
├── prefect.yaml                   # deployment definitions
├── flows/
│   ├── common.py                  # clone helper, S3 sync helpers, PrebuiltEnv (segenv/flowenv reference)
│   ├── segmentation_train.py      # segmentation model flow
│   └── detection3d_train.py       # 3D detection model flow (needs a detenv — not built yet)
├── api/
│   └── trigger.py                 # FastAPI: POST /trigger/{model_name}
└── .env.example                   # proxy, AWS creds, Prefect API URL
```

### What lives where

| Path | Purpose | Changes when... |
|---|---|---|
| `Dockerfile` | Builds two venvs into one image: `segenv` (mmsegmentation + CUDA-matched torch/mmcv) and `flowenv` (Prefect/FastAPI orchestration) | you add a new framework/model, or bump system packages |
| `setup_env/install_mmseg_cu124/cu118.sh` | The actual mmsegmentation + torch + mmcv install, baked in at **build time** | mmsegmentation/torch/mmcv versions change |
| `setup_env/requirements.txt` | flowenv deps only — never model deps | Prefect/FastAPI/boto3 versions change |
| `flows/*.py` | One file per model framework. Clones the lightweight **model repo** (configs/training code) at trigger time and runs it against the pre-baked env | you add a new model, or change how a model's training is invoked |
| `flows/common.py` | Shared: `clone_or_update_repo`, `s3_sync_down`/`s3_sync_up`, `PrebuiltEnv` | shared plumbing changes (rare) |
| `prefect.yaml` | Registers each flow as a deployment against a work pool | you add a new model or change work pool config |
| `api/trigger.py` | The REST surface — `POST /trigger/{model_name}` maps to a deployment via `MODEL_DEPLOYMENTS` | you add a new model (one dict entry) |
| `docker-compose.yml` | Local orchestration: Prefect server + worker + trigger API, volumes, proxy/AWS env | local infra changes |

### Architecture in one paragraph

Framework environments (`segenv`, `flowenv`) are installed **once, at image
build time** — not at trigger time. A triggered run only has to (1) get the
lightweight model repo's code (clone, bind-mount, or use what's already
baked in — see options below), optionally (2) sync training data down from
S3, (3) run training against the pre-built env, and (4) sync outputs back
up to S3. Nothing gets `pip install`ed mid-trigger — that's what makes
triggers fast and removes "broken install mid-run" as a failure mode.

---

## 2. Deploying (building + starting everything)

```bash
cp .env.example .env
```

Fill in `.env`:
- `http_proxy` / `https_proxy` / `no_proxy` — only if your network requires a proxy. **Important**: also list internal Docker service names (`prefect-server`, `prefect-worker`, `trigger-api`, `localhost`, `127.0.0.1`) in `no_proxy`/`NO_PROXY`, or container-to-container calls get routed through the external proxy and fail (`RemoteProtocolError: Server disconnected...`).
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` — only for local S3 testing. On EC2/ECS, use an IAM instance/task role instead and leave these blank.

Build and start everything:

```bash
docker compose up --build
```

First build takes a while — `segenv` installs torch + compiles/downloads
mmcv + clones mmsegmentation. Subsequent builds are cached unless
`setup_env/` changes.

Once containers are healthy (the worker waits on `prefect-server`'s
healthcheck before starting), register the deployments:

```bash
docker compose exec trigger-api prefect deploy --all
```

Check everything's up:
- Prefect UI: http://localhost:4200
- Trigger API health + registered model names: `curl http://localhost:8000/health`

---

## 3. How to run each option

All triggers go through the same endpoint:

```
POST http://localhost:8000/trigger/{model_name}
```

Currently registered `model_name`s: `segmentation-train`, `detection3d-train`
(the latter needs a `detenv` + install script added before it'll actually run).

### Option A — Production path: clone a real model repo

Default behavior. Clones `repo_url` fresh (or pulls latest if already
cloned), installs any extra `requirements.txt` the model repo ships, then
trains.

```bash
curl -X POST http://localhost:8000/trigger/segmentation-train \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/YOUR_ORG/segmentation-train.git",
    "config_path": "configs/my_segmentation_config.py"
  }'
```

### Option B — Test mode: use the mmsegmentation baked into the image

Skips cloning entirely, trains directly against `/app/mmsegmentation` (the
copy installed by `setup_env/install_mmseg_cu124/cu118.sh` at build time). Good
for confirming the trigger → flow run → training pipeline works before you
have a real model repo.

```bash
curl -X POST http://localhost:8000/trigger/segmentation-train \
  -H "Content-Type: application/json" \
  -d '{
    "use_baked_repo": true,
    "config_path": "configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
  }'
```

### Option C — Local dev mode: bind-mount a repo from your host

For iterating on configs/code on your host machine without cloning or
rebuilding the image. Requires a bind mount in `docker-compose.yml`:

```yaml
  prefect-worker:
    volumes:
      - /path/on/your/host/mmsegmentation:/workspace/local-repo
```

Then trigger with `local_repo_path` pointing at the mounted path:

```bash
curl -X POST http://localhost:8000/trigger/segmentation-train \
  -H "Content-Type: application/json" \
  -d '{
    "local_repo_path": "/workspace/local-repo",
    "config_path": "configs/your_new_config.py"
  }'
```

Edits to files on your host show up immediately on the next trigger — no
restart needed (bind mounts are live, not baked in).

**Priority if you pass more than one of these**: `local_repo_path` wins
over `use_baked_repo`, which wins over the default clone — see
`segmentation_train_flow`'s parameter checks in `flows/segmentation_train.py`.

### Adding S3 data sync (any of the above options)

Add `data_s3_uri` / `artifacts_s3_uri` to sync training data down before,
and checkpoints/logs up after:

```bash
curl -X POST http://localhost:8000/trigger/segmentation-train \
  -H "Content-Type: application/json" \
  -d '{
    "use_baked_repo": true,
    "config_path": "configs/my_config.py",
    "data_s3_uri": "s3://my-bucket/datasets/cityscapes/",
    "artifacts_s3_uri": "s3://my-bucket/runs/segmentation/run-001/"
  }'
```

Omit either field to skip that step. Sync is idempotent (`aws s3 sync`), so
repeat triggers against the same `data_s3_uri` only transfer new/changed
files — the local dataset cache lives on the `training-datasets` volume.
Your config's `data_root` needs to point at wherever the data actually
lands locally (`dataset_dir`, default `/workspace/datasets/segmentation`).

---

## 4. Adding the next model (e.g. `detection3d-train`)

1. Add its framework env to the Dockerfile (`python3 -m venv /app/detenv`)
   and write `setup_env/install_mmdet3d_cu124/cu118.sh` (pin exact versions with
   confirmed prebuilt wheels — see troubleshooting below), rebuild the image.
2. `flows/detection3d_train.py` already has the flow-side pattern ready —
   point its defaults at your real model repo.
3. `detection3d-train-local` deployment already exists in `prefect.yaml`.
4. Confirm `MODEL_DEPLOYMENTS` in `api/trigger.py` has an entry for it — no
   new endpoint needed, `POST /trigger/{model_name}` already routes by name.
5. `docker compose exec trigger-api prefect deploy --all` to register it.

---

## 5. Troubleshooting (issues actually hit while building this)

**`ModuleNotFoundError: No module named 'griffe.dataclasses'`**
`griffe` released a `1.0.0` that reorganized a module Prefect 2.x still
imports from. Fix: pin `griffe<1.0.0` in `setup_env/requirements.txt`
(already applied) and rebuild.

**`No such option: --overwrite` on `prefect work-pool create`**
That flag doesn't exist in this Prefect version's CLI. Fix: let creation
fail harmlessly if the pool already exists —
`(prefect work-pool create local-process-pool --type process || true)`
— instead of trying to force-overwrite it.

**`502 Bad Gateway` / worker crashes on startup**
`depends_on` alone only waits for the `prefect-server` **container** to
start, not for the API inside it to be ready. Fix: a real `healthcheck` on
`prefect-server` (`python -c "urllib.request..."` — no extra package needed)
and `depends_on: prefect-server: condition: service_healthy` on both
`prefect-worker` and `trigger-api`.

**`httpcore.RemoteProtocolError: Server disconnected without sending a response`**
Your corporate proxy is intercepting internal Docker network traffic —
`http://prefect-server:4200` doesn't mean anything to an external proxy.
Fix: explicitly set `no_proxy`/`NO_PROXY` at **container runtime** (not just
build time) including all internal service hostnames:
`localhost,127.0.0.1,prefect-server,prefect-worker,trigger-api`.

**`mmcv` build fails with `Python.h: No such file or directory`**
mmcv couldn't find a prebuilt wheel for your exact torch/CUDA combo and
fell back to compiling from source, which then fails without a full
compiler toolchain. Two possible causes/fixes:
- An unpinned version **range** (e.g. `mmcv>=2.1.0`) resolved to a release
  with no wheel for your combo — pin the **exact** version instead (this
  repo pins `mmcv==2.1.0` for `cu124/cu118`/`torch==2.4.1`).
- If no version of mmcv has a wheel for your specific combo at all, you'd
  need a `devel`-stage compile (nvcc + headers) instead — a heavier,
  multi-stage-build fix; worth confirming the exact-pin fix first since
  it's usually sufficient.

**Proxy build args seem to do nothing (Dockerfile `ARG` stays empty)**
`.bashrc` and `.env` don't automatically reach `docker build` — only
`--build-arg` (CLI) or an explicit `args:` block under `build:` in
`docker-compose.yml` does. This repo's `docker-compose.yml` already has
that `args:` block wired to read from `.env`.

---

## 6. Moving to AWS production (later)

- The baked-env pattern maps directly onto per-model **ECR images**: build
  and push this image (or one per framework, if they diverge further) to
  ECR, then reference it via `job_variables.image` in `prefect.yaml` with a
  `docker`/`ecs` work pool.
- Point `PREFECT_API_URL` at Prefect Cloud or a self-hosted server on AWS.
- Move the trigger API behind API Gateway (Lambda + Mangum, or a small
  Fargate service) — the `run_deployment()` call itself doesn't change.
- For private model repos, add a GitHub token as a Prefect Secret block and
  pass it to `git clone` in `flows/common.py`'s `clone_or_update_repo`.
- Credentials: don't bake or pass static AWS keys in production — attach an
  IAM instance role (EC2) or task role (ECS Fargate) scoped to your
  buckets; the `aws` CLI picks it up automatically.
- CUDA note: this is a single-stage `runtime` image (no `nvcc`/devel
  toolchain) — fine as long as every install resolves to a prebuilt wheel.
  If a future framework's install ever falls back to compiling from source
  with no wheel available for any version, that's when a `devel`-stage +
  multi-stage copy-in becomes necessary.
- Consider splitting `prefect-server` onto the official lightweight
  `prefecthq/prefect` image instead of this repo's full custom image — the
  server never touches mmsegmentation/CUDA, so it doesn't need those layers.