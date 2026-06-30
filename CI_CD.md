# CI/CD Guide

This project uses GitHub Actions for continuous integration and Docker image
publishing. The published image can be used directly as a RunPod custom image.

## What CI/CD Does

The CI/CD setup has two workflows:

```text
.github/workflows/ci.yml
.github/workflows/docker-publish.yml
```

`ci.yml` checks that the project is healthy:

```bash
python -m py_compile ...
pytest -q
```

`docker-publish.yml` builds and pushes a Docker image to GitHub Container
Registry.

The image name will be:

```text
ghcr.io/<github-owner>/<repo-name>:latest
ghcr.io/<github-owner>/<repo-name>:<commit-sha>
```

## Required Files

The CI/CD setup uses:

```text
Dockerfile
.dockerignore
requirements.txt
scripts/ci_checks.sh
scripts/start_api.sh
scripts/runpod_smoke_test.sh
.github/workflows/ci.yml
.github/workflows/docker-publish.yml
```

## First-Time GitHub Setup

Create a GitHub repository and push this project:

```bash
git init
git add .
git commit -m "Initial wifi log analyzer"
git branch -M main
git remote add origin https://github.com/<github-owner>/<repo-name>.git
git push -u origin main
```

Then open the GitHub repository:

1. Go to **Actions**.
2. Enable workflows if GitHub asks.
3. Open the **CI** workflow.
4. Confirm tests pass.
5. Open the **Build RunPod Image** workflow.
6. Confirm the Docker image is published.

No extra GitHub secret is required for GHCR publishing. The workflow uses the
built-in `GITHUB_TOKEN`.

## GitHub Actions Secrets

Add secrets in GitHub only when a workflow needs to use that value. The current
CI and Docker workflows do not need `GROQ_API_KEY`, because tests mock Groq
calls and the Docker image should not contain API keys.

To add a secret in GitHub:

1. Open your GitHub repository.
2. Go to **Settings**.
3. Go to **Secrets and variables**.
4. Choose **Actions**.
5. Click **New repository secret**.
6. Add the name and value.

Useful names:

```text
GROQ_API_KEY
RUNPOD_API_KEY
```

`GROQ_API_KEY` is only needed in GitHub Actions if you later add an integration
test or deployment step that calls Groq from the workflow.

Example workflow usage:

```yaml
env:
  GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

Do not pass `GROQ_API_KEY` as a Docker build argument or write it into the image.
The Docker image may be reused, cached, pushed to GHCR, and inspected later.

## GitHub Actions Node24 Warning

GitHub Actions runners use Node24 by default beginning June 16, 2026. This
project uses Node24-compatible action versions:

```text
actions/checkout@v6
actions/setup-python@v6
docker/setup-buildx-action@v4
docker/login-action@v4
docker/build-push-action@v7
```

Do not set `ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true` unless you are forced
to run an old action temporarily. For GitHub-hosted runners such as
`ubuntu-latest`, no extra runner setup is required. For self-hosted runners,
upgrade the runner to `v2.327.1` or newer.

## Docker Image

The Docker image starts the API automatically:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Inside the Docker image, application code lives in:

```text
/app
```

Runtime files live in the mounted RunPod workspace:

```text
/workspace/wifi-log-analyzer
```

Keep these separate. If the app code is copied under `/workspace`, a RunPod
network volume mounted at `/workspace` can hide the code and cause
`ModuleNotFoundError: No module named 'src'`.

The Docker image also copies bundled `data/` and `models/` into `/app`, so the
API can run with files already present in the container. Generated outputs still
belong in `/workspace/wifi-log-analyzer`.

Readable input paths are checked under the workspace first and then under the
app directory:

```text
/workspace/wifi-log-analyzer/data/...
/app/data/...
```

For FLAN-T5 inference, relative `model_dir` values are checked under the
workspace first and then under the app directory:

```text
/workspace/wifi-log-analyzer/models/...
/app/models/...
```

It exposes:

```text
8000
```

The image intentionally does not include heavy runtime data:

```text
data/
models/
outputs/
*.pcap
*.pcapng
*.jsonl
*.log
```

Upload or mount those files on the RunPod volume.

## RunPod Deployment With CI/CD Image

After the Docker workflow succeeds, use this image in RunPod:

```text
ghcr.io/<github-owner>/<repo-name>:latest
```

In RunPod:

1. Create a GPU Pod or a Serverless load-balancing endpoint.
2. Choose **Custom Image**.
3. Paste the GHCR image name.
4. Expose HTTP port `80` for the Docker image.
5. Mount a persistent volume at `/workspace`.
6. Set environment variables.

Recommended environment variables:

```bash
WIFI_ANALYZER_WORKSPACE=/workspace/Wifi_log_analyzer
WIFI_ANALYZER_JOB_WORKERS=1
GROQ_API_KEY=<your_groq_api_key>
```

GitHub Actions secrets are not automatically available inside RunPod. Add
`GROQ_API_KEY` separately in the RunPod Pod/template environment variables or
RunPod secrets. That is what makes `/diagnosis/groq` and `/pipeline/groq` work
at runtime.

If the GHCR image is private, make the package public or configure RunPod with
GitHub Container Registry credentials.

## Runtime Data On RunPod

Because the Docker image excludes large files, place these under the persistent
volume:

```text
/workspace/wifi-log-analyzer/data/
/workspace/wifi-log-analyzer/models/
/workspace/wifi-log-analyzer/outputs/
```

For example:

```text
/workspace/wifi-log-analyzer/models/flan-t5-log-lora-model
/workspace/wifi-log-analyzer/data/samples/wifi_events_3600.txt
/workspace/wifi-log-analyzer/data/samples/wifi_events_3600.pcap
```

## Smoke Test On RunPod

After the Pod starts, open a terminal and run:

```bash
cd /app
bash scripts/runpod_smoke_test.sh
```

If you are using the Docker image and only mounted data/models, the source code
is already inside the image at `/app`. If you cloned the repo manually instead,
run the checks from that cloned repo directory:

```bash
pip install -r requirements.txt
pytest -q
```

## API Health Check

After starting the Pod, open:

```text
https://<your-runpod-url>/docs
```

Or check health:

```bash
curl https://<your-runpod-url>/health
```

Expected response:

```json
{
  "status": "ok",
  "workspace_root": "/workspace/wifi-log-analyzer"
}
```

## Deploy Flow

Use this flow when you change code:

1. Edit code locally.
2. Commit changes.
3. Push to GitHub.
4. Wait for **CI** to pass.
5. Wait for **Build RunPod Image** to publish.
6. Restart or recreate the RunPod Pod using:

```text
ghcr.io/<github-owner>/<repo-name>:latest
```

## Manual Local Commands

Run checks locally:

```bash
bash scripts/ci_checks.sh
```

Start API locally:

```bash
export WIFI_ANALYZER_WORKSPACE="$(pwd)"
export WIFI_ANALYZER_JOB_WORKERS=1
bash scripts/start_api.sh
```

Then open:

```text
http://localhost:8000/docs
```

## Troubleshooting

If CI fails while importing ML libraries, check:

```bash
pip install -r requirements.txt
```

If the Docker image is too large, confirm `.dockerignore` is excluding:

```text
data/
models/
outputs/
```

If RunPod cannot pull the image:

1. Check that the image exists in GitHub Packages.
2. Make the package public, or configure GHCR credentials in RunPod.
3. Use the exact lowercase image name:

```text
ghcr.io/<github-owner>/<repo-name>:latest
```

If API calls take too long, use background job endpoints:

```text
POST /jobs/pipeline/groq
GET  /jobs/{job_id}
```
