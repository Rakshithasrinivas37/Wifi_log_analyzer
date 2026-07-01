# Google Kubernetes Engine Cluster Run

Use this guide to run the WiFi analyzer as a two-pod Kubernetes batch job on a
two-node Google Kubernetes Engine cluster.

The default job runs:

```text
Pod index 0 -> /app/data/inputs/wifi_logs.txt
Pod index 1 -> /app/data/inputs/wifi_logs-1.txt
```

Each pod writes its own inference/PCAP output under its local `/workspace` and
prints a preview to pod logs. This avoids shared-disk setup for the basic
cluster experiment. Use Filestore/ReadWriteMany storage later if you need a
single merged `final_results.jsonl` across nodes.

## 1. Open Google Cloud Shell

Open Google Cloud Console, then click **Activate Cloud Shell**.

Set your project:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable container.googleapis.com compute.googleapis.com
```

## 2. Create A Two-Node CPU Cluster

```bash
gcloud container clusters create wifi-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-standard-2 \
  --release-channel regular
```

Connect `kubectl`:

```bash
gcloud container clusters get-credentials wifi-cluster \
  --zone us-central1-a
```

Check nodes:

```bash
kubectl get nodes -o wide
```

## 3. Create Namespace

```bash
kubectl create namespace wifi-analyzer
kubectl config set-context --current --namespace=wifi-analyzer
```

## 4. Allow GKE To Pull The Image

If your GHCR image is public, skip this step.

For a private GHCR image:

```bash
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=Rakshithasrinivas37 \
  --docker-password=YOUR_GITHUB_PAT_WITH_read_packages
```

Then add this under `spec.template.spec` in the job YAML:

```yaml
imagePullSecrets:
  - name: ghcr-secret
```

## 5. Run The Two-Pod Project Job

The CPU manifest is:

```text
k8s/gke-wifi-cluster-job.yaml
```

Apply it:

```bash
kubectl apply -f k8s/gke-wifi-cluster-job.yaml
```

Watch placement:

```bash
kubectl get pods -o wide -w
```

The manifest uses pod anti-affinity, so on a two-node cluster you should see
one pod on each node.

## 6. Read Results

List pods:

```bash
kubectl get pods -o wide
```

Read each pod log:

```bash
kubectl logs POD_NAME
```

Or read the job logs:

```bash
kubectl logs job/wifi-cluster-inference
```

Look for:

```text
[Node 0/2] WiFi analyzer cluster worker
Log file      : /app/data/inputs/wifi_logs.txt

[Node 1/2] WiFi analyzer cluster worker
Log file      : /app/data/inputs/wifi_logs-1.txt
```

## 7. Optional: Run On GPU Nodes

Create a GPU node pool:

```bash
gcloud container node-pools create gpu-pool \
  --cluster wifi-cluster \
  --zone us-central1-a \
  --machine-type g2-standard-4 \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=default \
  --num-nodes 2
```

Check GPU resources:

```bash
kubectl describe nodes | grep -i "nvidia.com/gpu"
```

Run the GPU manifest:

```bash
kubectl delete job wifi-cluster-inference --ignore-not-found
kubectl apply -f k8s/gke-wifi-cluster-job-gpu.yaml
```

## 8. Cleanup

Delete the job:

```bash
kubectl delete job wifi-cluster-inference --ignore-not-found
```

Delete the cluster when done to stop billing:

```bash
gcloud container clusters delete wifi-cluster \
  --zone us-central1-a
```
