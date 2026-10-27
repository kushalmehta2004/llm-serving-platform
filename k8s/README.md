# K8s Deploy — LLM Serving Platform (Phase 7)

## Quick start (kind / k3d — CPU, local)

```bash
# 1. Create a local cluster
kind create cluster --name llm-serving
# or: k3d cluster create llm-serving

# 2. Build and load the image
docker build -t llm-serving-platform:latest .
kind load docker-image llm-serving-platform:latest --name llm-serving

# 3. Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Watch startup (model load takes ~30s–120s)
kubectl get pods -n llm-serving -w

# 5. Forward a port and send a request
kubectl port-forward svc/llm-server-nodeport 8000:8000 -n llm-serving &
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'
```

## The readiness-probe war story

**Problem (break it on purpose):** Set `startupProbe.failureThreshold: 1` and
`periodSeconds: 5`. The kubelet fires the first probe after 5 s — the model is
still loading — gets HTTP 500/no-response, counts a failure, restarts the pod.
Repeat → `CrashLoopBackOff`.

```
NAME           READY   STATUS             RESTARTS
llm-server-x   0/1     CrashLoopBackOff   5
```

**Why it happens:** the model download + weight materialisation takes 30–120 s
on a cold node. Any probe with a total budget shorter than that will lose.

**Fix:** use a `startupProbe` (available since K8s 1.18) with a generous
`failureThreshold × periodSeconds` budget (we use 30 × 10 s = 300 s).
The `startupProbe` blocks `livenessProbe` and `readinessProbe` from running
until it first succeeds, so a single well-tuned startup probe is all that's
needed.

## GPU cluster

1. Uncomment `nvidia.com/gpu: "1"` in `deployment.yaml` limits.
2. Uncomment the `nodeSelector` / `tolerations` stanzas.
3. Ensure the NVIDIA device plugin DaemonSet is installed on the cluster.
4. `kubectl apply -f k8s/` as above.

## Observability in K8s

Deploy Prometheus + Grafana via Helm:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prom prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

The pod annotations in `deployment.yaml` (`prometheus.io/scrape: "true"`) let
the default Prometheus scrape config pick up the `/metrics` endpoint
automatically.
