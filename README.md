# Agentic AI DT

A Python project for agentic AI-driven xApp orchestration.

Further [documentation](docs/index.md).


## Project Structure

```
agentic-ai-dt/
├── main.py                # Entry point for running the pipeline
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── modules/               # Core modules (bandit, rag_chain, evaluation, etc.)
├── db/                    # Database files (ChromaDB, etc.)
├── materials/             # Source documents and deliverables
├── pipeline_blacklist.json# Blacklist for pipeline filtering
├── run_metrics.jsonl      # Metrics output
├── README.md              # Project documentation
```

## Getting Started

1. **Clone the repository:**
	```bash
	git clone https://github.com/hpn-bristol/agentic-ai-dt.git
	cd agentic-ai-dt
	```
2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

3. **Set up a k3s Kubernetes cluster:**
	```bash
	curl -sfL https://get.k3s.io | sh -
	# Additional setup may be required for kubectl configuration
	```

4. **Install Argo Workflows using Helm:**
	```bash
	# Add the Argo Helm repository
	helm repo add argo https://argoproj.github.io/argo-helm
	helm repo update

	# Install Argo Workflows in your preferred namespace
	helm install argo argo/argo-workflows --namespace kube-system
	```

5. **Install the Argo CLI:**
	```bash
	# Download the latest version of the Argo CLI
	VERSION=$(curl --silent "https://api.github.com/repos/argoproj/argo-workflows/releases/latest" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
	curl -sLO https://github.com/argoproj/argo-workflows/releases/download/$VERSION/argo-linux-amd64.gz
	gunzip argo-linux-amd64.gz
	chmod +x argo-linux-amd64
	sudo mv ./argo-linux-amd64 /usr/local/bin/argo
	```

6. **Configure Permissions for Argo:**
	To allow Argo to deploy workflows and resources into the `default` namespace, apply the following Kubernetes configuration. This creates a `Role` and `RoleBinding` for the default service account in the `argo` namespace.

	Create a file named `argo-permissions.yaml` with the following content:
	```yaml
	apiVersion: rbac.authorization.k8s.io/v1
	kind: Role
	metadata:
	  name: argo-workflow-role
	  namespace: default
	rules:
	- apiGroups: ["", "apps", "extensions"]
	  resources: ["pods", "pods/exec", "services", "deployments", "ingresses", "workflows", "workflowtemplates"]
	  verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
	---
	apiVersion: rbac.authorization.k8s.io/v1
	kind: RoleBinding
	metadata:
	  name: argo-workflow-binding
	  namespace: default
	roleRef:
	  apiGroup: rbac.authorization.k8s.io
	  kind: Role
	  name: argo-workflow-role
	subjects:
	- kind: ServiceAccount
	  name: default
	  namespace: argo
	```
	Then apply it:
	```bash
	kubectl apply -f argo-permissions.yaml
	```

7. Add ghcr token as a k8s secret.
	```bash
	kubectl create secret docker-registry ghcr-creds --docker-server=https://ghcr.io --docker-username=$YOUR_GITHUB_USERNAME --docker-password=$YOUR_GITHUB_TOKEN --docker-email=$YOUR_EMAIL
	```

8. **Run the main pipeline:**
	```bash
	python main.py
	```

## Modules
- `bandit.py`: Bandit algorithm utilities
- `rag_chain.py`: Retrieval-Augmented Generation pipeline
- `evaluation.py`: Evaluation logic
- `feedback.py`: Feedback collection
- `pipeline_utils.py`: Pipeline helpers
- `data_loaders.py`: Data loading utilities
- `argo_utils.py`: Utilities for interacting with Argo Workflows

## Data & Materials
- `materials/`: Contains source documents, deliverables, and reference files
- `db/`: ChromaDB and other database files

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
