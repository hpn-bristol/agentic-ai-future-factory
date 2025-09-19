import re
import os
import yaml
from collections import defaultdict

import config

MANIFESTS_DIR = os.path.join(config.CURRENT_DIR, "modules", "manifests")

# Map DAG node names (from config.MODULES_INFO keys) to manifest files
NODE_TO_MANIFEST = {
    "MoLMo": "molmo.yaml",
    "LSTM-Predictor": "traffic-steering.yaml",
    "Split-Computing-Ctrl": "ros-sc.yaml",
    "Adaptive-Transmitter": "navt.yaml",
}

def _read_manifest_for_node(node: str) -> str | None:
    fname = NODE_TO_MANIFEST.get(node)
    if not fname:
        return None
    path = os.path.join(MANIFESTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _split_manifest_docs(manifest_str: str) -> list[dict]:
    """Parse a YAML manifest string into a list of documents (dicts)."""
    try:
        docs = [d for d in yaml.safe_load_all(manifest_str) if d]
    except yaml.YAMLError:
        # Fallback: try single-doc load
        one = yaml.safe_load(manifest_str)
        docs = [one] if one else []
    return docs

def _dump_doc(doc: dict) -> str:
    """Dump a single YAML document as a string (no '---')."""
    return yaml.dump(doc, sort_keys=False)

def _doc_display_name(base: str, doc: dict, idx: int) -> str:
    kind = str(doc.get("kind", "obj")).lower()
    name = (doc.get("metadata") or {}).get("name") or f"{kind}-{idx}"
    return _sanitize_name(f"{base}-{kind}-{name}")

def _sanitize_name(name: str) -> str:
    # Lowercase and convert anything not [a-z0-9-] to '-'
    name = name.lower()
    name = re.sub(r'[^a-z0-9\-]', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    return name or "unnamed"

def _template_name_for_node(node: str) -> str:
    # Use the manifest base name (without extension) as the template/task name
    fname = NODE_TO_MANIFEST.get(node)
    if fname:
        base = os.path.splitext(os.path.basename(fname))[0]
    else:
        base = f"no-manifest-{node}"
    return _sanitize_name(base)

def parse_to_graph(cand_text: str):
    """Parses the agent's numbered list into a graph structure (nodes and edges)."""
    nodes = []
    edges = []
    step_map = {}
    module_to_step = {}

    for line in cand_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"([\d\.]+)\s+(.+)", line)
        if not match:
            continue
        step, name = match.groups()
        name = name.strip()
        nodes.append(name)
        step_map[step] = name
        module_to_step[name] = step

    for step, module in step_map.items():
        if '.' in step:
            parent_step_prefix = step.split('.')[0]
            potential_parents = [s for s in step_map if s.startswith(parent_step_prefix) and s != step and '.' not in s]
            if not potential_parents:
                parent_major_num = int(parent_step_prefix) - 1
                parent_steps = [s for s in step_map if s.startswith(str(parent_major_num))]
                for ps in parent_steps:
                    edges.append((step_map[ps], module))
            else:
                for ps in potential_parents:
                    edges.append((step_map[ps], module))

    return list(dict.fromkeys(nodes)), edges

def is_dag(nodes, edges):
    """Verifies that the graph is a DAG (has no cycles) using DFS."""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    visiting = set()
    visited = set()

    def has_cycle(node):
        visiting.add(node)
        for neighbor in graph.get(node, []):
            if neighbor in visiting:
                return True
            if neighbor not in visited and has_cycle(neighbor):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    for node in nodes:
        if node not in visited:
            if has_cycle(node):
                return False
    return True

def verify_dependencies(nodes, edges):
    """Checks if the graph structure respects the predefined module dependencies."""
    predecessors = defaultdict(set)
    for u, v in edges:
        predecessors[v].add(u)

    for node in nodes:
        required_deps = config.MODULE_DEPENDENCIES.get(node, [])
        if not required_deps:
            continue
        if not set(required_deps).issubset(predecessors[node]):
            missing = set(required_deps) - predecessors[node]
            print(f"🔥 Verification Error: Module '{node}' is missing dependencies: {list(missing)}")
            return False

    print("✅ All module dependencies are satisfied.")
    return True

def generate_argo_yaml(
    name: str,
    nodes: list,
    edges: list,
    namespace: str = "agentic-ai",
    wait_for_dependencies: bool = True,
    service_account: str | None = "argo-executor",
) -> str:
    """Generates an Argo Workflow YAML that applies module-specific manifests per node.
    Template/task names are derived from the manifest filename instead of the node name.
    """
    templates = []
    # Map each node to a list of template names (one per manifest doc)
    node_to_templates: dict[str, list[str]] = {}
    name_counts = {}

    def _unique(n: str) -> str:
        c = name_counts.get(n, 0)
        if c == 0:
            name_counts[n] = 1
            return n
        name_counts[n] = c + 1
        return f"{n}-{c+1}"

    # A resource template per node (inlines the manifest content)
    for node in nodes:
        tmpl_base = _template_name_for_node(node)
        manifest_str = _read_manifest_for_node(node)
        node_templates: list[str] = []

        if manifest_str:
            docs = _split_manifest_docs(manifest_str)
            if not docs:
                # No valid docs, create a no-op
                tname = _unique(tmpl_base)
                templates.append({
                    "name": tname,
                    "container": {
                        "image": "alpine:3.20",
                        "command": ["sh", "-c"],
                        "args": [f"echo 'Empty manifest for {node}. Skipping.'"]
                    }
                })
                node_templates.append(tname)
            else:
                # One template per document
                for i, doc in enumerate(docs, start=1):
                    tname = _unique(_doc_display_name(tmpl_base, doc, i))
                    templates.append({
                        "name": tname,
                        "resource": {
                            "action": "apply",
                            "manifest": _dump_doc(doc),
                            "setOwnerReference": True
                        }
                    })
                    node_templates.append(tname)
        else:
            # No manifest mapped for this node
            tname = _unique(tmpl_base)
            templates.append({
                "name": tname,
                "container": {
                    "image": "alpine:3.20",
                    "command": ["sh", "-c"],
                    "args": [f"echo 'No manifest mapped for {node}. Skipping.'"]
                }
            })
            node_templates.append(tname)

        node_to_templates[node] = node_templates

    # Build DAG tasks: include every per-doc template as its own task
    dag_tasks = []
    for node in nodes:
        # All upstream nodes that this node depends on
        upstream_nodes = {u for (u, v) in edges if v == node}
        # Flatten upstream templates
        upstream_templates = []
        for u in upstream_nodes:
            upstream_templates.extend(node_to_templates.get(u, []))
        # Create one task per template for this node
        for tname in node_to_templates.get(node, []):
            task = {"name": tname, "template": tname}
            if wait_for_dependencies and upstream_templates:
                # De-duplicate while preserving order
                seen = set()
                deps = [d for d in upstream_templates if not (d in seen or seen.add(d))]
                task["dependencies"] = deps
            dag_tasks.append(task)

    templates.append({"name": "main-dag", "dag": {"tasks": dag_tasks}})

    workflow_spec = {"entrypoint": "main-dag", "templates": templates}
    if not wait_for_dependencies:
        workflow_spec["parallelism"] = max(1, len(dag_tasks))
    if service_account:
        workflow_spec["serviceAccountName"] = service_account

    workflow = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"generateName": f"{name}-", "namespace": namespace},
        "spec": workflow_spec,
    }
    return yaml.dump(workflow, sort_keys=False)
