import datetime
import time
import os
import json
import argparse
from kubernetes import client, config
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import csv


HOSTNAME_MAP = {
    "controller": None,
    "worker1": "llmnode1",
    "worker2": "llmnode2",
    "worker3": "llmnode3",
    "worker4": "llmnode4",
    "worker5": "llmnode5",
    "worker6": "llmnode6",
    "worker-arm": None
}

CSV_PATH = None

class Monitor:
    def __init__(self, kepler_ip: str | None = None):
        try:
            config.load_kube_config()
        except Exception:
            config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.appsv1 = client.AppsV1Api()
        self.node_metrics = {}
        self.prev_metrics = {}
        self.metrics_recorder = []
        self.nodes = self.v1.list_node().items
        self.target_namespace_prefix = "default"  

    def fetch_combined_metrics(self, node_ip, kepler_ip, gpu_ip=None):
        metrics = {}
        try:
            response = requests.get(f"http://{node_ip}:9100/metrics")
            response.raise_for_status()
            metrics['node_exporter'] = response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching node exporter metrics from {node_ip}: {e}")
            metrics['node_exporter'] = None

        try:
            if kepler_ip:
                response = requests.get(f"http://{kepler_ip}:28281/metrics", timeout=3)
                response.raise_for_status()
                metrics['kepler'] = response.text
            else:
                metrics['kepler'] = None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching kepler metrics from {kepler_ip}: {e}")
            metrics['kepler'] = None

        try:
            if gpu_ip:
                response = requests.get(f"http://{gpu_ip}:9400/metrics", timeout=3)
                response.raise_for_status()
                metrics['dcgm'] = response.text
            else:
                metrics['dcgm'] = None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching DCGM metrics from {gpu_ip}: {e}")
            metrics['dcgm'] = None

        return metrics

    def fetch_ips(self, kepler = False, gpu = False):
        node_ips = {node.metadata.name: {"node_ip":node.status.addresses[0].address} for node in self.nodes}
        
        # if kepler:
        #     daemonset = self.appsv1.read_namespaced_daemon_set(name="kepler", namespace="kepler")
        #     label_selector = daemonset.spec.selector.match_labels
        #     label_selector = ",".join([f"{key}={value}" for key, value in label_selector.items()])
        #     pods = self.v1.list_namespaced_pod(namespace="kepler", label_selector=label_selector)
        #     for pod in pods.items:
        #         node_name = pod.spec.node_name
        #         pod_ip = pod.status.pod_ip
        #         node_ips[node_name]["kepler_ip"] = pod_ip
        if kepler:
            for node_name in node_ips.keys():
                node_ips[node_name]["kepler_ip"] = '10.68.184.20'  # Hardcoded for testing, replace with actual logic

        if gpu:
            try:
                ds = self.appsv1.read_namespaced_daemon_set(
                    name="nvidia-dcgm-exporter", namespace="gpu-operator"
                )
                label_selector = ",".join([f"{k}={v}" for k, v in ds.spec.selector.match_labels.items()])
                pods = self.v1.list_namespaced_pod(namespace="gpu-operator", label_selector=label_selector).items
                if not pods:
                    pods = self.v1.list_namespaced_pod(namespace="nvidia", label_selector=label_selector).items
                for pod in pods:
                    node_name = pod.spec.node_name
                    pod_ip = pod.status.pod_ip
                    if node_name in node_ips and pod_ip:
                        node_ips[node_name]["gpu_ip"] = pod_ip
            except Exception as e:
                print(f"Error discovering DCGM exporter pods: {e}")

        return node_ips

    def parse_metrics(self, node_metrics, kepler_metrics, prev_metrics, interval, node_name, dcgm_metrics=None):
        parsed = {}
        kepler_match_name = (HOSTNAME_MAP.get(node_name) or node_name).lower()

        if node_metrics:
            for line in node_metrics.split('\n'):
                if line.startswith('#'):
                    continue
                if 'node_memory_MemAvailable_bytes' in line:
                    parsed['free_memory'] = float(line.split(' ')[1])
                elif 'node_memory_MemTotal_bytes' in line:
                    parsed['total_memory'] = float(line.split(' ')[1])
                elif 'node_network_transmit_bytes_total' in line:
                    parsed['tx_bytes'] = float(line.split(' ')[1])
                elif 'node_network_receive_bytes_total' in line:
                    parsed['rx_bytes'] = float(line.split(' ')[1])
                elif 'node_cpu_seconds_total' in line:
                    if 'idle_cpu_seconds' not in parsed:
                        parsed['idle_cpu_seconds'] = 0
                    if 'mode="idle"' in line:
                        parsed['idle_cpu_seconds'] += float(line.split(' ')[1])
                    if 'total_cpu_seconds' not in parsed:
                        parsed['total_cpu_seconds'] = 0
                    parsed['total_cpu_seconds'] += float(line.split(' ')[1])

        if kepler_metrics:
            parsed['node_cpu_power'] = 0
            for line in kepler_metrics.split('\n'):
                if line.startswith('#'):
                    continue
                try:
                    if 'kepler_vm_cpu_watts' in line:
                        vm_name_match = re.search(r'vm_name="([^"]+)"', line)
                        
                        if vm_name_match:
                            vm_name = vm_name_match.group(1).lower()

                            if vm_name == kepler_match_name:
                                
                                parts = line.split()
                                if len(parts) >= 2:
                                    power_watts = float(parts[-1])
                                    
                                    parsed['node_cpu_power'] += power_watts
                except (ValueError, IndexError) as e:
                    print(f"Error parsing Kepler VM metric line: {line[:50]}... - {e}")
                    continue

        if dcgm_metrics:
            metric_lookup = {
                'DCGM_FI_DEV_SM_CLOCK': 'gpu_freq',
                'DCGM_FI_DEV_MEM_CLOCK': 'vram_freq',
                'DCGM_FI_DEV_ENC_UTIL': 'enc_util',
                'DCGM_FI_DEV_DEC_UTIL': 'dec_util',
                'DCGM_FI_DEV_POWER_USAGE': 'power_usage',
                'DCGM_FI_DEV_GPU_UTIL': 'gpu_util',
                'DCGM_FI_DEV_MEM_COPY_UTIL': 'vram_util',
                'DCGM_FI_DEV_FB_FREE': 'vram_free',
                'DCGM_FI_DEV_FB_USED': 'vram_used',
            }
            for line in dcgm_metrics.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                key = None
                value = None
                if '}' in line and ' ' in line:
                    try:
                        left, right = line.split('}', 1)
                        key = left.split('{', 1)[0].strip()
                        value = float(right.strip().split()[0])
                    except Exception:
                        continue
                elif ' ' in line:
                    try:
                        parts = line.split(' ', 1)
                        key = parts[0].strip()
                        value = float(parts[1].strip().split()[0])
                    except Exception:
                        continue
                if key and key in metric_lookup and value is not None:
                    parsed[metric_lookup[key]] = value

        if prev_metrics:
            if 'idle_cpu_seconds' in parsed and 'total_cpu_seconds' in parsed:
                total_cpu_diff = parsed['total_cpu_seconds'] - prev_metrics.get('total_cpu_seconds', 0)
                if total_cpu_diff > 0: 
                    cpu_usage = 100.0 * (1 - (parsed['idle_cpu_seconds'] - prev_metrics.get('idle_cpu_seconds', 0)) / total_cpu_diff)
                    parsed['cpu'] = cpu_usage
                else:
                    parsed['cpu'] = 0  
            
            if 'tx_bytes' in parsed and 'rx_bytes' in parsed:
                tx_rate_mbps = (parsed['tx_bytes'] - prev_metrics.get('tx_bytes', 0)) * 8 / interval / 1e6
                rx_rate_mbps = (parsed['rx_bytes'] - prev_metrics.get('rx_bytes', 0)) * 8 / interval / 1e6
                parsed['tx_rate'] = tx_rate_mbps
                parsed['rx_rate'] = rx_rate_mbps

        return parsed

    def collect_metrics(self, duration_seconds, interval, livesave=False):
        start_time = time.time()
        nodes = self.fetch_ips(kepler=True, gpu=True)

        while time.time() - start_time < duration_seconds:
            with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
                future_to_node = {
                    executor.submit(
                        self.fetch_combined_metrics,
                        node_data["node_ip"],
                        node_data.get("kepler_ip"),
                        node_data.get("gpu_ip"),
                    ): node_name
                    for node_name, node_data in nodes.items()
                }
                
                current_metrics = {}
                for future in as_completed(future_to_node):
                    node_name = future_to_node[future]
                    data = future.result()
                    if data and (data.get('node_exporter') or data.get('kepler') or data.get('dcgm')):
                        if node_name in self.prev_metrics:
                            parsed_metrics = self.parse_metrics(
                                data.get('node_exporter'), 
                                data.get('kepler'),
                                self.prev_metrics[node_name], 
                                interval,
                                node_name,
                                dcgm_metrics=data.get('dcgm'),
                            )
                        else:
                            parsed_metrics = self.parse_metrics(
                                data.get('node_exporter'), 
                                data.get('kepler'),
                                {}, 
                                interval,
                                node_name,
                                dcgm_metrics=data.get('dcgm'),
                            )
                        current_metrics[node_name] = parsed_metrics
                
                self.prev_metrics = current_metrics
                self.node_metrics = current_metrics
                
                if time.time() - start_time > interval:   
                    self.metrics_recorder.append(self.node_metrics)
                    if len(self.metrics_recorder) > 1:  
                        self.metrics_recorder = self.metrics_recorder[1:]  
                    if self.metrics_recorder: 
                        self.prev_metrics = self.metrics_recorder[0] 
                else:
                    self.metrics_recorder.append(self.node_metrics)
                
                if livesave:
                    self.save_metrics_as_json() 
            time.sleep(interval) 

    def save_metrics_as_json(self, output_file="data.json"):
        if self.node_metrics:
            self.col.insert_one({"timestamp": datetime.datetime.now(), "metrics": self.node_metrics})

def _print_metrics_as_json(self, output_file: str = "data.json"):
    """Runtime override for Monitor.save_metrics_as_json: print payload instead of DB insert."""
    try:
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": self.node_metrics,
        }
        print(json.dumps(payload, default=str))
        if CSV_PATH:
            _save_metrics_as_csv(self, CSV_PATH)
    except Exception as e:
        print(f"[monitoring] print_metrics error: {e}")

def _save_metrics_as_csv(self, csv_path: str):
    """Append current node metrics as CSV rows (one row per node)."""
    try:
        if not self.node_metrics:
            return

        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        all_keys = set()
        for node_vals in self.node_metrics.values():
            if isinstance(node_vals, dict):
                all_keys.update(node_vals.keys())

        preferred_order = [
            "free_memory", "total_memory",
            "tx_bytes", "rx_bytes", "tx_rate", "rx_rate",
            "idle_cpu_seconds", "total_cpu_seconds", "cpu",
            "node_cpu_power",
            "gpu_freq", "vram_freq", "enc_util", "dec_util",
            "power_usage", "gpu_util", "vram_util", "vram_free", "vram_used",
        ]
        remaining = [k for k in sorted(all_keys) if k not in preferred_order]
        header = ["timestamp", "node"] + preferred_order + remaining

        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

        now = datetime.datetime.now().isoformat()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            for node, vals in self.node_metrics.items():
                row = [now, node]
                for key in preferred_order + remaining:
                    row.append(vals.get(key, "") if isinstance(vals, dict) else "")
                writer.writerow(row)
    except Exception as e:
        print(f"[monitoring] csv save error: {e}")

Monitor.save_metrics_as_json = _print_metrics_as_json

def main():
    parser = argparse.ArgumentParser(description="Standalone cluster metrics monitor (prints JSON).")
    parser.add_argument("--duration", type=int, default=600, help="Total seconds to run.")
    parser.add_argument("--interval", type=int, default=2, help="Sampling interval in seconds.")
    parser.add_argument("--livesave", action="store_true", help="Print metrics every interval window.")
    parser.add_argument("--csv", type=str, default=None, help="CSV file path to append metrics (used with --livesave).")
    args = parser.parse_args()

    global CSV_PATH
    CSV_PATH = args.csv

    m = Monitor()
    print(f"[monitoring] Starting collection for {args.duration}s, interval={args.interval}s, livesave={args.livesave}, csv={CSV_PATH}")
    m.collect_metrics(duration_seconds=args.duration, interval=args.interval, livesave=args.livesave)
    print("[monitoring] Done.")

if __name__ == "__main__":
    main()