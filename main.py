import argparse
import os
import json
import numpy as np
import subprocess
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config
from helpers.data_loaders import load_documents
from helpers.rag_chain import get_retriever, build_chain, run_intent
from helpers.feedback import purge_feedback_vectors, log_feedback
from helpers.bandit import load_bandit_state, save_bandit_state
from helpers.pipeline_utils import split_cands, pipe_key
from helpers.evaluation import evaluate
from helpers.argo_utils import parse_to_graph, is_dag, verify_dependencies, generate_argo_yaml

from functools import lru_cache

import threading
import time
import csv
import psutil

load_dotenv()
_EMB = OpenAIEmbeddings(model="text-embedding-3-small")


@lru_cache(maxsize=4096)
def _emb(txt: str) -> np.ndarray:
    return np.asarray(_EMB.embed_query(txt), dtype=np.float32)


def phi(intent_txt: str, pipeline_txt: str) -> np.ndarray:
    return np.concatenate([_emb(intent_txt), _emb(pipeline_txt)], axis=0)


def _utilization_worker(path: str, interval: float, stop_event: threading.Event):
    proc = psutil.Process(os.getpid())

    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    start_ts = time.time()
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "elapsed_s", "pid", "proc_cpu_percent",
                "proc_mem_percent", "proc_rss_bytes", "sys_cpu_percent",
                "sys_mem_percent", "disk_read_bytes", "disk_write_bytes",
                "net_bytes_sent", "net_bytes_recv"
            ])
            f.flush()

        while not stop_event.is_set():
            time.sleep(interval)
            now = time.time()
            elapsed = now - start_ts

            proc_cpu = proc.cpu_percent(interval=None)
            sys_cpu = psutil.cpu_percent(interval=None)

            proc_mem_pct = proc.memory_percent()
            rss = proc.memory_info().rss
            sys_mem_pct = psutil.virtual_memory().percent

            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()

            writer.writerow([
                f"{now:.6f}",
                f"{elapsed:.3f}",
                proc.pid,
                f"{proc_cpu:.2f}",
                f"{proc_mem_pct:.2f}",
                rss,
                f"{sys_cpu:.2f}",
                f"{sys_mem_pct:.2f}",
                getattr(disk, "read_bytes", 0),
                getattr(disk, "write_bytes", 0),
                getattr(net, "bytes_sent", 0),
                getattr(net, "bytes_recv", 0),
            ])
            f.flush()


def start_utilization_logger(path: str, interval: float) -> threading.Event:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stop_event = threading.Event()
    t = threading.Thread(target=_utilization_worker, args=(path, interval, stop_event), daemon=True)
    t.start()
    return stop_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="delete previous RAG logs and vector DB")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--util-log", type=str, default="hardware_usage.csv",
                        help="CSV path to write hardware utilization samples")
    parser.add_argument("--util-interval", type=float, default=1.0,
                        help="Sampling interval in seconds for utilization logging")
    args = parser.parse_args()

    if args.reset:
        for path in [config.RAG_FEEDBACK_PATH, config.ATS_LOG_PATH, config.BANDIT_STATE_PATH]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(config.PERSIST_DIR):
            embed = OpenAIEmbeddings(model="text-embedding-3-small")
            db = Chroma(persist_directory=config.PERSIST_DIR,
                        embedding_function=embed)
            purge_feedback_vectors(db)
        print("✔ feedback history cleared; core corpus retained")
        if os.path.exists(config.BL_PATH):
            os.remove(config.BL_PATH)
        exit(0)

    util_stop_event = start_utilization_logger(args.util_log, args.util_interval)
    print(f"🧭 Utilization CSV: {os.path.abspath(args.util_log)}")

    try:
        print("🖇  RAG log  :", os.path.abspath(config.RAG_FEEDBACK_PATH))
        print("📂 Chroma DB:", os.path.abspath(config.PERSIST_DIR))

        docs, upd = load_documents()
        retriever, db = get_retriever(docs, upd)
        intents = list(config.GOLD.keys())
        rag_chain = build_chain(retriever, intents, k=5)

        train_intents = intents[0:-2]
        test_intents = intents[5:6]
        results = []

        def run_phase(intent_list, phase: str,
                      update_bandit: bool,
                      stop_on_perfect: bool = True,
                      consec_success_needed: int = 2,
                      log_theta: bool = True):
            for intent in intent_list:
                print(f"\n=== {phase.upper()} | {intent} ===")
                bandit = load_bandit_state(config.EMB_DIM)
                success_hist = []
                theta_hist = []

                consec = 0
                attempts_at_consec = config.MAX_T + 1

                for t in range(1, config.MAX_T + 1):
                    llm_out = run_intent(intent, rag_chain)
                    cands = split_cands(llm_out)

                    pool_all = {cid: phi(intent, txt)
                                for cid, txt in cands.items()}
                    pool = {cid: v for cid, v in pool_all.items()
                            if pipe_key(cands[cid]) not in config.BLACKLIST.get(intent, [])}
                    if not pool:
                        print("⚠ all candidates black-listed; skip this round")
                        continue

                    chosen = bandit.select(pool)
                    chosen_txt = cands[chosen]
                    reward, label = evaluate(intent, chosen_txt)

                    log_feedback(db, intent, chosen_txt, label, reward)

                    if update_bandit:
                        bandit.update(pool[chosen], reward)
                        save_bandit_state(bandit)

                    theta_norm = float(np.linalg.norm(bandit.A_inv @ bandit.b))
                    success_hist.append(int(reward == 1.0))
                    if log_theta:
                        theta_hist.append(theta_norm)

                    one_line = " | ".join(ln.strip()
                                          for ln in chosen_txt.splitlines())
                    print(
                        f"t={t:02d} | reward={reward:.1f} | θ‖≈{theta_norm:.2f} | {one_line}")

                    if reward == 1.0:
                        consec += 1
                        if consec >= consec_success_needed and attempts_at_consec == config.MAX_T + 1:
                            attempts_at_consec = t

                            print(
                                "\n🚀 Perfect pipeline found. Verifying and preparing for deployment...")
                            nodes, edges = parse_to_graph(chosen_txt)

                            is_valid_dag = is_dag(nodes, edges)
                            deps_ok = verify_dependencies(nodes, edges)

                            if is_valid_dag and deps_ok:
                                print("✅ Graph is a valid DAG and dependencies are met.")

                                intent_slug = intent.lower().replace(
                                    ' ', '-').replace('(', '').replace(')', '')[:20]
                                yaml_content = generate_argo_yaml(
                                    intent_slug, nodes, edges, wait_for_dependencies=False)

                                yaml_filename = f"{intent_slug}-workflow.yaml"
                                with open(yaml_filename, "w") as f:
                                    f.write(yaml_content)
                                print(
                                    f"✅ Argo Workflow YAML saved to '{yaml_filename}'")

                                try:
                                    print(
                                        f"🚢 Submitting '{yaml_filename}' to Argo...")
                                    result = subprocess.run(
                                        ["argo", "submit", yaml_filename,
                                            "-n", "default"],
                                        capture_output=True, text=True, check=True
                                    )
                                    print("✅ Workflow submitted successfully!")
                                    print(result.stdout)
                                except FileNotFoundError:
                                    print(
                                        "🔥 Deployment Error: 'argo' command not found. Is Argo CLI installed and in your PATH?")
                                except subprocess.CalledProcessError as e:
                                    print(f"🔥 Deployment Error: 'argo submit' failed.")
                                    print(e.stderr)
                            else:
                                print("🔥 Verification Failed. Skipping deployment.")

                            if stop_on_perfect:
                                print(
                                    f"PERFECT {consec_success_needed}× in a row at step {t}")
                                break
                    else:
                        consec = 0

                results.append({
                    "run_id": os.getenv("SEED", "0"),
                    "phase": phase,
                    "intent": intent,
                    "ATS": attempts_at_consec,
                    "theta_final": theta_hist[-1] if theta_hist else 0.0,
                    "succ_series": success_hist,
                    "theta_series": theta_hist
                })

        # run_phase(train_intents, "train", update_bandit=True,
        #           stop_on_perfect=True, consec_success_needed=2)
        run_phase(test_intents,  "test",  update_bandit=False, stop_on_perfect=True)

        with open(config.RUN_METRICS_PATH, "a") as f:
            for row in results:
                json.dump(row, f)
                f.write("\n")
        print(f"✔ All metrics appended to {config.RUN_METRICS_PATH}")
    finally:
        util_stop_event.set()


if __name__ == "__main__":
    main()
