import os
import json

# --- Directories ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "db")
MATERIALS_DIR = os.path.join(CURRENT_DIR, "materials")

# --- File Paths ---
PERSIST_DIR = os.path.join(DB_DIR, "chroma_db_openai")
HASH_FILE_PATH = os.path.join(MATERIALS_DIR, "file_hashes.json")
INTENTS_JSON_PATH = os.path.join(CURRENT_DIR, "intents-REASON.json")
BL_PATH = os.path.join(CURRENT_DIR, "pipeline_blacklist.json")
BANDIT_STATE_PATH = os.path.join(CURRENT_DIR, "bandit_state.pkl")
RAG_FEEDBACK_PATH = os.path.join(CURRENT_DIR, "rag_feedback.txt")
RUN_METRICS_PATH = os.path.join(CURRENT_DIR, "run_metrics.jsonl")
ATS_LOG_PATH = os.path.join(CURRENT_DIR, "ats_log.csv")

# --- Meterial Subdirectories ---
URL_PATH = os.path.join(MATERIALS_DIR, "website.txt")
WIKIPEDIA_PATH = os.path.join(MATERIALS_DIR, "wikipedia.txt")
ORAN_DIR = os.path.join(MATERIALS_DIR, "oran")
GPP_DIR = os.path.join(MATERIALS_DIR, "gpp")
PAPER_DIR = os.path.join(MATERIALS_DIR, "papers")
DELIVER_DIR = os.path.join(MATERIALS_DIR, "deliverables")

# --- RAG & LLM Constants ---
MODULES_INFO = {
    "UE-Monitor": "Monitor and get each robot/drone's position, velocity, battery level, computing resources and onboard video feed.",
    "UE-Controller": "Issues motion primitives (way-points, speed set-points) to user equipments.",
    "Wireless-Monitor": "Records Signal to Noise Ratio (SNR), Received Signal Strength Indicator (RSSI), Signal-to-Interference-plus-Noise Ratio (SINR) and link-outage statistics so other models can use as input.",
    "Wireless-Controller": "Performs on-line Wi-Fi/Li-Fi AP reselection & handover. This is a necessary module as long as wireless connection is needed.",
    "Server-Status-Monitor": "Tracks MEC CPU / GPU utilisation and memory headroom.",
    "YOLO": "It's a module that can be placed on the drone or splited into two parts to distribute the computing demands. It is a necessary module for any computer vision task, like detecting people or objects in the factory. It converts images to JSON.",
    "MoLMo": "It's a lightweight LLM that can be deployed on edge servers (MEC). This is a necessary module for any human interaction task, like asking questions about the video captured by drones.",
    "LSTM-Predictor": "Input current traffic, then predict future SNR of user equipments, which can used for handover and futher improving the QoE of user. ",
    "Semantic-Codec": "Compresses JSON telemetry into a smaller JSON, which can reduce bandwidth.",
    "Split-Computing-Ctrl": "It needs computing resource information of both user equipments and edge server, chooses the optimal split point of YOLO, so part of the model can run on UE, the other part on the MEC to reduce UE computing power. YOLO must be used before this module as the input. So as UE-Monitor and Server-Status-Monitor. And due to the distribution, wirelss connection is needed, so Wireless-Controller is needed after this module.",
    "Adaptive-Transmitter": "Adaptively selects the optimal wireless image-transmission parameters—such as carrier frequency band and resolution—based on the current SNR and the latency constraints. Wireless-Monitor must be used before this module as the input. And wireless controller is needed after this module to perform the actual transmission.",
}

OUTPUT_SPEC = (
    "When you answer **generate exactly {k} candidate DAGs** for the given intent.\n"
    "• Use numbered lines only. Example:\n"
    "  Candidate-1:\n"
    "    1. UE-Monitor\n"
    "    2. YOLO\n"
    "  Candidate-2 (with parallel step):\n"
    "    1. Wireless-Monitor\n"
    "    2.1 LSTM-Predictor\n"
    "    2.2 Semantic-Codec\n"
    "• Do NOT add any explanation or extra text.\n"
)

_CANON = {k.lower(): k for k in MODULES_INFO}
_CANON.update({
    "ue monitor": "UE-Monitor",
    "wireless monitor": "Wireless-Monitor",
    "server status monitor": "Server-Status-Monitor",
    "wireless controller": "Wireless-Controller",
    "yolo": "YOLO",
    "molmo": "MoLMo",
    "lstm predictor": "LSTM-Predictor",
    "semantic codec": "Semantic-Codec",
    "split computing ctrl": "Split-Computing-Ctrl",
    "split computing module": "Split-Computing-Ctrl",
    "adaptive transmitter": "Adaptive-Transmitter",
})

# MODULE_DEPENDENCIES = {
#     "UE-Monitor": [],
#     "UE-Controller": [],
#     "Wireless-Monitor": [],
#     "Server-Status-Monitor": [],
#     "YOLO": ["UE-Monitor"],
#     "MoLMo": ["UE-Monitor", "YOLO", "Semantic-Codec"],
#     "LSTM-Predictor": ["Wireless-Monitor"],
#     "Semantic-Codec": ["YOLO"],
#     "Split-Computing-Ctrl": ["UE-Monitor", "YOLO", "Server-Status-Monitor"],
#     "Adaptive-Transmitter": ["UE-Monitor", "Wireless-Monitor"],
#     "Wireless-Controller": [],
# }


# test only
MODULE_DEPENDENCIES = {
    "UE-Monitor": [],
    "UE-Controller": [],
    "Wireless-Monitor": [],
    "Server-Status-Monitor": [],
    "YOLO": [],
    "MoLMo": [],
    "LSTM-Predictor": [],
    "Semantic-Codec": [],
    "Split-Computing-Ctrl": [],
    "Adaptive-Transmitter": [],
    "Wireless-Controller": [],
}

GOLD = {
    "Improve the QoE for user equipment (UEs) while they are moving in the factory": [ #Intent 1 from paper
        ("all", {"UE-Monitor", "Wireless-Monitor"}),
        ("all", {"LSTM-Predictor"}),
        ("all", {"Wireless-Controller"}),
    ],
    "Enable drones to perform on-board model detection and minimize bandwidth usage when transmitting the resulting JSON files to the MEC.": [ #Intent 2 from paper
        ("all", {"UE-Monitor"}),
        ("all", {"YOLO"}),
        ("all", {"Semantic-Codec"}),
        ("opt", {"LSTM-Predictor"}),
        ("all", {"Wireless-Controller"}),
    ],
    "Enable drones to perform on-board model detection and reduce the computing-energy costs of the drones": [ #Intent 3 from paper
        ("all", {"UE-Monitor"}),
        ("all", {"YOLO", "Server-Status-Monitor"}),
        ("all", {"Split-Computing-Ctrl"}),
        ("opt", {"LSTM-Predictor"}),
        ("all", {"Wireless-Controller"}),
    ],
    "When the robots move around, ensure they transmit according to real-time link quality to achieve the best video quality within latency limitations": [ #Intent 4 from paper
        ("all", {"Wireless-Monitor"}),
        ("all", {"Adaptive-Transmitter"}),
        ("opt", {"LSTM-Predictor"}),
        ("all", {"Wireless-Controller"}),
    ],
    "Enable drones to perform on-board model detection and reduce bandwidth during video transmission, then enable people to ask questions about drone-captured videos": [ #Unseen intent 5 from paper
        ("all", {"UE-Monitor"}),
        ("all", {"YOLO"}),
        ("all", {"Semantic-Codec"}),
        ("opt", {"LSTM-Predictor"}),
        ("all", {"Wireless-Controller"}),
        ("all", {"MoLMo"}),
    ],
    "When user equipment (UEs) move around the factory, improve their QoE and avoid connection loss": [    #unused intent 6
        ("all", {"UE-Monitor", "Wireless-Monitor"}),
        ("all", {"LSTM-Predictor"}),
        ("all", {"Wireless-Controller"}),
    ],
}


# --- Bandit & Training Constants ---
MAX_T = 150
EMB_DIM = 3072  # text-embedding-3-small (1536) + text-embedding-3-small (1536)

# --- Blacklist ---
try:
    with open(BL_PATH, 'r') as f:
        BLACKLIST: dict[str, list[str]] = json.load(f)
except FileNotFoundError:
    BLACKLIST = {}
