"""Configurazione centrale dell'intero sistema di ottimizzazione RAG."""

import os
import json
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(PROJECT_ROOT, "src", ".env"))

DOMAIN = os.environ.get("DOMAIN", "rfc").lower()

DOCS_DIR = os.path.join(PROJECT_ROOT, "assets", DOMAIN, "base_conoscenza")
DATASET_PATH = os.path.join(PROJECT_ROOT, "assets", DOMAIN, "domande", "optimization.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", DOMAIN, "optimization")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PREBUILD_DIR = os.path.join(RESULTS_DIR, "prebuild")

# Hardware setup
if os.environ.get("EMBEDDING_DEVICE", "").lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch
    torch_gpu_available = torch.cuda.is_available()
    if not torch_gpu_available or os.environ.get("EMBEDDING_DEVICE", "").lower() == "cpu":
        torch.set_num_threads(8)
except ImportError:
    torch_gpu_available = False

EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cuda" if torch_gpu_available else "cpu")
try:
    EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64" if torch_gpu_available else "16"))
except ValueError:
    EMBED_BATCH_SIZE = 64 if torch_gpu_available else 16

# Groq API Keys
keys_str = os.environ.get("GROQ_API_KEYS", "YOUR_GROQ_API_KEY")
GROQ_API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else "YOUR_GROQ_API_KEY"

GROQ_QT_MODEL = "llama-3.1-8b-instant"
GROQ_RERANK_MODEL = "llama-3.1-8b-instant"

# Default params
DEFAULT_QT = "none"
DEFAULT_RETRIEVAL = "vector"
DEFAULT_K = 5

PASSAGE_EVAL_MODEL = "BAAI/bge-reranker-v2-m3"
PASSAGE_SIM_THRESHOLD = 0.1

BLOCK1_MAX_WORKERS = 1
BLOCK2_MAX_WORKERS = 2
BLOCK3_MAX_WORKERS = 2

USE_LOCAL_RERANKER = os.environ.get(
    "USE_LOCAL_RERANKER", "True" if torch_gpu_available else "False"
).lower() in ("true", "1", "yes")

# Load configs from JSON
configs_path = os.path.join(PROJECT_ROOT, "src", "optimization_phase", "configs", "experiment_configs.json")
with open(configs_path, "r", encoding="utf-8") as f:
    _configs = json.load(f)

BLOCK1_CONFIGS = _configs.get("block1", [])
BLOCK2_QT_TYPES = _configs.get("block2_qt_types", [])
BLOCK3_CONFIGS = _configs.get("block3", [])
