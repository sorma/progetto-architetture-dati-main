"""Configurazione centralizzata della fase generativa RAG."""

import os
import json
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(PROJECT_ROOT, "src", ".env"))

CURRENT_DOMAIN = os.environ.get("DOMAIN", "rfc").lower()

DATASET_PATH = os.path.join(PROJECT_ROOT, "assets", CURRENT_DOMAIN, "domande", "generation.json")
DOCS_DIR = os.path.join(PROJECT_ROOT, "assets", CURRENT_DOMAIN, "base_conoscenza")

BEST_PARAMS_PATH = os.path.join(PROJECT_ROOT, "results", CURRENT_DOMAIN, "optimization", "raw", "best_params.json")
GEN_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", CURRENT_DOMAIN, "generation")

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.1:8b")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")

LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")

PROMPTS_DIR = os.path.join(PROJECT_ROOT, "assets", CURRENT_DOMAIN, "prompts")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "src", "generative_phase", "configs")

with open(os.path.join(CONFIGS_DIR, "experiment_configs.json"), encoding="utf-8") as f:
    EXPERIMENT_CONFIGS = json.load(f)

