import json
import os
from optimization_phase.configs.config import RESULTS_DIR, RAW_RESULTS_DIR, PLOTS_DIR
from optimization_phase.analysis.plots_b1 import plot_block1
from optimization_phase.analysis.plots_b2 import plot_block2
from optimization_phase.analysis.plots_b3 import plot_block3
from optimization_phase.analysis.plots_noisy import plot_noisy_analysis

def genera_tutti_i_grafici(raw_dir: str = RAW_RESULTS_DIR, out_dir: str = PLOTS_DIR) -> None:
    """Legge i file JSON dei tre blocchi e genera i relativi grafici (multi-dominio)."""
    b1_path = os.path.join(raw_dir, "block1_results.json")
    b2_path = os.path.join(raw_dir, "block2_results.json")
    b3_path = os.path.join(raw_dir, "block3_results.json")
    
    os.makedirs(out_dir, exist_ok=True)

    dati_blocchi = {}

    if os.path.exists(b1_path):
        print("[grafici] Generazione grafici Blocco 1...")
        with open(b1_path, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        dati_blocchi["1"] = d1
        plot_block1(d1, out_dir)
    else:
        print(f"[grafici] {b1_path} non trovato — Blocco 1 non ancora eseguito.")

    if os.path.exists(b2_path):
        print("[grafici] Generazione grafici Blocco 2...")
        with open(b2_path, "r", encoding="utf-8") as f:
            d2 = json.load(f)
        dati_blocchi["2"] = d2
        plot_block2(d2, out_dir)
    else:
        print(f"[grafici] {b2_path} non trovato — Blocco 2 non ancora eseguito.")

    if os.path.exists(b3_path):
        print("[grafici] Generazione grafici Blocco 3...")
        with open(b3_path, "r", encoding="utf-8") as f:
            d3 = json.load(f)
        dati_blocchi["3"] = d3
        plot_block3(d3, out_dir)
    else:
        print(f"[grafici] {b3_path} non trovato — Blocco 3 non ancora eseguito.")


    print("[grafici] Grafici salvati in:", out_dir)
