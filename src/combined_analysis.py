import os
import json
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.framealpha": 0.85,
    "legend.fontsize":   9,
})

COLORS = {
    "TP": "#55A868",
    "TN": "#4C72B0",
    "FP": "#C44E52",
    "FN": "#3B2E5A" 
}

def _salva(path: str) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [OK] Salvato: {path}")

def carica_matrice(domain: str) -> dict:
    percorso = os.path.join(PROJECT_ROOT, "results", domain, "generation", "aggregated_metrics.json")
    matrice = {}
    if os.path.exists(percorso):
        with open(percorso, encoding='utf-8') as f:
            data = json.load(f)
            data.sort(key=lambda x: int(x['config_id'][1:]))
            for riga in data:
                matrice[riga['config_id']] = {
                    "TP": riga['TP'],
                    "TN": riga['TN'],
                    "FP": riga['FP'],
                    "FN": riga['FN']
                }
    return matrice

def crea_grafico_matrice_congiunto():
    print(f"Generazione grafico matrice congiunto dai dati di tutti i domini...")
    matrice_rfc = carica_matrice("rfc")
    matrice_ww1 = carica_matrice("ww1")
    
    if not matrice_rfc or not matrice_ww1:
        print(f"Dati mancanti in uno dei domini. Impossibile creare grafico congiunto.")
        return

    configs = list(matrice_rfc.keys())
    
    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    
    bottom_rfc = np.zeros(len(configs))
    bottom_ww1 = np.zeros(len(configs))
    
    metrics_order = ["TP", "TN", "FP", "FN"]
    
    for i, metric in enumerate(metrics_order):
        vals_rfc = [matrice_rfc[c][metric] for c in configs]
        vals_ww1 = [matrice_ww1[c][metric] for c in configs]
        
        rects_rfc = ax.bar(x - width/2, vals_rfc, width, label=metric, color=COLORS[metric], bottom=bottom_rfc, edgecolor='white', linewidth=0.5)
        rects_ww1 = ax.bar(x + width/2, vals_ww1, width, color=COLORS[metric], bottom=bottom_ww1, edgecolor='white', linewidth=0.5)
        
        for rect in rects_rfc:
            h = rect.get_height()
            if h > 3:
                ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + h/2.,
                        f'{int(h)}', ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')
                        
        for rect in rects_ww1:
            h = rect.get_height()
            if h > 3:
                ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + h/2.,
                        f'{int(h)}', ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')
                        
        bottom_rfc += vals_rfc
        bottom_ww1 += vals_ww1

    for idx in range(len(configs)):
        ax.text(x[idx] - width/2, -3, 'RFC', ha='center', va='top', fontsize=8, color='#333')
        ax.text(x[idx] + width/2, -3, 'WW1', ha='center', va='top', fontsize=8, color='#333')

    ax.set_ylabel("Quantità (n° domande)")
    ax.set_xlabel("Configurazioni")
    ax.set_title("Matrice di Confusione Quantitativa (TP, TN, FP, FN) Congiunta - Fase Generativa", pad=20)
    
    ax.set_ylim(0, 160)
    ax.set_xlim(-0.5, len(configs) + 0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.tick_params(axis='x', pad=15)
    
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc='lower right', framealpha=0.9)
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generation_confusion.png")
    
    _salva(out_path)



def carica_opt_json(domain: str, block: str) -> list[dict]:
    path = os.path.join(PROJECT_ROOT, "results", domain, "optimization", "raw", f"{block}_results.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def crea_grafico_opt_b1_tempi_congiunto():
    d1_rfc = carica_opt_json("rfc", "block1")
    d1_ww1 = carica_opt_json("ww1", "block1")
    if not d1_rfc or not d1_ww1: return

    etichette = [f"C{r['config']['id']}\n{r['config']['splitter'][:4]}-{r['config']['chunk_size']}" for r in d1_rfc]
    
    x = np.arange(len(etichette))
    w = 0.2 
    
    t_chunk_rfc = [r.get("chunk_time_sec", 0) for r in d1_rfc]
    t_embed_rfc = [r.get("embed_time_sec", 0) for r in d1_rfc]
    
    if len(d1_ww1) != len(d1_rfc): return
    
    t_chunk_ww1 = [r.get("chunk_time_sec", 0) for r in d1_ww1]
    t_embed_ww1 = [r.get("embed_time_sec", 0) for r in d1_ww1]

    fig, ax = plt.subplots(figsize=(16, 7))
    
    b1 = ax.bar(x - 1.5*w, t_embed_ww1, w, label='embedding_ww1', color='#8172B2', edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x - 0.5*w, t_chunk_ww1, w, label='chunking_ww1', color='#4C72B0', edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + 0.5*w, t_embed_rfc, w, label='embedding_rfc', color='#CDB370', edgecolor='white', linewidth=0.5)
    b4 = ax.bar(x + 1.5*w, t_chunk_rfc, w, label='chunking_rfc', color='#55A868', edgecolor='white', linewidth=0.5)
    
    for barre, vals in [(b1, t_embed_ww1), (b2, t_chunk_ww1), (b3, t_embed_rfc), (b4, t_chunk_rfc)]:
        for barra, t in zip(barre, vals):
            if t > 0:
                ax.text(barra.get_x() + barra.get_width() / 2,
                        barra.get_height() + (max(t_chunk_rfc + t_embed_rfc + t_chunk_ww1 + t_embed_ww1)*0.01),
                        f"{t:.0f}s", ha="center", va="bottom", fontsize=8)
                        
    ax.set_ylabel("Tempo (secondi)")
    ax.set_title("Blocco 1 — Tempi di Chunking ed Embedding per Configurazione e Dominio", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, fontsize=9)
    ax.legend(loc="upper left", ncol=2)
    
    max_t = max(t_chunk_rfc + t_embed_rfc + t_chunk_ww1 + t_embed_ww1)
    ax.set_ylim(0, max_t * 1.15 if max_t else 1)
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    _salva(os.path.join(out_dir, "block1_tempi.png"))

def crea_grafico_opt_b2_latenza_congiunto():
    d2_rfc = carica_opt_json("rfc", "block2")
    d2_ww1 = carica_opt_json("ww1", "block2")
    if not d2_rfc or not d2_ww1: return
    
    etichette = [r["qt_type"] for r in d2_rfc]
    x = np.arange(len(etichette))
    w = 0.35
    
    lat_rfc = [r.get("avg_latency_sec", 0) for r in d2_rfc]
    lat_ww1 = [r.get("avg_latency_sec", 0) for r in d2_ww1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects_rfc = ax.bar(x - w/2, lat_rfc, w, label='RFC', color="#4C72B0", zorder=3)
    rects_ww1 = ax.bar(x + w/2, lat_ww1, w, label='WW1', color="#C44E52", zorder=3)
    
    for rect in rects_rfc:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
    for rect in rects_ww1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    max_lat = max(max(lat_rfc), max(lat_ww1))
    ax.set_ylim(0, max_lat * 1.25 if max_lat else 1)
    
    ax.set_ylabel("Latenza Media (Secondi)")
    ax.set_xlabel("Strategia QT")
    ax.set_title("Latenza Media Congiunta (Blocco 2)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(etichette)
    ax.legend(loc="upper left")
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    _salva(os.path.join(out_dir, "block2_latenza.png"))

def crea_grafico_opt_b2_tokens_congiunto():
    d2_rfc = carica_opt_json("rfc", "block2")
    d2_ww1 = carica_opt_json("ww1", "block2")
    if not d2_rfc or not d2_ww1: return
    
    etichette = [r["qt_type"] for r in d2_rfc]
    x = np.arange(len(etichette))
    w = 0.35
    
    in_rfc = np.array([r.get("avg_input_tokens", 0) for r in d2_rfc])
    out_rfc = np.array([r.get("avg_output_tokens", 0) for r in d2_rfc])
    tot_rfc = in_rfc + out_rfc
    
    in_ww1 = np.array([r.get("avg_input_tokens", 0) for r in d2_ww1])
    out_ww1 = np.array([r.get("avg_output_tokens", 0) for r in d2_ww1])
    tot_ww1 = in_ww1 + out_ww1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - w/2, in_rfc, w, label='RFC Input', color='tab:blue', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x - w/2, out_rfc, w, bottom=in_rfc, label='RFC Output', color='tab:blue', alpha=0.45, edgecolor='white', linewidth=0.5)
    
    ax.bar(x + w/2, in_ww1, w, label='WW1 Input', color='tab:orange', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, out_ww1, w, bottom=in_ww1, label='WW1 Output', color='tab:orange', alpha=0.45, edgecolor='white', linewidth=0.5)
    
    for i in range(len(etichette)):
        if tot_rfc[i] > 0:
            ax.text(x[i] - w/2, tot_rfc[i] + max(max(tot_rfc), max(tot_ww1))*0.01, f"{tot_rfc[i]:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(x[i] - w/2, in_rfc[i]/2, f"{in_rfc[i]:.1f}", ha='center', va='center', fontsize=8)
            ax.text(x[i] - w/2, in_rfc[i] + out_rfc[i]/2, f"{out_rfc[i]:.1f}", ha='center', va='center', fontsize=8)
            
        if tot_ww1[i] > 0:
            ax.text(x[i] + w/2, tot_ww1[i] + max(max(tot_rfc), max(tot_ww1))*0.01, f"{tot_ww1[i]:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(x[i] + w/2, in_ww1[i]/2, f"{in_ww1[i]:.1f}", ha='center', va='center', fontsize=8)
            ax.text(x[i] + w/2, in_ww1[i] + out_ww1[i]/2, f"{out_ww1[i]:.1f}", ha='center', va='center', fontsize=8)
            
    ax.set_ylabel("Token LLM medi per query")
    ax.set_xlabel("Strategia QT")
    ax.set_title("Blocco 2 — Consumo Medio Token LLM (Input vs Output)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(etichette)
    ax.legend(loc="upper left")
    
    max_tot = max(max(tot_rfc), max(tot_ww1))
    ax.set_ylim(0, max_tot * 1.25 if max_tot else 1)
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    _salva(os.path.join(out_dir, "block2_tokens.png"))

def crea_grafico_opt_b3_latenza_congiunto():
    d3_rfc = carica_opt_json("rfc", "block3")
    d3_ww1 = carica_opt_json("ww1", "block3")
    if not d3_rfc or not d3_ww1: return
    
    etichette = [f"C{r['config']['id']}" for r in d3_rfc]
    x = np.arange(len(etichette))
    w = 0.35
    
    lat_rfc = [r.get("avg_latency_sec", 0) for r in d3_rfc]
    lat_ww1 = [r.get("avg_latency_sec", 0) for r in d3_ww1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects_rfc = ax.bar(x - w/2, lat_rfc, w, label='RFC', color="#4C72B0", zorder=3)
    rects_ww1 = ax.bar(x + w/2, lat_ww1, w, label='WW1', color="#C44E52", zorder=3)
    
    for rect in rects_rfc:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
    for rect in rects_ww1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    max_lat = max(max(lat_rfc), max(lat_ww1))
    ax.set_ylim(0, max_lat * 1.25 if max_lat else 1)
    
    ax.set_ylabel("Latenza Media (Secondi)")
    ax.set_xlabel("Configurazione Retrieval")
    ax.set_title("Blocco 3 — Latenza Media Congiunta", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(etichette)
    ax.legend(loc="upper left")
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    _salva(os.path.join(out_dir, "block3_latenza.png"))

def genera_tutti_grafici_congiunti():
    print("\n--- Generazione Grafici Congiunti (Tutti i Domini) ---")
    crea_grafico_matrice_congiunto()

    crea_grafico_opt_b1_tempi_congiunto()
    crea_grafico_opt_b2_latenza_congiunto()
    crea_grafico_opt_b2_tokens_congiunto()
    crea_grafico_opt_b3_latenza_congiunto()
    print("--- Completato ---\n")

if __name__ == "__main__":
    genera_tutti_grafici_congiunti()
