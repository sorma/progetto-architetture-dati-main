import os
import json
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    "Accuracy": "#4C72B0",
    "Precision": "#55A868",
    "Recall": "#C44E52",
    "F1 Score": "#8172B2"
}

def _salva(path: str) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [OK] Salvato: {path}")

def carica_metriche_complete(domain: str) -> dict:
    percorso = os.path.join(PROJECT_ROOT, "results", domain, "generation", "aggregated_metrics.json")
    metrics = {}
    if os.path.exists(percorso):
        with open(percorso, encoding='utf-8') as f:
            data = json.load(f)
            data.sort(key=lambda x: int(x['config_id'][1:]))
            for riga in data:
                metrics[riga['config_id']] = riga
    return metrics

def crea_grafico_dominio(domain: str):
    print(f"Generazione grafico metriche per {domain.upper()}...")
    metrics = carica_metriche_complete(domain)
    
    if not metrics:
        print(f"Nessun dato per {domain}")
        return

    configs = list(metrics.keys())
    
    acc_scores = [metrics[c]["accuracy"] * 100 for c in configs]
    prec_scores = [metrics[c]["precision"] * 100 for c in configs]
    rec_scores = [metrics[c]["recall"] * 100 for c in configs]
    f1_scores = [metrics[c]["f1"] * 100 for c in configs]

    x = np.arange(len(configs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    
    rects1 = ax.bar(x - 1.5*width, acc_scores, width, label='Accuracy', color=COLORS["Accuracy"], zorder=3)
    rects2 = ax.bar(x - 0.5*width, prec_scores, width, label='Precision', color=COLORS["Precision"], zorder=3)
    rects3 = ax.bar(x + 0.5*width, rec_scores, width, label='Recall', color=COLORS["Recall"], zorder=3)
    rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color=COLORS["F1 Score"], zorder=3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    ax.set_ylim(0, 115)
    ax.set_ylabel("Punteggio (%)")
    ax.set_xlabel("Configurazioni")
    ax.set_title(f"Metriche di Valutazione (Acc, Prec, Rec, F1) - Fase Generativa (Dominio {domain.upper()})", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    
    ax.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    
    out_dir = os.path.join(PROJECT_ROOT, "results", domain, "generation", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generation_metrics.png")
    
    _salva(out_path)

def crea_grafico_latenza(domain: str):
    metrics = carica_metriche_complete(domain)
    if not metrics: return
    
    configs = list(metrics.keys())
    latenze = [metrics[c]["avg_latency_sec"] for c in configs]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    barre = ax.bar(configs, latenze, color="#3498db", alpha=0.85, edgecolor="white", linewidth=0.5)
    for barra, t in zip(barre, latenze):
        ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.1,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=9)
                
    ax.set_ylabel("Latenza Media (Secondi)")
    ax.set_xlabel("Configurazioni")
    ax.set_title(f"Latenza Media di Risposta - Fase Generativa (Dominio {domain.upper()})", pad=15)
    ax.set_ylim(0, max(latenze)*1.25 if latenze else 1)
    
    out_dir = os.path.join(PROJECT_ROOT, "results", domain, "generation", "plots")
    os.makedirs(out_dir, exist_ok=True)
    _salva(os.path.join(out_dir, "generation_latenza.png"))



def crea_grafico_matrice(domain: str):
    metrics = carica_metriche_complete(domain)
    if not metrics: return
    
    configs = list(metrics.keys())
    metric_keys = ["TP", "TN", "FP", "FN"]
    colors = {"TP": "#55A868", "TN": "#4C72B0", "FP": "#C44E52", "FN": "#3B2E5A"}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(configs))
    w = 0.6
    bottoms = np.zeros(len(configs))
    
    for m in metric_keys:
        vals = [metrics[c].get(m, 0) for c in configs]
        rects = ax.bar(x, vals, w, label=m, color=colors[m], bottom=bottoms, edgecolor='white', linewidth=0.5)
        for rect in rects:
            h = rect.get_height()
            if h > 3:
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_y() + h/2,
                        f"{int(h)}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        bottoms += vals
        
    ax.set_ylabel("Quantità (n° domande)")
    ax.set_xlabel("Configurazioni")
    ax.set_title(f"Matrice di Confusione - Fase Generativa (Dominio {domain.upper()})", pad=15)
    ax.set_ylim(0, 160)
    ax.set_xlim(-0.5, len(configs) + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc="lower right", framealpha=0.9)
    
    out_dir = os.path.join(PROJECT_ROOT, "results", domain, "generation", "plots")
    os.makedirs(out_dir, exist_ok=True)
    _salva(os.path.join(out_dir, "generation_confusion.png"))

def genera_tutti_i_grafici(gen_results_dir: str):
    domain = os.path.basename(os.path.dirname(gen_results_dir))
    crea_grafico_dominio(domain)
    crea_grafico_latenza(domain)
    crea_grafico_matrice(domain)

if __name__ == "__main__":
    crea_grafico_dominio("rfc")
    crea_grafico_dominio("ww1")
