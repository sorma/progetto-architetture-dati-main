import os
import glob
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
    "RFC": "#4C72B0",
    "WW1": "#C44E52" 
}

def _salva(path: str) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [OK] Salvato: {path}")

def calcola_latenza(domain: str) -> dict:
    pattern = os.path.join(PROJECT_ROOT, "results", domain, "generation", "raw", "config_C*_results.json")
    files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('_')[1][1:]))
    
    latencies = {}
    for f in files:
        config_name = os.path.basename(f).split('_')[1]
        try:
            with open(f, encoding='utf-8') as file:
                data = json.load(file)
            lats = [d.get('latency_sec', 0) for d in data if 'latency_sec' in d]
            avg_lat = sum(lats) / len(lats) if lats else 0
            latencies[config_name] = avg_lat
        except Exception as e:
            print(f"Errore caricamento {f}: {e}")
            
    return latencies

def crea_grafico_latenza():
    print("Calcolo latenze...")
    lat_rfc = calcola_latenza("rfc")
    lat_ww1 = calcola_latenza("ww1")
    
    if not lat_rfc or not lat_ww1:
        print("Mancano i dati per uno o più domini.")
        return

    configs = list(lat_rfc.keys())
    
    val_rfc = [lat_rfc[c] for c in configs]
    val_ww1 = [lat_ww1[c] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width/2, val_rfc, width, label='RFC', color=COLORS["RFC"], zorder=3)
    rects2 = ax.bar(x + width/2, val_ww1, width, label='WW1', color=COLORS["WW1"], zorder=3)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylim(0, max(max(val_rfc), max(val_ww1)) * 1.25)
    
    ax.set_ylabel("Latenza Media (secondi)")
    ax.set_xlabel("Configurazioni")
    ax.set_title("Latenza Media di Generazione per Dominio", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(title='Dominio', loc='upper left')
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generation_latenza.png")
    
    _salva(out_path)

if __name__ == "__main__":
    crea_grafico_latenza()
