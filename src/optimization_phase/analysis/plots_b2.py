import os
import matplotlib.pyplot as plt
import numpy as np
from optimization_phase.analysis.plots_utils import COLORI, _salva, _val
from optimization_phase.analysis.plots_noisy import plot_noisy_per_block2


def plot_block2(dati: list[dict], out_dir: str) -> None:
    """
    Genera i grafici per il Blocco 2: metriche, latenza media, consumo token e analisi noisy.
    """
    if not dati:
        print("  [B2] Nessun dato — salto.")
        return

    k = 5
    etichette = [r["qt_type"] for r in dati]
    n = len(etichette)

    metriche_plot = ["DocRecall", "PassageRecall", "R-Precision"]
    valori = {m: [_val(r, m, k) for r in dati] for m in metriche_plot}
    
    dr = valori["DocRecall"]

    # Grafico 1: Metriche
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(n)
    n_m = len(metriche_plot)
    w = 0.15
    offsets = np.linspace(-(n_m - 1) / 2 * w, (n_m - 1) / 2 * w, n_m)

    for m, offs in zip(metriche_plot, offsets):
        vals = valori[m]
        barre = ax.bar(x + offs, vals, w, label=m, color=COLORI[m], alpha=0.88, edgecolor="white", linewidth=0.4)
        for barra, val in zip(barre, vals):
            if val > 0.005:
                ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=45)

    y_max = max(max(v) for v in valori.values()) * 1.18
    ax.set_ylim(0, max(y_max, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, fontsize=11)
    ax.set_ylabel(f"Valore metrica @{k}")
    ax.set_title(f"Blocco 2 — Metriche per Strategia di Query Transformation (k={k})")
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    _salva(os.path.join(out_dir, "block2_metriche.png"))

    # Grafico 2: Latenza
    latenze = [r.get("avg_latency_sec", 0) for r in dati]

    fig, ax = plt.subplots(figsize=(10, 5))
    barre = ax.bar(etichette, latenze, color=COLORI["latenza"], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    for barra, l in zip(barre, latenze):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.05,
                f"{l:.2f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Latenza media per query (secondi)")
    ax.set_title("Blocco 2 — Latenza Media per Strategia di Query Transformation")
    ax.set_ylim(0, max(latenze) * 1.25 if latenze else 1)
    plt.tight_layout()
    _salva(os.path.join(out_dir, "block2_latenza.png"))

    # Grafico 3: Token consumati per Query (Input vs Output)
    in_tok = np.array([r.get("avg_input_tokens", 0) for r in dati])
    out_tok = np.array([r.get("avg_output_tokens", 0) for r in dati])
    tot_tok = in_tok + out_tok

    if any(t > 0 for t in tot_tok):
        fig, ax = plt.subplots(figsize=(10, 5))
        w = 0.5
        x = np.arange(len(etichette))
        
        ax.bar(x, in_tok, w, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.5, label="Input Tokens")
        ax.bar(x, out_tok, w, bottom=in_tok, color="#4C72B0", alpha=0.45, edgecolor="white", linewidth=0.5, label="Output Tokens")
            
        for i in range(len(etichette)):
            if tot_tok[i] > 0:
                ax.text(i, tot_tok[i] + max(tot_tok)*0.01, f"{tot_tok[i]:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                ax.text(i, in_tok[i]/2, f"{in_tok[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
                ax.text(i, in_tok[i] + out_tok[i]/2, f"{out_tok[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
                
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(etichette, fontsize=11)
        ax.set_ylabel("Token LLM medi per query")
        ax.set_title("Blocco 2 — Consumo Medio Token LLM (Input vs Output)")
        ax.set_ylim(0, max(tot_tok) * 1.25 if max(tot_tok) else 1)
        plt.tight_layout()
        _salva(os.path.join(out_dir, "block2_tokens.png"))

    # Grafico 4: Noisy vs Standard per strategia QT
    plot_noisy_per_block2(dati, out_dir)


# =============================================================================
