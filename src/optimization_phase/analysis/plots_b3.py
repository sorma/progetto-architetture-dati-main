import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from optimization_phase.analysis.plots_utils import COLORI, _salva, _val


def plot_block3(dati: list[dict], out_dir: str) -> None:
    """
    Genera i grafici per il Blocco 3: metriche di retrieval & reranking, e latenza media.
    """
    if not dati:
        print("  [B3] Nessun dato — salto.")
        return

    n = len(dati)

    def _etichetta(r):
        c = r["config"]
        rerank = "+RR" if c["rerank"] else ""
        return f"C{c['id']}\n{c['retrieval'][:3]}-k{c['k_retrieve']}{rerank}\n->k{c['k_final']}"

    etichette = [_etichetta(r) for r in dati]

    # Raggruppa per k_final per la separazione visiva
    gruppi_k = {}
    for i, r in enumerate(dati):
        kf = r["config"]["k_final"]
        gruppi_k.setdefault(kf, []).append(i)

    bg_colori = ["#EEF4FF", "#FFF4EE", "#F0FFF0", "#FFF8EE"]

    metriche_plot = ["DocRecall", "PassageRecall", "DocNDCG", "PassageNDCG", "R-Precision"]
    valori = {
        m: [r.get("all", {}).get(f"{m}@{r['config']['k_final']}", 0) or 0 for r in dati]
        for m in metriche_plot
    }

    # Grafico 1: Grouped bar chart con separazione per k
    fig, ax = plt.subplots(figsize=(18, 8))
    x   = np.arange(n)
    n_m = len(metriche_plot)
    w   = 0.14
    offsets = np.linspace(-(n_m - 1) / 2 * w, (n_m - 1) / 2 * w, n_m)

    for gi, (kf, idxs) in enumerate(sorted(gruppi_k.items())):
        x_left  = x[idxs[0]]  - 0.5
        x_right = x[idxs[-1]] + 0.5
        ax.text((x_left + x_right) / 2, -0.07, f"k_final = {kf}",
                ha="center", va="top", fontsize=8.5, color="#555",
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#ccc", alpha=0.7))

    for m, offs in zip(metriche_plot, offsets):
        vals = valori[m]
        barre = ax.bar(x + offs, vals, w, label=m, color=COLORI[m], alpha=0.85,
                       edgecolor="white", linewidth=0.4)
        for barra, val in zip(barre, vals):
            if val > 0.005:
                ax.text(barra.get_x() + barra.get_width() / 2,
                        barra.get_height() + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=70)

    y_max = max(max(v) for v in valori.values()) * 1.22
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, fontsize=8)
    ax.set_ylabel("Valore metrica @k_final")
    ax.set_title("Blocco 3 — Metriche per Configurazione di Retrieval & Reranking")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    plt.subplots_adjust(bottom=0.18)
    _salva(os.path.join(out_dir, "block3_metriche.png"))

    # Heatmap rimossa come richiesto

    ndcg_vals   = valori["DocNDCG"]
    recall_vals = valori["DocRecall"]
    has_rerank  = [r["config"]["rerank"] for r in dati]
    colori_bar  = ["#C44E52" if r else "#4C72B0" for r in has_rerank]
    patch_no = mpatches.Patch(color="#4C72B0", label="Senza reranking")
    patch_rr = mpatches.Patch(color="#C44E52", label="Con reranking")
    colori_sc = ["#C44E52" if r else "#4C72B0" for r in has_rerank]
    markers   = ["*" if r else "o" for r in has_rerank]
    sizes     = [220 if r else 140 for r in has_rerank]

    # Grafico 2: Latenza
    latenze = [r.get("avg_latency_sec", 0) for r in dati]
    if any(l > 0 for l in latenze):
        fig, ax = plt.subplots(figsize=(14, 5))
        barre = ax.bar(etichette, latenze, color=colori_bar, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        for barra, l in zip(barre, latenze):
            ax.text(barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + (max(latenze)*0.01),
                    f"{l:.2f}s", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylabel("Latenza media per query (secondi)")
        ax.set_title("Blocco 3 — Latenza Media (Retrieval + Reranking)\n"
                     "(rosso = con reranking, blu = senza)")
        ax.set_ylim(0, max(latenze) * 1.25)
        ax.legend(handles=[patch_no, patch_rr], fontsize=10)
        plt.tight_layout()
        _salva(os.path.join(out_dir, "block3_latenza.png"))




# =============================================================================
