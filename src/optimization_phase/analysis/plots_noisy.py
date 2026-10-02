import os
import matplotlib.pyplot as plt
import numpy as np
from optimization_phase.analysis.plots_utils import _salva


_TIPO_COLORI = {
    "single_doc":  "#4C72B0",  
    "multi_doc":   "#55A868",  
    "noisy":       "#DD8452",  
    "unanswerable":"#C44E52",  
    "all":         "#8172B2",  
}
_TIPO_LABEL = {
    "single_doc":  "Single-doc",
    "multi_doc":   "Multi-doc",
    "noisy":       "Noisy",
    "unanswerable":"Unanswerable",
    "all":         "Tutti",
}


def _estrai_recall_per_tipo(dati: list[dict], k: int) -> dict[str, float]:
    """Estrae DocRecall@k per ogni tipo di query usando la configurazione migliore."""
    if not dati:
        return {}

    best = max(dati, key=lambda r: (r.get("all", {}) or {}).get(f"DocRecall@{k}", 0) or 0)
    tipi = ["single_doc", "multi_doc", "noisy", "unanswerable"]
    out = {}
    for tipo in tipi:
        v = (best.get(tipo, {}) or {}).get(f"DocRecall@{k}")
        if v is not None:
            out[tipo] = v
    return out


def plot_noisy_per_block2(dati: list[dict], out_dir: str) -> None:
    """Genera grafico a barre sulla resilienza alle query noisy per ogni strategia QT (Blocco 2)."""
    if not dati:
        return

    k = 5
    tipi_da_mostrare = ["single_doc", "multi_doc", "noisy"]

    ha_noisy = any(
        (r.get("noisy", {}) or {}).get(f"DocRecall@{k}") is not None
        for r in dati
    )
    if not ha_noisy:
        print("  [B2-noisy] Nessun dato 'noisy' nei risultati — grafico saltato.")
        print("             (Il grafico apparira dopo aver rieseguito il Blocco 2 con 90_RFC.json)")
        return

    etichette_qt = [r["qt_type"] for r in dati]
    n_qt = len(etichette_qt)
    n_tipi = len(tipi_da_mostrare)

    matrice = {}
    for tipo in tipi_da_mostrare:
        matrice[tipo] = [
            (r.get(tipo, {}) or {}).get(f"DocRecall@{k}") or 0
            for r in dati
        ]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(n_qt)
    w = 0.22
    offsets = np.linspace(-(n_tipi - 1) / 2 * w, (n_tipi - 1) / 2 * w, n_tipi)

    for tipo, offs in zip(tipi_da_mostrare, offsets):
        vals  = matrice[tipo]
        col   = _TIPO_COLORI.get(tipo, "#999999")
        label = _TIPO_LABEL.get(tipo, tipo)
        hatch = "//" if tipo == "noisy" else ""
        barre = ax.bar(
            x + offs, vals, w,
            label=label,
            color=col, alpha=0.85,
            edgecolor="white", linewidth=0.5,
            hatch=hatch,
        )
        for barra, val in zip(barre, vals):
            if val > 0.002:
                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 0.008,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, rotation=40,
                    color=col, fontweight="bold",
                )

    y_max = max(
        max(v for v in vals if v is not None) if matrice else 0
        for vals in matrice.values()
    )
    ax.set_ylim(0, max(y_max, 0.1) * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels(etichette_qt, fontsize=12)
    ax.set_ylabel(f"DocRecall@{k}", fontsize=11)
    ax.set_xlabel("Strategia di Query Transformation", fontsize=11)
    ax.set_title(
        "Blocco 2 — Resilienza alle Query Noisy per Strategia QT\n"
        "Confronto DocRecall@5: Single-doc  |  Multi-doc  |  Noisy (tratteggio)",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    _salva(os.path.join(out_dir, "block2_noisy.png"))


def plot_noisy_analysis(dati_blocchi: dict, out_dir: str) -> None:
    """Grafico cross-blocco sull'impatto delle query noisy usando le config migliori di ogni blocco."""
    k = 5
    tipi = ["single_doc", "multi_doc", "noisy"]

    recall_per_blocco: list[tuple[str, dict]] = []

    for b_key, b_nome in [("1", "Blocco 1\n(chunking ottimale)"),
                           ("2", "Blocco 2\n(QT ottimale)"),
                           ("3", "Blocco 3\n(retrieval ottimale)")]:
        dati = dati_blocchi.get(b_key, [])
        if not dati:
            continue

        best = max(
            dati,
            key=lambda r: (r.get("all", {}) or {}).get(f"DocRecall@{k}", 0) or 0,
        )
        recall_tipi = {}
        for tipo in tipi:
            v = (best.get(tipo, {}) or {}).get(f"DocRecall@{k}")
            if v is not None:
                recall_tipi[tipo] = v
        if recall_tipi:
            recall_per_blocco.append((b_nome, recall_tipi))

    ha_noisy = any("noisy" in rd for _, rd in recall_per_blocco)
    if not recall_per_blocco or not ha_noisy:
        print("  [noisy] Nessun dato noisy disponibile — grafico noisy_analysis saltato.")
        print("          Riesegui i blocchi con 90_RFC.json per generarlo.")
        return

    n_blocchi = len(recall_per_blocco)
    n_tipi    = len(tipi)
    w = 0.22
    x = np.arange(n_blocchi)
    offsets = np.linspace(-(n_tipi - 1) / 2 * w, (n_tipi - 1) / 2 * w, n_tipi)

    fig, ax = plt.subplots(figsize=(max(10, n_blocchi * 3.5), 6))

    for tipo, offs in zip(tipi, offsets):
        vals  = [(rd.get(tipo) or 0) for _, rd in recall_per_blocco]
        col   = _TIPO_COLORI.get(tipo, "#999999")
        label = _TIPO_LABEL.get(tipo, tipo)
        hatch = "//" if tipo == "noisy" else ""
        barre = ax.bar(
            x + offs, vals, w,
            label=label,
            color=col, alpha=0.85,
            edgecolor="white", linewidth=0.5,
            hatch=hatch,
        )
        for barra, val in zip(barre, vals):
            if val > 0.002:
                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9, color=col, fontweight="bold",
                )

    etichette_b = [nome for nome, _ in recall_per_blocco]
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)
    all_vals = [
        v for _, rd in recall_per_blocco for v in rd.values() if v is not None
    ]
    ax.set_ylim(0, max(all_vals, default=1.0) * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels(etichette_b, fontsize=10)
    ax.set_ylabel(f"DocRecall@{k}", fontsize=11)
    ax.set_title(
        "Analisi Resilienza Query Noisy — Confronto Cross-Blocco\n"
        "DocRecall@5 per tipo di query: Standard vs Noisy (tratteggio arancione)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)


    plt.tight_layout()
    _salva(os.path.join(out_dir, "noisy_analysis.png"))


