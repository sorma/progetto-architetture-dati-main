import os
import matplotlib.pyplot as plt
import numpy as np
from optimization_phase.analysis.plots_utils import COLORI, _salva, _val


def plot_block1(dati: list[dict], out_dir: str) -> None:
    """
    Genera i grafici per il Blocco 1: metriche di retrieval, latenza, e tempi di prebuild.
    """
    if not dati:
        print("  [B1] Nessun dato — salto.")
        return

    k = 5
    n = len(dati)

    # Etichette asse X: "C1\nsent-256"
    etichette_brevi = [
        f"C{r['config']['id']}\n{r['config']['splitter'][:4]}-{r['config']['chunk_size']}"
        for r in dati
    ]

    # Modello embedding abbreviato per annotazione sotto l'asse
    note_embedding = []
    for r in dati:
        m = r["config"]["embed_model"].split("/")[-1]
        m = (m.replace("all-MiniLM-L6-v2", "MiniLM-L6")
              .replace("bge-base-en-v1.5",  "bge-base")
              .replace("bge-large-en-v1.5", "bge-large"))
        note_embedding.append(m)

    metriche_plot = ["DocRecall", "PassageRecall", "R-Precision"]
    valori = {m: [_val(r, m, k) for r in dati] for m in metriche_plot}

    # Grafico 1: Metriche
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(n)
    n_m = len(metriche_plot)
    w = 0.15
    offsets = np.linspace(-(n_m - 1) / 2 * w, (n_m - 1) / 2 * w, n_m)

    for m, offs in zip(metriche_plot, offsets):
        vals = valori[m]
        barre = ax.bar(x + offs, vals, w, label=m, color=COLORI[m], alpha=0.88, edgecolor="white", linewidth=0.4)
        for barra, val in zip(barre, vals):
            if val > 0.005:
                ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=60)

    y_max = max(max(v) for v in valori.values()) * 1.18
    ax.set_ylim(0, max(y_max, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(etichette_brevi, fontsize=9)
    ax.set_ylabel(f"Valore metrica @{k}")
    ax.set_title(f"Blocco 1 — Metriche di Retrieval per Configurazione (k={k})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # Modello embedding sotto ogni gruppo di barre
    for pos, nota in zip(x, note_embedding):
        ax.annotate(
            nota,
            xy=(pos, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -38), textcoords="offset points",
            ha="center", va="top", fontsize=7.5, color="#555555",
            annotation_clip=False,
        )

    plt.subplots_adjust(bottom=0.18)
    _salva(os.path.join(out_dir, "block1_metriche.png"))

    # Grafico 2: Latenza media di retrieval
    latenze = [r.get("avg_retrieval_sec", 0) for r in dati]
    if any(l > 0 for l in latenze):
        fig, ax = plt.subplots(figsize=(12, 5))
        barre = ax.bar(etichette_brevi, latenze, color=COLORI["DocRecall"], alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        for barra, t in zip(barre, latenze):
            ax.text(barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 0.01,
                    f"{t:.3f}s", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Latenza media per query (secondi)")
        ax.set_title("Blocco 1 — Latenza Media di Retrieval per Configurazione")
        ax.set_ylim(0, max(latenze) * 1.25)
        plt.tight_layout()
        _salva(os.path.join(out_dir, "block1_latenza.png"))
    else:
        print("  [B1] avg_retrieval_sec tutti 0 — grafico latenza saltato.")

    # Grafico 3: Tempi di embedding e chunking
    tempi_emb   = [r.get("embed_time_sec", 0)  for r in dati]
    tempi_chunk = [r.get("chunk_time_sec", 0)  for r in dati]

    if any(t > 0 for t in tempi_emb + tempi_chunk):
        fig, ax = plt.subplots(figsize=(12, 5))
        w2 = 0.35
        xp = np.arange(n)
        b_emb   = ax.bar(xp - w2/2, tempi_emb,   w2, label="Embedding", color=COLORI["DocNDCG"],     alpha=0.85)
        b_chunk = ax.bar(xp + w2/2, tempi_chunk,  w2, label="Chunking",  color=COLORI["PassageNDCG"], alpha=0.85)
        for barre, vals in [(b_emb, tempi_emb), (b_chunk, tempi_chunk)]:
            for barra, t in zip(barre, vals):
                if t > 0:
                    ax.text(barra.get_x() + barra.get_width() / 2,
                            barra.get_height() + 1,
                            f"{t:.0f}s", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xp)
        ax.set_xticklabels(etichette_brevi, fontsize=9)
        ax.set_ylabel("Tempo (secondi)")
        ax.set_title("Blocco 1 — Tempi di Chunking ed Embedding per Configurazione")
        ax.legend()
        plt.tight_layout()
        _salva(os.path.join(out_dir, "block1_tempi.png"))
    else:
        print("  [B1] Tempi embedding/chunk tutti 0 "
              "(indici caricati da cache senza file block1_prebuild_times.json) "
              "— grafico tempi saltato.")


# =============================================================================
