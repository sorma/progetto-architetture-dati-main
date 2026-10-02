import matplotlib.pyplot as plt
import numpy as np


COLORI = {
    "DocRecall":    "#4C72B0",
    "PassageRecall":"#55A868",
    "R-Precision":  "#C44E52",
    "DocNDCG":      "#8172B2",
    "PassageNDCG":  "#CCB974",
    "latenza":      "#C44E52",
    "nessuno":      "#4C72B0",
}

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



def _salva(path: str) -> None:
    """Salva la figura corrente nella cartella indicata e la chiude."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [OK] Salvato: {path}")


def _val(r: dict, metrica: str, k: int) -> float:
    """Estrae il valore di una metrica@k dal dizionario 'all', con fallback a 0."""
    return r.get("all", {}).get(f"{metrica}@{k}", 0) or 0


