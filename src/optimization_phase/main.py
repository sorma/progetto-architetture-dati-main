"""Script principale di orchestrazione dell'ottimizzazione RAG."""

import argparse
import json
import os
import sys
import time

from optimization_phase.configs.config import (
    DATASET_PATH,
    DOCS_DIR,
    RESULTS_DIR,
    RESULTS_DIR,
    RAW_RESULTS_DIR,
    PREBUILD_DIR,
    DEFAULT_K,
)


def carica_dataset(path: str) -> list[dict]:
    """Carica il dataset di domande dal file JSON specificato."""
    if not os.path.exists(path):
        print(f"[ERRORE] File dataset non trovato: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Statistiche del dataset
    tipi = {}
    for item in dataset:
        t = item.get("type", "unknown")
        tipi[t] = tipi.get(t, 0) + 1

    print(f"[dataset] Caricato: {len(dataset)} domande da '{path}'")
    for tipo, conteggio in sorted(tipi.items()):
        print(f"  - {tipo}: {conteggio} domande")

    return dataset


def mostra_configurazione() -> None:
    """Stampa la configurazione attuale del sistema per verifica."""
    from optimization_phase.configs.config import (
        DOCS_DIR, DATASET_PATH, RESULTS_DIR, EMBEDDING_DEVICE,
        EMBED_BATCH_SIZE, GROQ_API_KEY, GROQ_QT_MODEL, PASSAGE_EVAL_MODEL,
        PASSAGE_SIM_THRESHOLD, USE_LOCAL_RERANKER, DEFAULT_K
    )
    print("\n" + "="*60)
    print("CONFIGURAZIONE SISTEMA RAG OPTIMIZER")
    print("="*60)
    print(f"  Knowledge base:    {DOCS_DIR}")
    print(f"  Dataset domande:   {DATASET_PATH}")
    print(f"  Risultati output:  {RESULTS_DIR}")
    print(f"  Dispositivo embed: {EMBEDDING_DEVICE}")
    print(f"  Batch size embed:  {EMBED_BATCH_SIZE}")
    print(f"  Modello Groq QT:   {GROQ_QT_MODEL}")
    print(f"  API Groq:          {'✓ configurata' if GROQ_API_KEY != 'YOUR_GROQ_API_KEY' else '⚠ NON configurata'}")
    print(f"  Modello eval pass: {PASSAGE_EVAL_MODEL} (soglia: {PASSAGE_SIM_THRESHOLD})")
    print(f"  Reranker locale:   {USE_LOCAL_RERANKER}")
    print(f"  k di default:      {DEFAULT_K}")
    print("="*60 + "\n")


def esegui_blocco1(dataset: list[dict], rebuild_index: bool = False):
    """Esegue il Blocco 1: ottimizzazione chunking & embedding salvando gli indici in cache su disco."""
    from optimization_phase.blocks.block1_chunking_embedding import run_block1, select_best_config, build_index, load_and_chunk_documents
    from optimization_phase.analysis.metrics import build_reference_passage_cache

    print("\n" + "#"*60)
    print("# BLOCCO 1: CHUNKING & EMBEDDING")
    print("#"*60)

    # Pre-calcola gli embedding dei reference passages con cache su disco
    build_reference_passage_cache(dataset)

    # Esegui il blocco 1 (salta le config già presenti in block1_results.json)
    risultati_b1 = run_block1(dataset, rebuild_index=rebuild_index)

    # Seleziona la configurazione migliore
    migliore_b1 = select_best_config(risultati_b1)
    config_migliore = migliore_b1["config"]

    # Carica l'indice della configurazione migliore (dalla cache su disco)
    print(f"\n[Blocco 1] Caricamento indice configurazione migliore (id={config_migliore['id']})...")
    nodi, _ = load_and_chunk_documents(config_migliore)
    indice_migliore, _ = build_index(config_migliore, nodi, rebuild=rebuild_index)

    print(f"\n✓ Blocco 1 completato. Configurazione migliore: Config {config_migliore['id']}")
    return indice_migliore, nodi, config_migliore, risultati_b1


def esegui_blocco2(dataset: list[dict], indice_b1, config_b1: dict):
    """Esegue il Blocco 2: ottimizzazione query transformation restituendo la strategia migliore."""
    from optimization_phase.blocks.block2_query_transformation import run_block2, select_best_qt

    print("\n" + "#"*60)
    print("# BLOCCO 2: QUERY TRANSFORMATION")
    print("#"*60)

    # Verifica che la chiave API Groq sia configurata
    from optimization_phase.configs.config import GROQ_API_KEY
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        print("\n[ATTENZIONE] GROQ_API_KEY non configurata.")
        print("  Imposta la variabile d'ambiente GROQ_API_KEY oppure modifica config.py.")
        print("  Verrà testata solo la strategia 'none'.")

    risultati_b2 = run_block2(dataset, indice_b1)
    migliore_qt = select_best_qt(risultati_b2)

    print(f"\n✓ Blocco 2 completato. Strategia migliore: '{migliore_qt}'")
    return migliore_qt, risultati_b2


def esegui_blocco3(
    dataset: list[dict],
    indice_b1,
    nodi_b1: list,
    migliore_qt: str,
):
    """Esegue il Blocco 3: ottimizzazione retrieval, k e reranking restituendo la config migliore."""
    from optimization_phase.blocks.block3_retrieval_reranking import run_block3, select_best_config_b3

    print("\n" + "#"*60)
    print("# BLOCCO 3: RETRIEVAL, K E RERANKING")
    print("#"*60)

    # I nodi sono necessari per il retriever BM25 (retrieval ibrido)
    risultati_b3 = run_block3(
        dataset=dataset,
        best_b1_index=indice_b1,
        best_b2_qt=migliore_qt,
        nodi_per_bm25=nodi_b1,
    )

    migliore_b3 = select_best_config_b3(risultati_b3)

    print(f"\n✓ Blocco 3 completato. Configurazione migliore: Config {migliore_b3['config']['id']}")
    return migliore_b3, risultati_b3


def salva_parametri_ottimali(
    config_b1: dict,
    migliore_qt: str,
    config_b3: dict,
) -> None:
    """Salva i parametri ottimali finali in results/best_params.json come punto di partenza per la Fase Generativa."""
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    c3 = config_b3["config"]

    best_params = {
        # Dal Blocco 1
        "splitter": config_b1["splitter"],
        "chunk_size": config_b1["chunk_size"],
        "chunk_overlap": config_b1["overlap"],
        "embed_model": config_b1["embed_model"],
        # Dal Blocco 2
        "qt_strategy": migliore_qt,
        # Dal Blocco 3
        "retrieval_type": c3["retrieval"],
        "k_retrieve": c3["k_retrieve"],
        "reranking": c3["rerank"],
        "k_final": c3["k_final"],
    }

    percorso = os.path.join(RAW_RESULTS_DIR, "best_params.json")
    with open(percorso, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("PARAMETRI OTTIMALI FINALI (raw/best_params.json):")
    print("="*60)
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")
    print("="*60)
    print(f"File salvato: {percorso}")


def main():
    """Funzione principale. Gestisce gli argomenti CLI e orchestra i blocchi."""
    parser = argparse.ArgumentParser(
        description="Sistema di ottimizzazione parametri RAG — Fase 1"
    )
    parser.add_argument(
        "--solo-blocco",
        type=int,
        choices=[1, 2, 3],
        help="Esegui solo il blocco specificato (1, 2 o 3).",
    )
    parser.add_argument(
        "--solo-grafici",
        action="store_true",
        help="Genera solo i grafici dai file di risultati esistenti.",
    )
    parser.add_argument(
        "--prebuild",
        action="store_true",
        help=(
            "Esegui solo il chunking e il calcolo degli embedding per tutte le 7 "
            "configurazioni del Blocco 1, salvando gli indici su disco. "
            "Non esegue retrieval né calcola metriche. "
            "Usa questo comando prima di avviare l'ottimizzazione completa."
        ),
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help=(
            "Forza il ricalcolo di tutti gli embedding del Blocco 1, "
            "cancellando la cache degli indici su disco. "
            "Usa questo flag se hai cambiato hardware o vuoi verificare la riproducibilità."
        ),
    )
    args = parser.parse_args()

    # Mostra la configurazione attuale
    mostra_configurazione()

    # --- Modalità: solo grafici ---
    if args.solo_grafici:
        from optimization_phase.analysis.plots_main import genera_tutti_i_grafici
        genera_tutti_i_grafici()
        return

    # Pre-costruzione indici (chunking + embedding per tutte e 7 le config) senza retrieval
    if args.prebuild:
        from optimization_phase.blocks.block1_chunking_embedding import prebuild_all_indexes
        prebuild_all_indexes(rebuild=args.rebuild_index)
        print("\nQuando sei pronto, avvia l'ottimizzazione completa con:")
        print("  run.bat              (tutti e 3 i blocchi)")
        print("  run.bat --solo-blocco 1  (solo valutazione metriche Blocco 1)")
        return

    # Carica il dataset di domande
    dataset = carica_dataset(DATASET_PATH)

    tempo_inizio = time.time()

    # --- Modalità Esecuzione Blocchi Specifici ---
    if args.solo_blocco == 1:
        indice_b1, nodi_b1, config_b1, _ = esegui_blocco1(dataset, rebuild_index=args.rebuild_index)

    elif args.solo_blocco == 2:
        # Carica l'indice migliore del Blocco 1 dalla cache su disco
        b1_path = os.path.join(RAW_RESULTS_DIR, "block1_results.json")
        if not os.path.exists(b1_path):
            print("[ERRORE] Blocco 1 non ancora eseguito. Prima esegui il Blocco 1.")
            sys.exit(1)
        with open(b1_path, "r", encoding="utf-8") as f:
            risultati_b1 = json.load(f)
        from optimization_phase.blocks.block1_chunking_embedding import select_best_config, build_index, load_and_chunk_documents
        config_b1 = select_best_config(risultati_b1)["config"]
        nodi_b1, _ = load_and_chunk_documents(config_b1)
        # build_index caricherà automaticamente l'indice dalla cache su disco
        indice_b1, _ = build_index(config_b1, nodi_b1)
        esegui_blocco2(dataset, indice_b1, config_b1)

    elif args.solo_blocco == 3:
        # Carica i risultati dei blocchi precedenti
        b1_path = os.path.join(RAW_RESULTS_DIR, "block1_results.json")
        b2_path = os.path.join(RAW_RESULTS_DIR, "block2_results.json")
        for p in [b1_path, b2_path]:
            if not os.path.exists(p):
                print(f"[ERRORE] File non trovato: {p}. Esegui prima i blocchi precedenti.")
                sys.exit(1)
        with open(b1_path, "r", encoding="utf-8") as f:
            risultati_b1 = json.load(f)
        with open(b2_path, "r", encoding="utf-8") as f:
            risultati_b2 = json.load(f)
        from optimization_phase.blocks.block1_chunking_embedding import select_best_config, build_index, load_and_chunk_documents
        from optimization_phase.blocks.block2_query_transformation import select_best_qt
        config_b1 = select_best_config(risultati_b1)["config"]
        nodi_b1, _ = load_and_chunk_documents(config_b1)
        # build_index caricherà automaticamente l'indice dalla cache su disco
        indice_b1, _ = build_index(config_b1, nodi_b1)
        migliore_qt = select_best_qt(risultati_b2)
        migliore_b3, _ = esegui_blocco3(dataset, indice_b1, nodi_b1, migliore_qt)
        salva_parametri_ottimali(config_b1, migliore_qt, migliore_b3)

    else:
        # Esecuzione sequenziale di tutti i blocchi
        indice_b1, nodi_b1, config_b1, _ = esegui_blocco1(dataset, rebuild_index=args.rebuild_index)

        # Blocco 2
        migliore_qt, _ = esegui_blocco2(dataset, indice_b1, config_b1)

        # Blocco 3
        migliore_b3, _ = esegui_blocco3(dataset, indice_b1, nodi_b1, migliore_qt)

        # Salva i parametri ottimali
        salva_parametri_ottimali(config_b1, migliore_qt, migliore_b3)

    # Genera i grafici al termine
    from optimization_phase.analysis.plots_main import genera_tutti_i_grafici
    genera_tutti_i_grafici()

    tempo_totale = time.time() - tempo_inizio
    ore = int(tempo_totale // 3600)
    minuti = int((tempo_totale % 3600) // 60)
    secondi = int(tempo_totale % 60)
    print(f"\n✓ Ottimizzazione completata in {ore}h {minuti}m {secondi}s")
    print(f"  Risultati salvati in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
