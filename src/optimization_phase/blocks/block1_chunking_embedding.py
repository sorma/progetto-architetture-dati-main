"""BLOCCO 1: Ottimizzazione di Chunking e Embedding."""

import json
import os
import time
import pickle
import re
import fitz  # PyMuPDF

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from optimization_phase.configs.config import (
    DOCS_DIR,
    RESULTS_DIR,
    RAW_RESULTS_DIR,
    PREBUILD_DIR,
    EMBEDDING_DEVICE,
    EMBED_BATCH_SIZE,
    DEFAULT_K,
    BLOCK1_CONFIGS,
)
from optimization_phase.analysis.metrics import evaluate_query, aggregate_metrics, build_reference_passage_cache


# Percorso del file di output del blocco 1
BLOCK1_OUTPUT = os.path.join(RAW_RESULTS_DIR, "block1_results.json")

# Prebuild directories per i nodi e gli indici di ciascuna config
BLOCK1_NODES_DIR = os.path.join(PREBUILD_DIR, "block1_nodes")

# Prebuild index directories
BLOCK1_INDEXES_DIR = os.path.join(PREBUILD_DIR, "block1_indexes")

# File per tenere traccia dei tempi di prebuild
BLOCK1_PREBUILD_TIMES = os.path.join(RAW_RESULTS_DIR, "block1_prebuild_times.json")


def _load_prebuild_times() -> dict:
    """
    Carica il dizionario dei tempi di prebuild dal file JSON su disco.
    Restituisce un dizionario vuoto se il file non esiste.

    Struttura del file:
        {
          "1": {"chunk_time_sec": 12.3, "embed_time_sec": 45.6},
          "2": {"chunk_time_sec": 10.1, "embed_time_sec": 60.2},
          ...
        }
    Le chiavi sono stringhe (config_id convertito a str).
    """
    if not os.path.exists(BLOCK1_PREBUILD_TIMES):
        return {}
    with open(BLOCK1_PREBUILD_TIMES, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_prebuild_times(tempi: dict) -> None:
    """
    Salva (o aggiorna) il dizionario dei tempi di prebuild su disco.

    Args:
        tempi: dizionario {str(config_id): {"chunk_time_sec": float, "embed_time_sec": float}}.
    """
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    esistenti = _load_prebuild_times()
    esistenti.update(tempi)
    with open(BLOCK1_PREBUILD_TIMES, "w", encoding="utf-8") as f:
        json.dump(esistenti, f, indent=2, ensure_ascii=False)


def get_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    """Inizializza un modello di embedding HuggingFace con fallback su CPU."""
    try:
        model = HuggingFaceEmbedding(
            model_name=model_name,
            device=EMBEDDING_DEVICE,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        print(f"  [embedding] Modello '{model_name}' caricato su {EMBEDDING_DEVICE}")
        return model
    except Exception as e:
        print(f"  [embedding] Errore su {EMBEDDING_DEVICE}: {e}. Fallback su CPU.")
        return HuggingFaceEmbedding(
            model_name=model_name,
            device="cpu",
            embed_batch_size=max(1, EMBED_BATCH_SIZE // 2),
        )


def get_splitter(config: dict):
    """Restituisce il parser di chunking appropriato in base alla configurazione."""
    if config["splitter"] == "sentence":
        return SentenceSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["overlap"],
        )
    elif config["splitter"] == "token":
        return TokenTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["overlap"],
        )
    elif config["splitter"] == "semantic":
        embed_model = get_embedding_model(config["embed_model"])
        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
    else:
        raise ValueError(f"Splitter sconosciuto: {config['splitter']}")


class FitzPDFReader:
    """Custom PDF reader using PyMuPDF (fitz) con pulizia del testo."""
    def load_data(self, file_path, extra_info=None):
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                t = page.get_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            print(f"  [PDFReader] Errore lettura {file_path} con fitz: {e}")
            
        # Basic sanitization: rimuoviamo solo gli a capo eccessivi
        text = self.sanitize_text(text)
        
        return [Document(text=text, metadata=extra_info or {})]
        
    def sanitize_text(self, text):
        # PyMuPDF estrae testo già molto pulito (no tag XML o artefatti binari).
        # Rimuovere i caratteri non-ASCII causerebbe perdita di dati (es. legature come 'fi', trattini lunghi, ecc.)
        # Ci limitiamo a compattare gli spazi vuoti eccessivi.
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def load_and_chunk_documents(config: dict) -> tuple[list, float]:
    """Carica i documenti e li suddivide in chunk salvandoli in cache."""
    os.makedirs(BLOCK1_NODES_DIR, exist_ok=True)
    # Identificatore univoco per questa configurazione (escludo embed_model dal nome file)
    config_id = config["id"]
    nodes_path = os.path.join(BLOCK1_NODES_DIR, f"config_{config_id}_nodes.pkl")

    # Se i nodi sono già stati calcolati, li carichiamo dalla cache
    if os.path.exists(nodes_path):
        print(f"  [cache] Nodi trovati su disco: {nodes_path}")
        with open(nodes_path, "rb") as f:
            nodi = pickle.load(f)
        # Recupera il tempo reale dal file di prebuild (se disponibile)
        tempi_prebuild = _load_prebuild_times()
        tempo_chunking = tempi_prebuild.get(str(config_id), {}).get("chunk_time_sec", 0.0)
        if tempo_chunking > 0.0:
            print(f"  [cache] Tempo chunking (da prebuild): {tempo_chunking:.1f}s")
        return nodi, tempo_chunking

    print(f"  [chunking] Caricamento documenti da: {DOCS_DIR}")
    extractor = {".pdf": FitzPDFReader()}
    documenti = SimpleDirectoryReader(DOCS_DIR, recursive=True, file_extractor=extractor).load_data()
    print(f"  [chunking] Documenti caricati: {len(documenti)}")

    parser = get_splitter(config)

    t0 = time.time()
    nodi = parser.get_nodes_from_documents(documenti, show_progress=True)
    tempo_chunking = time.time() - t0

    print(f"  [chunking] Chunk prodotti: {len(nodi)} in {tempo_chunking:.1f}s")

    # Salva su disco
    with open(nodes_path, "wb") as f:
        pickle.dump(nodi, f)

    return nodi, tempo_chunking


def build_index(config: dict, nodi: list, rebuild: bool = False) -> tuple:
    """Costruisce o ricarica un indice vettoriale persistendolo su disco."""
    os.makedirs(BLOCK1_INDEXES_DIR, exist_ok=True)
    indice_dir = os.path.join(BLOCK1_INDEXES_DIR, f"config_{config['id']}")

    # --- Caricamento dalla cache ---
    # Se l'indice è già stato calcolato e salvato, lo ricarichiamo direttamente.
    # Questo evita di rieseguire l'embedding (operazione costosa: minuti/ore).
    if os.path.exists(indice_dir) and not rebuild:
        print(f"  [cache] Indice trovato su disco: {indice_dir}")
        print(f"  [cache] Caricamento embedding dalla cache (nessun ricalcolo)...")
        # Dobbiamo impostare il modello di embedding PRIMA di caricare l'indice,
        # altrimenti LlamaIndex non saprebbe come eseguire nuove query.
        embed_model = get_embedding_model(config["embed_model"])
        Settings.embed_model = embed_model
        storage_context = StorageContext.from_defaults(persist_dir=indice_dir)
        indice = load_index_from_storage(storage_context)
        print(f"  [cache] Indice caricato in {len(nodi)} chunk — embedding già calcolati.")
        # Recupera il tempo reale dal file di prebuild (se disponibile)
        tempi_prebuild = _load_prebuild_times()
        tempo_embedding = tempi_prebuild.get(str(config["id"]), {}).get("embed_time_sec", 0.0)
        if tempo_embedding > 0.0:
            print(f"  [cache] Tempo embedding (da prebuild): {tempo_embedding:.1f}s")
        return indice, tempo_embedding

    # --- Calcolo degli embedding e costruzione dell'indice ---
    if rebuild and os.path.exists(indice_dir):
        print(f"  [rebuild] Ricalcolo forzato — rimozione cache esistente: {indice_dir}")
        import shutil
        shutil.rmtree(indice_dir)

    embed_model = get_embedding_model(config["embed_model"])
    Settings.embed_model = embed_model

    print(f"  [embedding] Calcolo embedding per {len(nodi)} chunk...")
    print(f"  [embedding] Modello: {config['embed_model']}")
    t0 = time.time()
    indice = VectorStoreIndex(nodi, show_progress=True)
    tempo_embedding = time.time() - t0

    print(f"  [embedding] Embedding completati in {tempo_embedding:.1f}s")

    # --- Persistenza su disco ---
    # Salva l'indice completo (chunk + vettori) nella sottocartella della config.
    # La prossima volta, load_index_from_storage() lo recupererà in pochi secondi.
    os.makedirs(indice_dir, exist_ok=True)
    indice.storage_context.persist(persist_dir=indice_dir)
    print(f"  [salvataggio] Indice salvato su disco: {indice_dir}")

    return indice, tempo_embedding


def prebuild_all_indexes(rebuild: bool = False) -> None:
    """Pre-costruisce gli indici vettoriali per le 7 config (caching per velocizzare le run)."""
    print(f"\n{'='*60}")
    print("PRE-COSTRUZIONE INDICI VETTORIALI (tutte le configurazioni)")
    print("Nessun retrieval né valutazione metriche — solo chunking + embedding")
    print(f"{'='*60}\n")

    os.makedirs(PREBUILD_DIR, exist_ok=True)
    os.makedirs(BLOCK1_NODES_DIR, exist_ok=True)
    os.makedirs(BLOCK1_INDEXES_DIR, exist_ok=True)

    tempo_inizio_totale = time.time()
    risultati_riepilogo = []

    for config in BLOCK1_CONFIGS:
        cid = config["id"]
        indice_dir = os.path.join(BLOCK1_INDEXES_DIR, f"config_{cid}")

        print(f"\n{'─'*60}")
        print(f"Config {cid}/{len(BLOCK1_CONFIGS)}: {config['note']}")
        print(f"  Splitter  : {config['splitter']}")
        print(f"  Chunk size: {config['chunk_size']}  Overlap: {config['overlap']}")
        print(f"  Embedding : {config['embed_model']}")
        print(f"{'─'*60}")

        # Controlla se l'indice è già stato costruito
        if os.path.exists(indice_dir) and not rebuild:
            print(f"  ✓ Indice già presente su disco — skip (usa --rebuild-index per forzare)")
            risultati_riepilogo.append({
                "config_id": cid,
                "stato": "già presente (skip)",
                "tempo_embedding_sec": 0.0,
            })
            continue

        # 1. Carica e chunka i documenti (con cache su disco)
        nodi, tempo_chunking = load_and_chunk_documents(config)
        print(f"  Chunk prodotti: {len(nodi)}")

        # 2. Calcola embedding e salva l'indice su disco
        _, tempo_embedding = build_index(config, nodi, rebuild=rebuild)

        # 3. Salva i tempi reali su disco (block1_prebuild_times.json)
        #    Questi vengono letti da run_block1 quando l'indice è in cache.
        _save_prebuild_times({
            str(cid): {
                "chunk_time_sec": round(tempo_chunking, 1),
                "embed_time_sec": round(tempo_embedding, 1),
            }
        })
        print(f"  [timing] Tempi salvati in: {BLOCK1_PREBUILD_TIMES}")

        risultati_riepilogo.append({
            "config_id": cid,
            "stato": "costruito",
            "n_chunk": len(nodi),
            "tempo_chunking_sec": round(tempo_chunking, 1),
            "tempo_embedding_sec": round(tempo_embedding, 1),
        })

    # Riepilogo finale
    tempo_totale = time.time() - tempo_inizio_totale
    ore = int(tempo_totale // 3600)
    minuti = int((tempo_totale % 3600) // 60)
    secondi = int(tempo_totale % 60)

    print(f"\n{'='*60}")
    print(f"PRE-COSTRUZIONE COMPLETATA in {ore}h {minuti}m {secondi}s")
    print(f"{'='*60}")
    print(f"  {'Config':<8} {'Stato':<25} {'Chunk':>7} {'T.embed':>9}")
    print(f"  {'─'*55}")
    for r in risultati_riepilogo:
        n_chunk = r.get("n_chunk", "─")
        t_emb = f"{r['tempo_embedding_sec']:.0f}s" if r["tempo_embedding_sec"] > 0 else "─"
        print(f"  {r['config_id']:<8} {r['stato']:<25} {str(n_chunk):>7} {t_emb:>9}")
    print(f"\n  Indici salvati in: {BLOCK1_INDEXES_DIR}")
    print(f"  Ora puoi eseguire l'ottimizzazione completa con: run.bat")
    print(f"  Gli embedding NON verranno ricalcolati.")


def run_block1(dataset: list[dict], rebuild_index: bool = False) -> list[dict]:
    """Esegue il Blocco 1 completo: chunking, embedding, retrieval e metriche per tutte le config."""
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # Carica i risultati già esistenti (esecuzione incrementale)
    risultati_esistenti: list[dict] = []
    if os.path.exists(BLOCK1_OUTPUT):
        with open(BLOCK1_OUTPUT, "r", encoding="utf-8") as f:
            risultati_esistenti = json.load(f)
    ids_completati = {r["config"]["id"] for r in risultati_esistenti}

    tutti_risultati = list(risultati_esistenti)

    for config in BLOCK1_CONFIGS:
        cid = config["id"]

        # Salta le configurazioni già eseguite (a meno che rebuild_index sia True)
        if cid in ids_completati and not rebuild_index:
            print(f"\n=== Config {cid} già eseguita — skip ===")
            continue

        print(f"\n{'='*60}")
        print(f"=== BLOCCO 1 — Config {cid}: {config['note']} ===")
        print(f"  Splitter: {config['splitter']}, chunk_size={config['chunk_size']}, overlap={config['overlap']}")
        print(f"  Embedding: {config['embed_model']}")
        print(f"{'='*60}")

        # 1. Carica e chunka i documenti (dalla cache su disco se disponibile)
        nodi, tempo_chunking = load_and_chunk_documents(config)

        # 2. Costruisce o carica l'indice con embedding
        #    Se l'indice esiste già su disco: caricato in pochi secondi, zero ricalcolo.
        #    Se non esiste: calcolato e poi salvato su disco per le run future.
        index, tempo_embedding = build_index(config, nodi, rebuild=rebuild_index)

        # 3. Retrieval per ogni domanda del dataset
        retriever = index.as_retriever(similarity_top_k=DEFAULT_K)
        valutazioni = []
        tempo_retrieval_totale = 0.0

        print(f"  [retrieval] Esecuzione retrieval su {len(dataset)} domande...")
        for i, item in enumerate(dataset):
            t0 = time.time()
            nodi_recuperati = retriever.retrieve(item["question"])
            tempo_retrieval_totale += time.time() - t0

            # Calcola le metriche per questa domanda
            eval_result = evaluate_query(
                retrieved_nodes=nodi_recuperati,
                item=item,
                k=DEFAULT_K,
                include_ndcg=False,  # NDCG non usato nel Blocco 1
            )
            valutazioni.append(eval_result)

            # Stampa progresso ogni 10 domande
            if (i + 1) % 10 == 0:
                print(f"    Domanda {i+1}/{len(dataset)} elaborata")

        # 4. Aggregazione delle metriche
        metriche = aggregate_metrics(valutazioni, k=DEFAULT_K, include_ndcg=False)
        metriche["embed_time_sec"] = round(tempo_embedding, 1)
        metriche["chunk_time_sec"] = round(tempo_chunking, 1)
        metriche["avg_retrieval_sec"] = round(tempo_retrieval_totale / len(dataset), 3)
        metriche["config"] = config

        # Stampa riepilogo
        print(f"\n  --- Risultati Config {cid} ---")
        print(f"  DocRecall@{DEFAULT_K}    (all): {metriche['all'].get(f'DocRecall@{DEFAULT_K}', 'N/A')}")
        print(f"  PassageRecall@{DEFAULT_K} (all): {metriche['all'].get(f'PassageRecall@{DEFAULT_K}', 'N/A')}")
        print(f"  DocPrecision@{DEFAULT_K} (all): {metriche['all'].get(f'DocPrecision@{DEFAULT_K}', 'N/A')}")
        print(f"  Tempo embedding: {metriche['embed_time_sec']}s")

        tutti_risultati.append(metriche)

        # Salva dopo ogni configurazione completata (checkpoint incrementale)
        with open(BLOCK1_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(tutti_risultati, f, indent=2, ensure_ascii=False)
        print(f"  [salvataggio] Risultati salvati in: {BLOCK1_OUTPUT}")

    print(f"\n{'='*60}")
    print(f"=== BLOCCO 1 COMPLETATO — {len(tutti_risultati)} configurazioni ===")
    print(f"{'='*60}")

    return tutti_risultati


def select_best_config(risultati: list[dict]) -> dict:
    """
    Seleziona la configurazione migliore del Blocco 1.
    Ottimizza in primis DocRecall, poi PassageRecall.
    """
    def score(r: dict):
        all_m = r.get("all", {})
        # Troviamo dinamicamente la chiave DocRecall@k e PassageRecall@k
        dr_keys = [k for k in all_m.keys() if k.startswith("DocRecall@")]
        dr = all_m.get(dr_keys[0]) if dr_keys else 0.0
        
        pr_keys = [k for k in all_m.keys() if k.startswith("PassageRecall@")]
        pr = all_m.get(pr_keys[0]) if pr_keys else 0.0
        
        return (dr, pr)

    migliore = max(risultati, key=score)
    print(f"\n[BEST CONFIG B1] Configurazione vincente: ID {migliore['config']['id']} - {migliore['config']['note']}")
    return migliore

