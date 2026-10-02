"""Calcolo delle metriche di retrieval per valutare la pipeline RAG."""

import math
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from sentence_transformers import CrossEncoder

from optimization_phase.configs.config import (
    EMBEDDING_DEVICE,
    EMBED_BATCH_SIZE,
    PASSAGE_EVAL_MODEL,
    PASSAGE_SIM_THRESHOLD,
    RESULTS_DIR,
    PREBUILD_DIR,
)

# Modello Reranker (Cross-Encoder) per la valutazione accurata della rilevanza passage

print(f"[metrics] Caricamento modello Reranker per valutazione: {PASSAGE_EVAL_MODEL}")
passage_eval_reranker = CrossEncoder(PASSAGE_EVAL_MODEL, device=EMBEDDING_DEVICE, max_length=512)


# Cache dei logit per le coppie (chunk_text, reference_passage) già calcolate

_cross_encoder_cache: dict[tuple[str, str], float] = {}
_CE_CACHE_PATH = os.path.join(PREBUILD_DIR, "cross_encoder_cache.pkl")


def _load_ce_cache() -> None:
    global _cross_encoder_cache
    if not _cross_encoder_cache and os.path.exists(_CE_CACHE_PATH):
        try:
            with open(_CE_CACHE_PATH, "rb") as f:
                _cross_encoder_cache = pickle.load(f)
            print(f"[metrics] Caricata cache Cross-Encoder da disco: {len(_cross_encoder_cache)} coppie.")
        except Exception as e:
            print(f"[metrics] Errore caricamento cache CE: {e}")
            _cross_encoder_cache = {}


def salva_ce_cache() -> None:
    global _cross_encoder_cache
    if _cross_encoder_cache:
        try:
            os.makedirs(os.path.dirname(_CE_CACHE_PATH), exist_ok=True)
            with open(_CE_CACHE_PATH, "wb") as f:
                pickle.dump(_cross_encoder_cache, f)
        except Exception as e:
            print(f"[metrics] Errore salvataggio cache CE: {e}")


def build_reference_passage_cache(dataset: list[dict]) -> None:
    """(Deprecato) Compatibilità con codice preesistente. Con il Cross-Encoder la cache è on-the-fly."""
    pass


# Utility Functions

def chunk_doc_relevant(node, relevant_documents: list[str]) -> bool:
    """Verifica se un chunk proviene da uno dei documenti rilevanti annotati."""
    file_name = node.metadata.get("file_name", "")
    return file_name in set(relevant_documents)


def chunk_passage_relevant(
    node,
    reference_passages: list[str],
    threshold: float = PASSAGE_SIM_THRESHOLD,
) -> list[str]:
    """Restituisce i reference_passage rilevanti rispetto al chunk tramite Cross-Encoder."""
    if not reference_passages:
        return []

    _load_ce_cache()

    chunk_text = node.text
    matched = []

    pairs_to_compute = []
    refs_to_compute = []

    for ref in reference_passages:
        cache_key = (chunk_text, ref)
        if cache_key in _cross_encoder_cache:
            score = _cross_encoder_cache[cache_key]
            if score >= threshold:
                matched.append(ref)
        else:
            pairs_to_compute.append([chunk_text, ref])
            refs_to_compute.append(ref)

    if pairs_to_compute:
        scores = passage_eval_reranker.predict(pairs_to_compute)
        
        try:
            iter(scores)
            if isinstance(scores, (str, bytes)): # just in case
                scores = [scores]
        except TypeError:
            scores = [scores]

        for ref, score in zip(refs_to_compute, scores):
            val = float(score)
            _cross_encoder_cache[(chunk_text, ref)] = val
            if val >= threshold:
                matched.append(ref)

    return matched


# Metriche di base

def recall_at_k(labels: list[int], total_relevant: int, k: int) -> float | None:
    """Calcola Recall@k: quanti elementi rilevanti sono tra i top-k."""
    if total_relevant == 0:
        return None
    return sum(labels[:k]) / total_relevant


def precision_at_k(labels: list[int], k: int) -> float:
    """Calcola Precision@k: quanti dei top-k sono rilevanti."""
    if k == 0:
        return 0.0
    return sum(labels[:k]) / k


def ndcg_at_k(labels: list[int], total_relevant: int, k: int) -> float:
    """Calcola NDCG@k (Normalized Discounted Cumulative Gain)."""
    labels_k = labels[:k]
    # DCG: somma di rel_i / log2(i+2), con i da 0
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(labels_k))
    # IDCG: ranking ideale (tutti i rilevanti nelle prime posizioni)
    n_ideale = min(total_relevant, k)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_ideale))
    return dcg / idcg if idcg > 0 else 0.0


# Valutazione Query Singola

def evaluate_query(
    retrieved_nodes: list,
    item: dict,
    k: int,
    include_ndcg: bool = False,
) -> dict:
    """Valuta i risultati del retriever per una singola domanda."""
    relevant_docs = item.get("relevant_documents", [])
    ref_passages = item.get("reference_passages", [])

    # --- Livello documento: 1 se il chunk proviene da un doc rilevante ---
    doc_labels = [
        1 if chunk_doc_relevant(n.node, relevant_docs) else 0
        for n in retrieved_nodes
    ]

    # --- Livello passage: lista dei passaggi matchati per ogni chunk ---
    pass_matches = [
        chunk_passage_relevant(n.node, ref_passages)
        for n in retrieved_nodes
    ]
    pass_labels = [1 if len(m) > 0 else 0 for m in pass_matches]

    # Conteggio totale elementi rilevanti (denominatore per recall)
    total_doc_rel = len(set(relevant_docs))
    total_pass_rel = len(set(ref_passages))

    # --- Calcolo DocRecall (Documenti Unici) ---
    retrieved_k = retrieved_nodes[:k]
    unique_docs_found = {
        n.node.metadata.get("file_name", "") for n in retrieved_k
        if n.node.metadata.get("file_name", "") in set(relevant_docs)
    }
    doc_recall = len(unique_docs_found) / total_doc_rel if total_doc_rel > 0 else None

    # --- Calcolo R-Precision ---
    if total_doc_rel > 0:
        r_prec_k = min(total_doc_rel, len(retrieved_nodes))
        r_hits = sum([1 for n in retrieved_nodes[:r_prec_k] if n.node.metadata.get("file_name", "") in set(relevant_docs)])
        r_precision = r_hits / total_doc_rel
    else:
        r_precision = 0.0

    # --- Calcolo PassageRecall (Passaggi Unici) ---
    unique_pass_found = set()
    for matches in pass_matches[:k]:
        unique_pass_found.update(matches)
    passage_recall = len(unique_pass_found) / total_pass_rel if total_pass_rel > 0 else None

    # Costruzione del risultato
    result = {
        "query_id": item.get("id", "?"),
        "type": item.get("type", "unknown"),
        "DocRecall": doc_recall,
        "PassageRecall": passage_recall,
        "R-Precision": r_precision,
    }

    if include_ndcg:
        # NDCG corretto: score assegnato solo alla prima occorrenza per evitare > 1
        
        doc_labels_ndcg = []
        visti_doc = set()
        for n in retrieved_nodes:
            fname = n.node.metadata.get("file_name", "")
            if fname in set(relevant_docs) and fname not in visti_doc:
                doc_labels_ndcg.append(1)
                visti_doc.add(fname)
            else:
                doc_labels_ndcg.append(0)

        pass_labels_ndcg = []
        visti_pass = set()
        for matches in pass_matches:
            nuovi = [m for m in matches if m not in visti_pass]
            if nuovi:
                pass_labels_ndcg.append(1)
                visti_pass.update(nuovi)
            else:
                pass_labels_ndcg.append(0)

        result["DocNDCG"] = ndcg_at_k(doc_labels_ndcg, total_doc_rel, k)
        result["PassageNDCG"] = ndcg_at_k(pass_labels_ndcg, total_pass_rel, k) if total_pass_rel > 0 else None

    return result


# Aggregazione Dataset

def aggregate_metrics(
    results: list[dict],
    k: int,
    include_ndcg: bool = False,
) -> dict:
    """Aggrega le metriche query per tipo e per l'intero dataset."""
    # Raggruppa i risultati per tipo di domanda
    per_tipo: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        per_tipo[r["type"]].append(r)
    per_tipo["all"] = results  # aggiunge anche il totale

    # Lista delle metriche da aggregare
    metric_names = ["DocRecall", "PassageRecall", "R-Precision"]
    if include_ndcg:
        metric_names += ["DocNDCG", "PassageNDCG"]

    aggregated = {}
    for tipo, items in per_tipo.items():
        aggregated[tipo] = {}
        for m in metric_names:
            # Esclude i None (metriche non applicabili, es. unanswerable)
            valori = [x[m] for x in items if m in x and x.get(m) is not None]
            chiave = f"{m}@{k}"
            aggregated[tipo][chiave] = round(float(np.mean(valori)), 4) if valori else None

    # Salva la cache delle coppie valutate su disco al termine di ogni configurazione
    salva_ce_cache()

    return aggregated
