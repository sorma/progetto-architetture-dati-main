"""BLOCCO 3: Ottimizzazione di Retrieval, k e Reranking."""

import json
import os
import time

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from optimization_phase.configs.config import (
    RESULTS_DIR,
    RAW_RESULTS_DIR,
    BLOCK3_CONFIGS,
)
from optimization_phase.analysis.metrics import evaluate_query, aggregate_metrics
from optimization_phase.blocks.block2_query_transformation import apply_qt_and_retrieve, get_groq_llm, token_counter

# Percorso del file di output del blocco 3
BLOCK3_OUTPUT = os.path.join(RAW_RESULTS_DIR, "block3_results.json")


def get_reranker(k_final: int):
    """Inizializza il reranker locale Cross-Encoder."""
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    print(f"  [reranker] Inizializzazione cross-encoder locale (BAAI/bge-reranker-v2-m3)...")
    return FlagEmbeddingReranker(
        model="BAAI/bge-reranker-v2-m3",
        top_n=k_final,
        use_fp16=False # Mantiene compatibilità con CPU/GPU generiche
    )


def get_retriever(config: dict, index, nodi_per_bm25: list):
    """Costruisce il retriever appropriato (vector, bm25, hybrid)."""
    tipo = config["retrieval"]
    k = config["k_retrieve"]

    if tipo == "vector":
        # Retrieval vettoriale puro: cosine similarity
        return index.as_retriever(
            similarity_top_k=k,
            vector_store_query_mode="default",
        )

    elif tipo == "bm25":
        # Ricerca lessicale pura (BM25 / Keyword Search)
        return BM25Retriever.from_defaults(
            nodes=nodi_per_bm25,
            similarity_top_k=k,
        )

    elif tipo == "hybrid":
        # Hybrid: combina BM25 (lexicale) + vector (semantico) con RRF
        retriever_vettoriale = index.as_retriever(similarity_top_k=k)
        retriever_bm25 = BM25Retriever.from_defaults(
            nodes=nodi_per_bm25,
            similarity_top_k=k,
        )
        return QueryFusionRetriever(
            retrievers=[retriever_vettoriale, retriever_bm25],
            similarity_top_k=k,
            num_queries=1,        # non genera nuove query, solo fonde i risultati
            mode="reciprocal_rerank",
            use_async=False,
            llm=get_groq_llm(),   # usa l'LLM di Groq per evitare il fallback su OpenAI
        )

    else:
        raise ValueError(f"Tipo di retrieval sconosciuto: {tipo}")


def run_block3(
    dataset: list[dict],
    best_b1_index,
    best_b2_qt: str,
    nodi_per_bm25: list,
) -> list[dict]:
    """Esegue il Blocco 3 completo: retrieval, reranking e metriche per tutte le config."""
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # Carica risultati già esistenti (esecuzione incrementale)
    risultati_esistenti: list[dict] = []
    if os.path.exists(BLOCK3_OUTPUT):
        with open(BLOCK3_OUTPUT, "r", encoding="utf-8") as f:
            risultati_esistenti = json.load(f)
    ids_completati = {r["config"]["id"] for r in risultati_esistenti}

    tutti_risultati = list(risultati_esistenti)

    for config in BLOCK3_CONFIGS:
        cid = config["id"]

        # Salta le configurazioni già eseguite
        if cid in ids_completati:
            print(f"\n=== Config {cid} già eseguita — skip ===")
            continue

        print(f"\n{'='*60}")
        print(f"=== BLOCCO 3 — Config {cid}: {config['note']} ===")
        print(f"  Retrieval: {config['retrieval']}, k_retrieve={config['k_retrieve']}, "
              f"rerank={config['rerank']}, k_final={config['k_final']}")
        print(f"  QT usata: {best_b2_qt}")
        print(f"{'='*60}")

        # Costruisce il retriever appropriato
        retriever = get_retriever(config, best_b1_index, nodi_per_bm25)

        # Inizializza il reranker locale Cross-Encoder fuori dal loop per efficienza (se attivo)
        reranker = None
        if config["rerank"]:
            reranker = get_reranker(k_final=config["k_final"])

        valutazioni = []
        latenze = []
        k_finale = config["k_final"]
        token_counter.reset_counts()

        for i, item in enumerate(dataset):
            # Gestione rate limit: delay di 0.0 secondi se usiamo le API di Groq per la Query Transformation
            richiede_api = (best_b2_qt != "none")
            if richiede_api and i > 0:
                time.sleep(0.0)

            # Rotazione delle chiavi API ad ogni chiamata LLM per la Query Transformation
            llm_qt = get_groq_llm() if best_b2_qt != "none" else None

            # Calcola latenza netta (QT + Retrieval + Reranking), escludendo i sleep
            t0 = time.time()

            # Applica QT e retrieval
            # Recupera k_retrieve chunk (poi il reranker li riduce a k_final)
            nodi_recuperati, _ = apply_qt_and_retrieve(
                question=item["question"],
                qt_type=best_b2_qt,
                retriever=retriever,
                llm=llm_qt,
                k=config["k_retrieve"],
            )

            # Applica il reranking se richiesto
            if reranker is not None:
                try:
                    # Il reranker riordina i chunk e ne mantiene solo k_final
                    nodi_recuperati = reranker.postprocess_nodes(
                        nodi_recuperati,
                        query_str=item["question"],
                    )
                except Exception as e:
                    print(f"  [reranker] Errore su domanda {item['id']}: {e}")
                    # Fallback: prendi semplicemente i primi k_final
                    nodi_recuperati = nodi_recuperati[:k_finale]
            else:
                # Senza reranker: tronca a k_final
                nodi_recuperati = nodi_recuperati[:k_finale]
            
            latenza_b3 = time.time() - t0
            latenze.append(latenza_b3)

            # Calcola le metriche (NDCG incluso per il Blocco 3)
            eval_result = evaluate_query(
                retrieved_nodes=nodi_recuperati,
                item=item,
                k=k_finale,
                include_ndcg=True,  # NDCG è la metrica chiave in questo blocco
            )
            valutazioni.append(eval_result)

            if (i + 1) % 10 == 0:
                print(f"  Domanda {i+1}/{len(dataset)} elaborata")

        # Aggregazione delle metriche (include NDCG)
        metriche = aggregate_metrics(valutazioni, k=k_finale, include_ndcg=True)
        metriche["config"] = config
        metriche["avg_latency_sec"] = round(sum(latenze) / len(latenze), 3) if latenze else 0.0
        
        prompt_tokens = token_counter.prompt_llm_token_count
        completion_tokens = token_counter.completion_llm_token_count
        tot_tokens = prompt_tokens + completion_tokens

        metriche["avg_input_tokens"] = round(prompt_tokens / len(dataset), 1) if dataset else 0.0
        metriche["avg_output_tokens"] = round(completion_tokens / len(dataset), 1) if dataset else 0.0
        metriche["avg_tokens_per_query"] = round(tot_tokens / len(dataset), 1) if dataset else 0.0

        # Stampa riepilogo
        print(f"\n  --- Risultati Config {cid} (k_final={k_finale}) ---")
        print(f"  DocRecall@{k_finale}    (all): {metriche['all'].get(f'DocRecall@{k_finale}', 'N/A')}")
        print(f"  PassageRecall@{k_finale} (all): {metriche['all'].get(f'PassageRecall@{k_finale}', 'N/A')}")
        print(f"  DocPrecision@{k_finale} (all): {metriche['all'].get(f'DocPrecision@{k_finale}', 'N/A')}")
        print(f"  DocNDCG@{k_finale}      (all): {metriche['all'].get(f'DocNDCG@{k_finale}', 'N/A')}")
        print(f"  PassageNDCG@{k_finale}  (all): {metriche['all'].get(f'PassageNDCG@{k_finale}', 'N/A')}")
        print(f"  Latenza media: {metriche['avg_latency_sec']}s/query")
        print(f"  Token medi: Input {metriche.get('avg_input_tokens', 0)} | Output {metriche.get('avg_output_tokens', 0)} (Totale: {metriche['avg_tokens_per_query']})")

        tutti_risultati.append(metriche)

        # Checkpoint incrementale
        with open(BLOCK3_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(tutti_risultati, f, indent=2, ensure_ascii=False)
        print(f"  [salvataggio] Risultati salvati in: {BLOCK3_OUTPUT}")

    print(f"\n{'='*60}")
    print(f"=== BLOCCO 3 COMPLETATO — {len(tutti_risultati)} configurazioni ===")
    print(f"{'='*60}")

    return tutti_risultati


def select_best_config_b3(risultati: list[dict]) -> dict:
    """Seleziona la configurazione migliore del Blocco 3 basandosi primariamente su Recall e NDCG."""
    def score(r: dict):
        k = r["config"]["k_final"]
        all_m = r.get("all", {})
        dr = all_m.get(f"DocRecall@{k}") or 0.0
        pr = all_m.get(f"PassageRecall@{k}") or 0.0
        dn = all_m.get(f"DocNDCG@{k}") or 0.0
        pn = all_m.get(f"PassageNDCG@{k}") or 0.0
        dp = all_m.get(f"DocPrecision@{k}") or 0.0
        return (dr, pr, dn, pn, dp)

    migliore = max(risultati, key=score)
    k = migliore["config"]["k_final"]

    print(f"\n[BEST CONFIG B3] Config {migliore['config']['id']}: {migliore['config']['note']}")
    print(f"  DocRecall@{k}:    {migliore['all'].get(f'DocRecall@{k}')}")
    print(f"  PassageRecall@{k}: {migliore['all'].get(f'PassageRecall@{k}')}")
    print(f"  DocNDCG@{k}:      {migliore['all'].get(f'DocNDCG@{k}')}")
    print(f"  DocPrecision@{k}: {migliore['all'].get(f'DocPrecision@{k}')}")

    return migliore
