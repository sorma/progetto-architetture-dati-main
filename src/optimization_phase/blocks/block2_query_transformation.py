"""BLOCCO 2: Ottimizzazione della Query Transformation (nessuna, hyde, multi_query, rag_fusion)."""

import json
import os
import time

from llama_index.llms.groq import Groq
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken

from optimization_phase.configs.config import (
    RESULTS_DIR,
    RAW_RESULTS_DIR,
    GROQ_API_KEYS,
    GROQ_QT_MODEL,
    DEFAULT_K,
    BLOCK2_QT_TYPES,
)
from optimization_phase.analysis.metrics import evaluate_query, aggregate_metrics

# Percorso del file di output del blocco 2
BLOCK2_OUTPUT = os.path.join(RAW_RESULTS_DIR, "block2_results.json")

# Ritardo minimo tra chiamate API Groq (in secondi).
# Aumenta questo valore se ricevi errori di rate limit.
GROQ_RETRY_DELAY = 0.0

# Numero massimo di tentativi per ogni chiamata API Groq in caso di errore temporaneo.
GROQ_MAX_RETRIES = 3

# Inizializza il contatore dei token per LlamaIndex
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
Settings.callback_manager = CallbackManager([token_counter])

_key_counter = 0

def get_groq_llm(model_name: str = GROQ_QT_MODEL) -> Groq:
    """Inizializza il modello LLM Groq ruotando tra le chiavi API configurate."""
    global _key_counter
    if not GROQ_API_KEYS:
        raise ValueError("Nessuna chiave API Groq configurata in config.py o .env.")
    
    # Seleziona la chiave in base al contatore rotante
    key = GROQ_API_KEYS[_key_counter % len(GROQ_API_KEYS)]
    _key_counter += 1
    
    # Stampiamo a console quale chiave stiamo usando per trasparenza
    key_masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "..."
    print(f"  [Groq LLM] Utilizzo chiave: {key_masked}")
    
    return Groq(
        model=model_name,
        api_key=key,
        temperature=0.1,  # bassa temperatura per output più deterministico
    )


def apply_qt_and_retrieve(
    question: str,
    qt_type: str,
    retriever,
    llm,
    k: int = DEFAULT_K,
) -> tuple[list, float]:
    """Applica la strategia di query transformation specificata ed esegue il retrieval."""
    t0 = time.time()

    # --- Strategia 1: nessuna trasformazione ---
    if qt_type == "none":
        nodi = retriever.retrieve(question)
        latenza = time.time() - t0
        return nodi[:k], latenza

    # --- Strategia 2: HyDE (Hypothetical Document Embedding) ---
    elif qt_type == "hyde":
        try:
            from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
            hyde = HyDEQueryTransform(llm=llm, include_original=True)
            # hyde_bundle va passato direttamente al retriever per non ricalcolare l'embedding
            hyde_bundle = hyde(question)
            nodi = retriever.retrieve(hyde_bundle)
        except Exception as e:
            print(f"  [HyDE] Errore: {e}. Fallback su query originale.")
            nodi = retriever.retrieve(question)
        latenza = time.time() - t0
        return nodi[:k], latenza

    # --- Strategia 3: MultiQuery (3 riformulazioni + RRF) ---
    elif qt_type == "multi_query":
        try:
            fusion_retriever = QueryFusionRetriever(
                retrievers=[retriever],
                llm=llm,
                num_queries=4,   # 1 originale + 3 riformulazioni
                mode="reciprocal_rerank",
                use_async=False,
                similarity_top_k=k,
            )
            nodi = fusion_retriever.retrieve(question)
        except Exception as e:
            print(f"  [MultiQuery] Errore: {e}. Fallback su query originale.")
            nodi = retriever.retrieve(question)
        latenza = time.time() - t0
        return nodi[:k], latenza

    # --- Strategia 4: RAG Fusion (5 riformulazioni + RRF esplicita) ---
    elif qt_type == "rag_fusion":
        try:
            fusion_retriever = QueryFusionRetriever(
                retrievers=[retriever],
                llm=llm,
                num_queries=5,   # 1 originale + 4 riformulazioni
                mode="reciprocal_rerank",
                use_async=False,
                similarity_top_k=k,
            )
            nodi = fusion_retriever.retrieve(question)
        except Exception as e:
            print(f"  [RAGFusion] Errore: {e}. Fallback su query originale.")
            nodi = retriever.retrieve(question)
        latenza = time.time() - t0
        return nodi[:k], latenza

    else:
        raise ValueError(f"Tipo di query transformation sconosciuto: {qt_type}")


def run_block2(dataset: list[dict], best_b1_index) -> list[dict]:
    """Esegue il Blocco 2 completo: testa le 4 strategie di query transformation."""
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # Retriever base (vector similarity, k=5)
    retriever = best_b1_index.as_retriever(similarity_top_k=DEFAULT_K)

    # Carica risultati già esistenti
    risultati_esistenti: list[dict] = []
    if os.path.exists(BLOCK2_OUTPUT):
        with open(BLOCK2_OUTPUT, "r", encoding="utf-8") as f:
            risultati_esistenti = json.load(f)
    qt_completate = {r["qt_type"] for r in risultati_esistenti}

    tutti_risultati = list(risultati_esistenti)

    for qt_type in BLOCK2_QT_TYPES:

        # Salta le strategie già eseguite
        if qt_type in qt_completate:
            print(f"\n=== QT '{qt_type}' già eseguita — skip ===")
            continue

        print(f"\n{'='*60}")
        print(f"=== BLOCCO 2 — Strategia: {qt_type.upper()} ===")
        print(f"{'='*60}")

        valutazioni = []
        latenze = []
        errori = 0
        token_counter.reset_counts()  # Azzera contatore token per questa strategia

        for i, item in enumerate(dataset):
            # Gestione rate limit: attesa tra le chiamate API
            if qt_type != "none" and i > 0:
                time.sleep(GROQ_RETRY_DELAY)

            # Rotazione delle chiavi API ad ogni chiamata LLM per evitare rate limit
            llm = get_groq_llm(GROQ_QT_MODEL) if qt_type != "none" else None

            # Tenta il retrieval con retry in caso di errore API
            nodi_recuperati = None
            for tentativo in range(GROQ_MAX_RETRIES):
                try:
                    nodi_recuperati, latenza = apply_qt_and_retrieve(
                        question=item["question"],
                        qt_type=qt_type,
                        retriever=retriever,
                        llm=llm,
                        k=DEFAULT_K,
                    )
                    break
                except Exception as e:
                    print(f"  [retry {tentativo+1}/{GROQ_MAX_RETRIES}] Errore: {e}")
                    time.sleep(2 ** tentativo * 2)  # backoff esponenziale

            if nodi_recuperati is None:
                # Fallback: retrieval senza trasformazione
                print(f"  [fallback] Domanda {item['id']}: uso query originale.")
                nodi_recuperati = retriever.retrieve(item["question"])[:DEFAULT_K]
                latenza = 0.0
                errori += 1

            latenze.append(latenza)

            # Calcola le metriche per questa domanda
            eval_result = evaluate_query(
                retrieved_nodes=nodi_recuperati,
                item=item,
                k=DEFAULT_K,
                include_ndcg=False,
            )
            valutazioni.append(eval_result)

            # Stampa progresso ogni 10 domande
            if (i + 1) % 10 == 0:
                print(f"  Domanda {i+1}/{len(dataset)} — latenza media: {sum(latenze)/len(latenze):.2f}s")

        # Aggregazione delle metriche
        metriche = aggregate_metrics(valutazioni, k=DEFAULT_K, include_ndcg=False)
        metriche["qt_type"] = qt_type
        metriche["avg_latency_sec"] = round(sum(latenze) / len(latenze), 3) if latenze else 0.0
        # Calcolo dei token medi consumati per query
        prompt_tokens = token_counter.prompt_llm_token_count
        completion_tokens = token_counter.completion_llm_token_count
        tot_tokens = prompt_tokens + completion_tokens

        metriche["avg_input_tokens"] = round(prompt_tokens / len(dataset), 1) if dataset else 0.0
        metriche["avg_output_tokens"] = round(completion_tokens / len(dataset), 1) if dataset else 0.0
        metriche["avg_tokens_per_query"] = round(tot_tokens / len(dataset), 1) if dataset else 0.0
        metriche["errori_api"] = errori

        # Stampa riepilogo
        print(f"\n  --- Risultati QT '{qt_type}' ---")
        print(f"  DocRecall@{DEFAULT_K}    (all): {metriche['all'].get(f'DocRecall@{DEFAULT_K}', 'N/A')}")
        print(f"  PassageRecall@{DEFAULT_K} (all): {metriche['all'].get(f'PassageRecall@{DEFAULT_K}', 'N/A')}")
        print(f"  DocPrecision@{DEFAULT_K} (all): {metriche['all'].get(f'DocPrecision@{DEFAULT_K}', 'N/A')}")
        print(f"  Latenza media: {metriche['avg_latency_sec']}s/query")
        print(f"  Token medi: Input {metriche.get('avg_input_tokens', 0)} | Output {metriche.get('avg_output_tokens', 0)} (Totale: {metriche['avg_tokens_per_query']})")
        if errori > 0:
            print(f"  ⚠ Errori API: {errori} domande gestite con fallback")

        tutti_risultati.append(metriche)

        # Checkpoint incrementale
        with open(BLOCK2_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(tutti_risultati, f, indent=2, ensure_ascii=False)
        print(f"  [salvataggio] Risultati salvati in: {BLOCK2_OUTPUT}")

    print(f"\n{'='*60}")
    print(f"=== BLOCCO 2 COMPLETATO — {len(tutti_risultati)} strategie ===")
    print(f"{'='*60}")

    return tutti_risultati


def select_best_qt(risultati: list[dict]) -> str:
    """Seleziona la migliore strategia di query transformation del Blocco 2."""
    from optimization_phase.configs.config import DEFAULT_K
    
    def score(r: dict):
        all_m = r.get("all", {})
        dr = all_m.get(f"DocRecall@{DEFAULT_K}", 0.0)
        pr = all_m.get(f"PassageRecall@{DEFAULT_K}", 0.0)
        err = r.get("errori_api", 0)
        return (dr, pr, -err)

    migliore = max(risultati, key=score)
    qt_type = migliore["qt_type"]
    
    print(f"\n[BEST QT] Strategia vincente: '{qt_type}'")
    k = DEFAULT_K
    print(f"  DocRecall@{k}:    {migliore['all'].get(f'DocRecall@{k}')}")
    print(f"  PassageRecall@{k}: {migliore['all'].get(f'PassageRecall@{k}')}")
    
    return qt_type

