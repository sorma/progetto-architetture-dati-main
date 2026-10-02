"""Esegue il loop di generazione per una singola configurazione sperimentale."""

import json
import os
import time
from openai import OpenAI

from generative_phase.configs.config import PROMPTS_DIR, CURRENT_DOMAIN, LOCAL_LLM_URL


def get_client_local() -> OpenAI:
    """Restituisce un client OpenAI per connettersi all'endpoint locale Ollama."""
    return OpenAI(base_url=LOCAL_LLM_URL, api_key="ollama")


def carica_prompt(strategia: str) -> dict:
    """Carica il prompt per la strategia indicata dal dominio corrente."""
    percorso = os.path.join(PROMPTS_DIR, "prompts.json")
    with open(percorso, encoding="utf-8") as f:
        return json.load(f)[strategia]


def recupera_contesto(retriever, domanda: str) -> str:
    """
    Usa il retriever per trovare i chunk più rilevanti per la domanda, unendoli con un separatore visibile ("---").
    """
    nodi = retriever.retrieve(domanda)
    return "\n\n---\n\n".join(nodo.get_content() for nodo in nodi)


def costruisci_messaggi(prompt_config: dict, domanda: str, contesto: str) -> list:
    """Costruisce la lista di messaggi nel formato richiesto dall'API Ollama/OpenAI."""
    # Riempie il template usando string.replace per evitare conflitti JSON
    contenuto_utente = prompt_config["user_template"].replace(
        "{context}", contesto
    ).replace(
        "{question}", domanda
    )
    return [
        {"role": "system", "content": prompt_config["system"]},
        {"role": "user",   "content": contenuto_utente}
    ]


def ottieni_o_crea_cache_retrieval(domande: list, retriever) -> dict:
    """Gestisce la cache del retrieval per evitare di recuperare il contesto ripetutamente per ogni config."""
    from generative_phase.configs.config import BEST_PARAMS_PATH, GEN_RESULTS_DIR
    
    percorso_cache = os.path.join(GEN_RESULTS_DIR, "retrieval_cache.json")
    
    # Carica i best_params correnti
    if not os.path.exists(BEST_PARAMS_PATH):
        raise FileNotFoundError(f"File best_params.json non trovato: {BEST_PARAMS_PATH}")
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        best_params_correnti = json.load(f)
        
    use_cache = False
    cache_data = {}
    
    if os.path.exists(percorso_cache):
        try:
            with open(percorso_cache, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            # Verifica se i parametri sono cambiati
            if cache_data.get("best_params") == best_params_correnti:
                use_cache = True
            else:
                print("  [retrieval-cache] Parametri di ottimizzazione cambiati. Ricalcolo della cache...")
        except Exception as e:
            print(f"  [retrieval-cache] Errore nel caricamento della cache: {e}. Ricalcolo...")
            
    if use_cache:
        print(f"  [retrieval-cache] Caricata cache valida da: {percorso_cache} ({len(cache_data['items'])} domande)")
        return cache_data
        
    print(f"\n  [retrieval-cache] Cache non valida o assente. Avvio pre-retrieval di {len(domande)} domande...")
    
    cache_items = {}
    t_start = time.time()
    
    for i, d in enumerate(domande):
        t0 = time.time()
        contesto = recupera_contesto(retriever, d["question"])
        latenza = time.time() - t0
        
        cache_items[str(d["id"])] = {
            "context": contesto,
            "retrieval_latency_sec": round(latenza, 3)
        }
        
        if (i + 1) % 10 == 0:
            print(f"    [retrieval] Domanda {i+1}/{len(domande)} recuperata (latenza media: {(time.time() - t_start)/(i+1):.2f}s)")
            
    cache_data = {
        "best_params": best_params_correnti,
        "items": cache_items
    }
    
    os.makedirs(os.path.dirname(percorso_cache), exist_ok=True)
    with open(percorso_cache, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
    print(f"  [retrieval-cache] Pre-retrieval completato in {time.time() - t_start:.1f}s. Cache salvata in: {percorso_cache}")
    return cache_data


def esegui_configurazione(config: dict, domande: list, retriever, retrieval_cache: dict = None) -> list:
    """Esegue una configurazione sperimentale su tutte le domande del dataset."""
    risultati = []

    # Carica il prompt iniziale
    prompt_config = carica_prompt(config["prompt_strategy"])

    print(f"  Strategia '{config['prompt_strategy']}' — prompt caricato da: "
          "prompts.json")

    # WARM-UP: Caricamento pre-test del modello
    print(f"  [Warm-up] Precaricamento del modello {config['model']} in memoria...")
    try:
        get_client_local().chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=1
        )
    except Exception as e:
        print(f"  [Warm-up] Errore: {e}")

    # Inizializza il client API
    client = get_client_local()

    for i, domanda in enumerate(domande):
        # Recupera contesto (via cache o dal vivo)
        qid_str = str(domanda["id"])
        if retrieval_cache is not None and "items" in retrieval_cache and qid_str in retrieval_cache["items"]:
            contesto = retrieval_cache["items"][qid_str]["context"]
            retrieval_latency = retrieval_cache["items"][qid_str].get("retrieval_latency_sec", 0.0)
        else:
            t_ret_start = time.time()
            contesto = recupera_contesto(retriever, domanda["question"])
            retrieval_latency = time.time() - t_ret_start

        # Costruisce i messaggi per l'API
        messaggi = costruisci_messaggi(prompt_config, domanda["question"], contesto)

        # Chiama l'LLM locale
        inizio = time.time()
        risposta = client.chat.completions.create(
            model=config["model"],
            messages=messaggi,
            temperature=config["temperature"],
            max_tokens=2048,
            extra_body={"options": {"num_ctx": 16384}}
        )
        
        latenza = time.time() - inizio

        # Raccoglie tutti i dati del risultato in un dizionario
        risultati.append({
            "question_id":           domanda["id"],
            "question":              domanda["question"],     # testo completo, necessario per il judge
            "question_type":         domanda["type"],
            "domain":                CURRENT_DOMAIN,
            "config_id":             config["config_id"],
            "model":                 config["model"],
            "prompt_strategy":       config["prompt_strategy"],
            "generated_answer":      risposta.choices[0].message.content,
            "reference_answer":      domanda["answer"],
            "reference_passages":    domanda.get("reference_passages", []),
            "latency_sec":           round(latenza, 3),
            "retrieval_latency_sec": round(retrieval_latency, 3),
            "input_tokens":          risposta.usage.prompt_tokens,
            "output_tokens":         risposta.usage.completion_tokens,
            "total_tokens":          risposta.usage.total_tokens
        })

        # Log progresso
        if (i + 1) % 10 == 0:
            print(f"  Domanda {i + 1}/{len(domande)} elaborata")

    return risultati
