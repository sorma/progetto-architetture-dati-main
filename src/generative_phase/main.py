"""Orchestratore della pipeline della fase generativa RAG."""

import argparse
import json
import os
import sys
import time
import pandas as pd

from generative_phase.configs.config import (
    DATASET_PATH,
    CURRENT_DOMAIN,
    DOCS_DIR,
    GEN_RESULTS_DIR,
    EXPERIMENT_CONFIGS
)
from generative_phase.runner.retriever import costruisci_retriever
from generative_phase.runner.experiment_runner import esegui_configurazione
from generative_phase.analysis.judge import valuta_risposta
from generative_phase.analysis.metrics import aggiungi_campi_derivati, calcola_metriche_aggregate




def carica_dataset() -> list:
    """Carica il dataset delle domande per il dominio."""
    if not os.path.exists(DATASET_PATH):
        print(f"[ERRORE] Dataset non trovato: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        domande = json.load(f)

    conteggi_tipo = {}
    for d in domande:
        tipo = d.get("type", "unknown")
        conteggi_tipo[tipo] = conteggi_tipo.get(tipo, 0) + 1

    print(f"[dataset] Caricate {len(domande)} domande da '{DATASET_PATH}'")
    for tipo, n in sorted(conteggi_tipo.items()):
        print(f"  - {tipo}: {n}")

    return domande


def mostra_configurazione() -> None:
    """Stampa la configurazione corrente."""
    from generative_phase.configs.config import BEST_PARAMS_PATH, JUDGE_MODEL, EMBEDDING_DEVICE
    print("\n" + "=" * 60)
    print(f"  FASE GENERATIVA RAG — DOMINIO: {CURRENT_DOMAIN.upper()}")
    print("=" * 60)
    print(f"  Dataset:          {DATASET_PATH}")
    print(f"  Documenti PDF:    {DOCS_DIR}")
    print(f"  Best params RAG:  {BEST_PARAMS_PATH}")
    print(f"  Output:           {GEN_RESULTS_DIR}")
    print(f"  Modello judge:    {JUDGE_MODEL}")
    print(f"  Embedding device: {EMBEDDING_DEVICE}")
    print("=" * 60 + "\n")


def esegui_pipeline_config(config: dict, domande: list, retriever, retrieval_cache: dict, ricalcola_judge: bool = False) -> list:
    """Esegue generazione + valutazione judge per una singola configurazione, saltandola se già calcolata."""
    cartella_raw = os.path.join(GEN_RESULTS_DIR, "raw")
    os.makedirs(cartella_raw, exist_ok=True)
    
    percorso_gen = os.path.join(cartella_raw, f"config_{config['config_id']}_generated.json")
    percorso_eval = os.path.join(cartella_raw, f"config_{config['config_id']}_results.json")

    if not ricalcola_judge and os.path.exists(percorso_eval):
        try:
            with open(percorso_eval, "r", encoding="utf-8") as f:
                dati = json.load(f)
                if len(dati) == len(domande) and all("score" in r for r in dati):
                    print(f"  Config {config['config_id']} già completamente valutata — skip.")
                    return dati
        except Exception as e:
            print(f"  [Errore] Caricamento {percorso_eval} fallito: {e}. Verrà ricalcolato.")

    import sys
    solo_gen = "--solo-generazione" in sys.argv
    if solo_gen and os.path.exists(percorso_gen):
        try:
            with open(percorso_gen, "r", encoding="utf-8") as f:
                dati = json.load(f)
                if len(dati) == len(domande):
                    print(f"  Config {config['config_id']} (solo generazione) già completata — skip.")
                    return dati
        except Exception as e:
            print(f"  [Errore] Caricamento {percorso_gen} fallito: {e}. Verrà rigenerato.")

    print(f"\n{'=' * 60}")
    print(f"  Config {config['config_id']}: {config['model']} | {config['prompt_strategy']}")
    print(f"{'=' * 60}")

    risultati = []
    if os.path.exists(percorso_gen):
        try:
            with open(percorso_gen, "r", encoding="utf-8") as f:
                risultati = json.load(f)
        except Exception as e:
            print(f"  [Errore] Caricamento {percorso_gen} fallito: {e}. Verrà rigenerato.")
            risultati = []
    elif os.path.exists(percorso_eval):
        try:
            with open(percorso_eval, "r", encoding="utf-8") as f:
                risultati = json.load(f)
        except Exception as e:
            print(f"  [Errore] Caricamento {percorso_eval} fallito: {e}. Verrà rigenerato.")
            risultati = []

    if len(risultati) == len(domande):
        print(f"  Trovato file generato completo {percorso_gen}, salto generazione...")
    else:
        if len(risultati) > 0:
            print(f"  Rilevata generazione parziale ({len(risultati)}/{len(domande)} domande). Ripresa generazione...")
        
        domande_rimanenti = domande[len(risultati):]
        risultati_nuovi = esegui_configurazione(config, domande_rimanenti, retriever, retrieval_cache)
        risultati.extend(risultati_nuovi)
        
        with open(percorso_gen, "w", encoding="utf-8") as f:
            json.dump(risultati, f, indent=2, ensure_ascii=False)
        print(f"  Salvato file generazione: {percorso_gen}")
        
    if solo_gen:
        print(f"  [Skip Judge] Modalità --solo-generazione attiva per config {config['config_id']}")
        return risultati

    print(f"\n  [judge] Valutazione {len(risultati)} risposte con {config['config_id']}...")

    from generative_phase.configs.config import JUDGE_MODEL, LOCAL_LLM_URL
    print(f"  [Warm-up] Precaricamento del modello judge ({JUDGE_MODEL}) in memoria...")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LOCAL_LLM_URL, api_key="ollama")
        client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=1
        )
    except Exception as e:
        print(f"  [Warm-up] Errore: {e}")

    risultati_valutati = []
    if os.path.exists(percorso_eval):
        try:
            with open(percorso_eval, "r", encoding="utf-8") as f:
                risultati_valutati = json.load(f)
        except Exception:
            risultati_valutati = []

    giudizi_esistenti = {}
    if not ricalcola_judge:
        for r in risultati_valutati:
            if "score" in r:
                giudizi_esistenti[r["question_id"]] = {
                    "score": r["score"],
                    "judge_justification": r.get("judge_justification", ""),
                    "judge_input_tokens": r.get("judge_input_tokens", 0),
                    "judge_output_tokens": r.get("judge_output_tokens", 0)
                }

    for i, r in enumerate(risultati):
        qid = r["question_id"]
        if qid in giudizi_esistenti:
            g = giudizi_esistenti[qid]
            r["score"]               = g["score"]
            r["judge_justification"] = g["judge_justification"]
            r["judge_input_tokens"]  = g["judge_input_tokens"]
            r["judge_output_tokens"] = g["judge_output_tokens"]
        else:
            giudizio = valuta_risposta(
                domanda=r["question"],
                risposta_reference=r["reference_answer"],
                risposta_generata=r["generated_answer"]
            )

            r["score"]               = giudizio["score"]
            r["judge_justification"] = giudizio["justification"]
            r["judge_input_tokens"]  = giudizio["judge_input_tokens"]
            r["judge_output_tokens"] = giudizio["judge_output_tokens"]
            
            with open(percorso_eval, "w", encoding="utf-8") as f:
                json.dump(risultati[:i+1], f, indent=2, ensure_ascii=False)

        if (i + 1) % 10 == 0:
            print(f"  [judge] {i + 1}/{len(risultati)} valutazioni completate")

    with open(percorso_eval, "w", encoding="utf-8") as f:
        json.dump(risultati, f, indent=2, ensure_ascii=False)

    print(f"  Config {config['config_id']} completata: {len(risultati)} domande valutate.")
    print(f"  Salvato file valutato: {percorso_eval}")

    return risultati


def salva_metriche(tutti_risultati: list) -> None:
    """Calcola e salva le metriche aggregate in JSON."""
    df = pd.DataFrame(tutti_risultati)
    df = aggiungi_campi_derivati(df)
    df_metriche = calcola_metriche_aggregate(df)

    # Salva il file JSON aggregato nella cartella risultati
    percorso_metrics = os.path.join(GEN_RESULTS_DIR, "aggregated_metrics.json")

    # Salviamo solo le metriche aggregate in JSON
    # (i risultati completi sono già nei JSON grezzi)
    df_metriche.to_json(percorso_metrics, orient="records", indent=2, force_ascii=False)

    print(f"\n  Metriche aggregate salvate in: {percorso_metrics}")




def main():
    parser = argparse.ArgumentParser(
        description=f"Pipeline Fase Generativa RAG — Dominio: {CURRENT_DOMAIN.upper()}"
    )
    parser.add_argument(
        "--solo-config", type=str,
        help="Esegui solo la configurazione specificata (es. C1, C3)."
    )
    parser.add_argument(
        "--solo-metriche", action="store_true",
        help="Ricalcola le metriche aggregate dai raw JSON già esistenti (senza rieseguire)."
    )
    parser.add_argument(
        "--solo-grafici", action="store_true",
        help="Genera solo i grafici dal CSV già esistente (senza rieseguire)."
    )
    parser.add_argument(
        "--solo-generazione", action="store_true",
        help="Esegui SOLO la generazione locale, fermandoti prima della valutazione Groq."
    )
    parser.add_argument(
        "--ricalcola-judge", action="store_true",
        help="Riesegue la valutazione (judge) per tutte le configurazioni, ignorando i punteggi precedenti."
    )
    args = parser.parse_args()

    mostra_configurazione()

    if args.solo_grafici:
        from generative_phase.analysis.plots import genera_tutti_i_grafici
        genera_tutti_i_grafici(GEN_RESULTS_DIR)
        return

    domande = carica_dataset()

    if args.solo_metriche:
        cartella_raw = os.path.join(GEN_RESULTS_DIR, "raw")
        tutti_risultati = []
        for nome_file in sorted(os.listdir(cartella_raw)):
            if nome_file.endswith(".json"):
                with open(os.path.join(cartella_raw, nome_file), encoding="utf-8") as f:
                    tutti_risultati.extend(json.load(f))
        salva_metriche(tutti_risultati)
        print("Metriche ricalcolate e salvate.")
        return

    print(f"\n[retriever] Inizializzazione retriever per {CURRENT_DOMAIN.upper()}...")
    retriever, _ = costruisci_retriever()

    from generative_phase.runner.experiment_runner import ottieni_o_crea_cache_retrieval
    retrieval_cache = ottieni_o_crea_cache_retrieval(domande, retriever)

    configs_da_eseguire = EXPERIMENT_CONFIGS
    if args.solo_config:
        configs_da_eseguire = [
            c for c in EXPERIMENT_CONFIGS if c["config_id"] == args.solo_config
        ]
        if not configs_da_eseguire:
            print(f"[ERRORE] Config '{args.solo_config}' non trovata in experiment_configs.json.")
            sys.exit(1)

    tutti_risultati = []
    tempo_inizio = time.time()

    for config in configs_da_eseguire:
        risultati_config = esegui_pipeline_config(config, domande, retriever, retrieval_cache, ricalcola_judge=args.ricalcola_judge)
        tutti_risultati.extend(risultati_config)

    if not args.solo_generazione:
        if args.solo_config:
            print(f"\n  [Skip Metriche] Hai runnato solo la config {args.solo_config}. Le metriche globali e i grafici verranno generati solo quando eseguirai tutte le config (o se usi l'opzione --solo-metriche).")
        else:
            salva_metriche(tutti_risultati)

            from generative_phase.analysis.plots import genera_tutti_i_grafici
            genera_tutti_i_grafici(GEN_RESULTS_DIR)

    tempo_totale = time.time() - tempo_inizio
    ore     = int(tempo_totale // 3600)
    minuti  = int((tempo_totale % 3600) // 60)
    secondi = int(tempo_totale % 60)
    print(f"\n✓ Pipeline completata in {ore}h {minuti}m {secondi}s")
    print(f"  Risultati in: {GEN_RESULTS_DIR}")


if __name__ == "__main__":
    main()
