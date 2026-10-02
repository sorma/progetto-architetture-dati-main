"""LLM-as-a-Judge per la valutazione semantica delle risposte generate."""

import json
import os
import re
import time


from generative_phase.configs.config import PROMPTS_DIR, JUDGE_MODEL, CURRENT_DOMAIN


def carica_prompt_judge() -> dict:
    """Carica il prompt del judge specifico per il dominio corrente."""
    percorso = os.path.join(PROMPTS_DIR, "eval_prompts.json")
    with open(percorso, encoding="utf-8") as f:
        prompt_template = json.load(f)["judge_prompt"]
    return prompt_template


def valuta_risposta(
    domanda: str,
    risposta_reference: str,
    risposta_generata: str
) -> dict:
    """Usa un LLM-as-a-Judge per valutare la bontà di una singola risposta generata."""
    prompt_config = carica_prompt_judge()

    # Se la reference è vuota, chiediamo al judge di verificare che il modello si sia astenuto
    if not risposta_reference.strip():
        risposta_reference = "[UNANSWERABLE] The reference does not provide an answer because this question is unanswerable from the given context. A correct generated answer MUST abstain or state that the information is missing."

    # Compila il template inserendo domanda e risposte (uso .replace per non rompere il JSON)
    contenuto_utente = prompt_config["user_template"].replace(
        "{question}", domanda
    ).replace(
        "{reference_answer}", risposta_reference
    ).replace(
        "{generated_answer}", risposta_generata
    )

    from openai import OpenAI
    from generative_phase.configs.config import LOCAL_LLM_URL
    
    current_client = OpenAI(base_url=LOCAL_LLM_URL, api_key="ollama")
    
    risposta = current_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user",   "content": contenuto_utente}
        ],
        temperature=0.0
    )

    testo_raw = risposta.choices[0].message.content.strip()

    # Estrae il JSON dall'output del judge usando regex
    match = re.search(r'\{.*?\}', testo_raw, re.DOTALL)
    if match:
        try:
            risultato = json.loads(match.group())
        except json.JSONDecodeError:
            # In caso di errore nel JSON, penalizziamo con score 0
            risultato = {"score": 0, "justification": f"parse error: {testo_raw[:100]}"}
    else:
        # Penalizziamo se l'LLM non ha prodotto JSON validi
        risultato = {"score": 0, "justification": f"no JSON found: {testo_raw[:100]}"}

    # Aggiunge il conteggio dei token del judge al risultato
    risultato["judge_input_tokens"]  = risposta.usage.prompt_tokens
    risultato["judge_output_tokens"] = risposta.usage.completion_tokens

    return risultato
