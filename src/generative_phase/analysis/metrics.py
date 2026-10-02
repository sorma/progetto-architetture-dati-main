"""Modulo per il calcolo delle metriche generative."""

import os
import pandas as pd


def aggiungi_campi_derivati(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonne derivate (correttezza) al dataframe delle risposte."""
    # Una risposta si considera tentata se la generazione non è un rifiuto standard (non si è astenuto)
    frasi_astensione = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "not enough information"
    ]
    df["is_attempted"] = df["generated_answer"].apply(
        lambda testo: 0 if any(frase in testo.lower() for frase in frasi_astensione) else 1
    )

    # Classificazione TP, TN, FP, FN
    def classify(row):
        is_attempted = row["is_attempted"]
        score = row["score"]  # Ora è 0 o 1
        qtype = row["question_type"]
        
        tp = tn = fp = fn = 0
        
        if qtype != "unanswerable":
            if is_attempted and score == 1:
                tp = 1
            elif is_attempted and score == 0:
                fp = 1
            elif not is_attempted:
                fn = 1
        else:
            if not is_attempted:
                tn = 1
            elif is_attempted:
                fp = 1
                
        return pd.Series([tp, tn, fp, fn], index=["TP", "TN", "FP", "FN"])

    df[["TP", "TN", "FP", "FN"]] = df.apply(classify, axis=1)

    return df


def calcola_metriche_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le metriche aggregate per ciascuna configurazione 
    (config_id, model, prompt_strategy, domain) sommando le confusion matrix.

    Returns:
        DataFrame con una riga per combinazione, pronto per export in JSON.
    """
    righe_aggregate = []

    for (config_id, model, strategia, dominio), gruppo in df.groupby(
        ["config_id", "model", "prompt_strategy", "domain"]
    ):
        totale = len(gruppo)
        tp = gruppo["TP"].sum()
        tn = gruppo["TN"].sum()
        fp = gruppo["FP"].sum()
        fn = gruppo["FN"].sum()

        accuracy = (tp + tn) / totale if totale > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        righe_aggregate.append({
            "config_id":             config_id,
            "model":                 model,
            "prompt_strategy":       strategia,
            "domain":                dominio,
            "total_questions":       totale,
            "TP":                    int(tp),
            "TN":                    int(tn),
            "FP":                    int(fp),
            "FN":                    int(fn),
            "accuracy":              round(accuracy,  4),
            "precision":             round(precision, 4),
            "recall":                round(recall,    4),
            "f1":                    round(f1,        4),
            "avg_latency_sec":       round(gruppo["latency_sec"].mean(), 3),
            "total_input_tokens":    int(gruppo["input_tokens"].sum()),
            "total_output_tokens":   int(gruppo["output_tokens"].sum()),
            "total_tokens":          int(gruppo["total_tokens"].sum())
        })

    return pd.DataFrame(righe_aggregate)
