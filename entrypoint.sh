#!/bin/bash

echo "==================================================="
echo "  Sistema di Ottimizzazione RAG e Generazione"
echo "==================================================="
echo ""

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    echo "Attivazione del virtual environment esistente..."
    source .venv/bin/activate
    echo "Dipendenze gia' installate. Salto l'installazione."
else
    echo "Creazione del virtual environment in corso..."
    python3 -m venv .venv

    echo "Attivazione del virtual environment..."
    source .venv/bin/activate

    echo ""
    echo "Seleziona l'ambiente hardware per PyTorch:"
    echo "1) CPU (Piu' leggero, ideale se non hai una GPU NVIDIA)"
    echo "2) CUDA (Pesante, sfrutta l'accelerazione della tua GPU NVIDIA)"
    echo ""
    read -p "Scelta hardware (1 o 2): " hw_scelta

    if [ "$hw_scelta" = "2" ]; then
        echo "Installazione PyTorch con supporto CUDA in corso..."
        pip install torch==2.12.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installazione PyTorch versione CPU in corso..."
        pip install torch==2.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
    fi

    echo ""
    echo "Installazione delle restanti dipendenze in corso..."
    pip install -r src/requirements.txt
fi

echo ""
echo "Seleziona l'operazione da eseguire:"
echo "1) Ottimizzazione RAG (Tutti i blocchi)"
echo "3) Ottimizzazione RAG (Solo Blocco 1)"
echo "4) Ottimizzazione RAG (Solo Blocco 2)"
echo "5) Ottimizzazione RAG (Solo Blocco 3)"
echo "6) Fase Generativa RAG (Tutte le configurazioni)"
echo "7) Generazione Grafici Congiunti"
echo "8) Esci"
echo ""
read -p "Scelta: " scelta

if [ "$scelta" = "1" ]; then
    python src/main.py --phase optimization
elif [ "$scelta" = "3" ]; then
    python src/main.py --phase optimization --solo-blocco 1
elif [ "$scelta" = "4" ]; then
    python src/main.py --phase optimization --solo-blocco 2
elif [ "$scelta" = "5" ]; then
    python src/main.py --phase optimization --solo-blocco 3
elif [ "$scelta" = "6" ]; then
    python src/main.py --phase generation
elif [ "$scelta" = "7" ]; then
    python src/main.py --phase combined-graphs
else
    echo "Uscita in corso..."
fi

echo ""
read -p "Premi Invio per continuare..."
