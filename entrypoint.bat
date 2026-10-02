@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Sistema di Ottimizzazione RAG e Generazione
echo ===================================================
echo.

cd /d "%~dp0"

IF EXIST ".venv" (
    echo Attivazione del virtual environment esistente...
    call .venv\Scripts\activate.bat
    echo Dipendenze gia' installate. Salto l'installazione.
    goto menu
)

echo Creazione del virtual environment in corso...
python -m venv .venv

echo Attivazione del virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Seleziona l'ambiente hardware per PyTorch:
echo 1) CPU (Piu' leggero, ideale se non hai una GPU NVIDIA)
echo 2) CUDA (Pesante, sfrutta l'accelerazione della tua GPU NVIDIA)
echo.
set /p hw_scelta="Scelta hardware (1 o 2): "

if "%hw_scelta%"=="2" (
    echo Installazione PyTorch con supporto CUDA in corso...
    pip install torch==2.12.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installazione PyTorch versione CPU in corso...
    pip install torch==2.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Installazione delle restanti dipendenze in corso...
pip install -r src\requirements.txt

:menu
echo.
echo Seleziona l'operazione da eseguire:
echo 1) Ottimizzazione RAG (Tutti i blocchi)
echo 3) Ottimizzazione RAG (Solo Blocco 1)
echo 4) Ottimizzazione RAG (Solo Blocco 2)
echo 5) Ottimizzazione RAG (Solo Blocco 3)
echo 6) Fase Generativa RAG (Tutte le configurazioni)
echo 7) Generazione Grafici Congiunti
echo 8) Esci
echo.
set /p scelta="Scelta: "

if "%scelta%"=="1" (
    python src\main.py --phase optimization
) else if "%scelta%"=="3" (
    python src\main.py --phase optimization --solo-blocco 1
) else if "%scelta%"=="4" (
    python src\main.py --phase optimization --solo-blocco 2
) else if "%scelta%"=="5" (
    python src\main.py --phase optimization --solo-blocco 3
) else if "%scelta%"=="6" (
    python src\main.py --phase generation
) else if "%scelta%"=="7" (
    python src\main.py --phase combined-graphs
) else (
    echo Uscita in corso...
)

echo.
pause
