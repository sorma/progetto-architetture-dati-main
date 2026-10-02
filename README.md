# Sistema RAG: Ottimizzazione & Generazione

Questo progetto implementa una pipeline completa per l'ottimizzazione e la generazione di sistemi RAG su basi di conoscenza specifiche.

## Configurazione Iniziale

Prima di avviare il progetto, è necessario configurare le variabili d'ambiente.
Copia il file di esempio (che non contiene credenziali reali) e modifica la copia:

```bash
cp src/.env.example src/.env
```

Su Windows PowerShell:

```powershell
Copy-Item src/.env.example src/.env
```

Il file `src/.env` è escluso da Git, mentre `src/.env.example` documenta la configurazione richiesta.

Esempio di contenuto per `src/.env`:
```env
# Chiavi API Groq (separate da virgola senza spazi)
GROQ_API_KEYS=la_tua_chiave_1,la_tua_chiave_2

# Dominio attualmente in esecuzione
DOMAIN=

# Reranker locale Cross-Encoder (True) oppure LLM-based via Groq (False)
USE_LOCAL_RERANKER=True

# Dispositivo per i modelli di embedding (cpu oppure cuda)
EMBEDDING_DEVICE=cpu

# Modello LLM usato come judge per la fase generativa (fisso, locale)
JUDGE_MODEL=llama3.1:8b

# Configurazione LLM Locale (Ollama)
LOCAL_LLM_URL=http://localhost:11434/v1
```

## Avvio Rapido

### Su Windows:
Doppio click su `entrypoint.bat` oppure esegui da terminale:
```cmd
.\entrypoint.bat
```

### Su Linux/MacOS:
Rendi eseguibile lo script e avvialo:
```bash
chmod +x entrypoint.sh
./entrypoint.sh
```

Lo script si occuperà automaticamente di creare il virtual environment, installare i pacchetti e mostrarti un menù per selezionare l'operazione desiderata.
