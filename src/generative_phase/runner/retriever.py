"""Costruisce il retriever ottimizzato leggendo i parametri da best_params.json."""

import json
from generative_phase.configs.config import BEST_PARAMS_PATH, DOCS_DIR, EMBEDDING_DEVICE, JUDGE_MODEL


def carica_best_params() -> dict:
    """Carica i parametri ottimali dal file JSON generato dall'ottimizzazione."""
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class CustomOptimizedRetriever:
    """Wrapper che implementa le scelte della fase di ottimizzazione RAG."""
    def __init__(self, index, nodi, params):
        self.index = index
        self.nodi = nodi
        self.params = params
        
        # 1. Configura il retriever di base (vettoriale, BM25 o ibrido)
        k_retrieve = params.get("k_retrieve", params.get("k_final", 5))
        retrieval_type = params.get("retrieval_type", "vector")
        
        if retrieval_type == "vector":
            self.base_retriever = index.as_retriever(similarity_top_k=k_retrieve)
        elif retrieval_type == "bm25":
            from llama_index.retrievers.bm25 import BM25Retriever
            self.base_retriever = BM25Retriever.from_defaults(nodes=nodi, similarity_top_k=k_retrieve)
        elif retrieval_type == "hybrid":
            from llama_index.core.retrievers import QueryFusionRetriever
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.llms.ollama import Ollama
            
            retriever_vettoriale = index.as_retriever(similarity_top_k=k_retrieve)
            retriever_bm25 = BM25Retriever.from_defaults(nodes=nodi, similarity_top_k=k_retrieve)
            
            llm_local = Ollama(model=JUDGE_MODEL, base_url="http://localhost:11434", temperature=0.1)
            
            self.base_retriever = QueryFusionRetriever(
                retrievers=[retriever_vettoriale, retriever_bm25],
                similarity_top_k=k_retrieve,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                llm=llm_local
            )
        else:
            self.base_retriever = index.as_retriever(similarity_top_k=k_retrieve)

        # 2. Configura HyDE se qt_strategy è "hyde"
        self.hyde_transform = None
        if params.get("qt_strategy", "none") == "hyde":
            from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
            from llama_index.llms.ollama import Ollama
            llm_local = Ollama(model=JUDGE_MODEL, base_url="http://localhost:11434", temperature=0.1)
            self.hyde_transform = HyDEQueryTransform(llm=llm_local, include_original=True)

        # 3. Configura Reranker
        self.reranker = None
        if params.get("reranking", False):
            from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
            k_final = params.get("k_final", 5)
            print("      - Utilizzo modello locale BAAI/bge-reranker-v2-m3")
            self.reranker = FlagEmbeddingReranker(
                model="BAAI/bge-reranker-v2-m3",
                top_n=k_final,
                use_fp16=False
            )

    def retrieve(self, query: str):
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=query)
        
        # 1. Applica HyDE
        if self.hyde_transform is not None:
            query_bundle = self.hyde_transform(query_bundle)
            
        # 2. Recupero
        nodi_recuperati = self.base_retriever.retrieve(query_bundle)
        
        # 3. Reranker
        if self.reranker is not None:
            nodi_recuperati = self.reranker.postprocess_nodes(
                nodi_recuperati,
                query_str=query_bundle.query_str
            )
        else:
            nodi_recuperati = nodi_recuperati[:self.params["k_final"]]
            
        return nodi_recuperati


def costruisci_retriever():
    """Costruisce il retriever LlamaIndex caricando indice e nodi pre-calcolati."""
    import os
    import pickle
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from generative_phase.configs.config import PROJECT_ROOT, CURRENT_DOMAIN

    params = carica_best_params()

    print(f"  [retriever] Parametri ottimali caricati da: {BEST_PARAMS_PATH}")
    print(f"  [retriever] Splitter: {params['splitter']}, chunk_size: {params['chunk_size']}, "
          f"overlap: {params['chunk_overlap']}")
    print(f"  [retriever] Embedding: {params['embed_model']}")
    print(f"  [retriever] k_final: {params['k_final']}")

    # Modello di embedding ottimale
    modello_embedding = HuggingFaceEmbedding(
        model_name=params["embed_model"],
        device=EMBEDDING_DEVICE
    )

    # Ricerca dinamica della configurazione ottimale pre-calcolata
    risultati_path = os.path.join(PROJECT_ROOT, "results", CURRENT_DOMAIN, "optimization", "raw", "block1_results.json")
    if not os.path.exists(risultati_path):
        raise FileNotFoundError(f"File risultati non trovato in {risultati_path}")
        
    with open(risultati_path, "r", encoding="utf-8") as f:
        block1_results = json.load(f)
        
    matching_config_id = None
    for res in block1_results:
        cfg = res.get("config", {})
        if (cfg.get("embed_model") == params["embed_model"] and
            cfg.get("chunk_size") == params["chunk_size"] and
            cfg.get("overlap") == params["chunk_overlap"] and
            cfg.get("splitter") == params["splitter"]):
            matching_config_id = cfg.get("id")
            break
            
    if matching_config_id is None:
        raise ValueError("I parametri attuali in best_params.json non corrispondono a nessuna configurazione pre-calcolata in block1_results.json!")

    # Percorsi dinamici per la configurazione trovata
    nodes_path = os.path.join(PROJECT_ROOT, "results", CURRENT_DOMAIN, "optimization", "prebuild", "block1_nodes", f"config_{matching_config_id}_nodes.pkl")
    index_dir = os.path.join(PROJECT_ROOT, "results", CURRENT_DOMAIN, "optimization", "prebuild", "block1_indexes", f"config_{matching_config_id}")

    # Caricamento nodi pre-calcolati
    print(f"  [retriever] Trovato match dinamico: config_{matching_config_id}")
    print(f"  [retriever] Caricamento nodi pre-calcolati da: {nodes_path}")
    with open(nodes_path, "rb") as f:
        nodi = pickle.load(f)
    print(f"  [retriever] {len(nodi)} nodi caricati dal disco.")

    # Caricamento indice vettoriale
    print(f"  [retriever] Caricamento indice vettoriale dal disco: {index_dir}")
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    indice = load_index_from_storage(storage_context, embed_model=modello_embedding)

    # Inizializzazione CustomOptimizedRetriever
    print("  [retriever] Inizializzazione CustomOptimizedRetriever...")
    retriever = CustomOptimizedRetriever(indice, nodi, params)
    print("  [retriever] Retriever pronto.\n")

    return retriever, nodi
