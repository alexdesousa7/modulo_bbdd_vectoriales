"""
metrics.py

Responsable de:
- Calcular métricas de ranking (nDCG@10, Recall@10, MRR@10)
- Evaluar la calidad de búsqueda en desarrollo
- Medir latencia p50/p95
- Evaluar fidelidad ANN comparando Qdrant con un oráculo exacto
"""

import time
import numpy as np
import pandas as pd

from src.search_engine import search
from src.embeddings import embed_query, embed_products
from src.config import CATALOGO_COMPLETO, ESCI_MAPPING, TOP_K, log


# ============================================================
# MÉTRICAS DE RANKING
# ============================================================

def dcg_at_k(relevances, k=10):
    """Calcula DCG@k."""
    relevances = np.array(relevances[:k])
    return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))


def ndcg_at_k(relevances, k=10, ideal_relevances=None):
    """Calcula nDCG@k usando todos los juicios disponibles para el ideal."""
    dcg = dcg_at_k(relevances, k)
    source = relevances if ideal_relevances is None else ideal_relevances
    ideal = dcg_at_k(sorted(source, reverse=True), k)
    return dcg / (ideal + 1e-9)


def recall_at_k(relevances, k=10, total_relevant=None):
    """Recall@k sobre el número total de documentos relevantes."""
    relevances = np.array(relevances[:k])
    relevant = np.sum(relevances > 0)
    if total_relevant is None:
        total_relevant = relevant
    return relevant / (total_relevant + 1e-9)


def mrr_at_k(relevances, k=10):
    """MRR@k: Reciprocal Rank del primer relevante."""
    for i, r in enumerate(relevances[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# EVALUACIÓN COMPLETA DE BÚSQUEDA
# ============================================================

def evaluate_search(queries_csv, relevances_csv, model_name="e5_small"):
    """
    Evalúa nDCG@10, Recall@10 y MRR@10 sobre consultas de desarrollo.
    """

    df_q = pd.read_csv(queries_csv)
    df_r = pd.read_csv(relevances_csv)

    ndcgs = []
    recalls = []
    mrrs = []

    for _, row in df_q.iterrows():
        query_id = row["query_id"]
        query_text = row["query_text"]

        # Relevancias de la consulta
        df_rel = df_r[df_r["query_id"] == query_id]

        # Ejecutar búsqueda
        results = search(query_text, top_k=TOP_K, model_name=model_name)

        # Construir el ranking recuperado y todos los juicios de la consulta.
        relevance_by_product = {
            row["product_id"]: ESCI_MAPPING[row["esci_label"]]
            for _, row in df_rel.iterrows()
        }
        ideal_relevances = list(relevance_by_product.values())
        relevances = []
        for res in results:
            product_id = res["product_id"]
            relevances.append(relevance_by_product.get(product_id, 0))

        # Métricas
        ndcgs.append(ndcg_at_k(relevances, TOP_K, ideal_relevances))
        total_relevant = sum(relevance > 0 for relevance in ideal_relevances)
        recalls.append(recall_at_k(relevances, TOP_K, total_relevant))
        mrrs.append(mrr_at_k(relevances, TOP_K))

    metrics = {
        "ndcg@10": float(np.mean(ndcgs)),
        "recall@10": float(np.mean(recalls)),
        "mrr@10": float(np.mean(mrrs)),
    }

    log(f"[METRICS] nDCG@10={metrics['ndcg@10']:.4f}, Recall@10={metrics['recall@10']:.4f}, MRR@10={metrics['mrr@10']:.4f}")

    return metrics


# ============================================================
# LATENCIA p50 / p95
# ============================================================

def measure_latency(query_list, model_name="e5_small", repeats=5, warmup=True):
    """
    Mide latencia p50 y p95 con calentamiento y repeticiones controladas.
    """

    if not query_list:
        raise ValueError("query_list no puede estar vacío")

    if warmup:
        for query in query_list:
            search(query, top_k=TOP_K, model_name=model_name)

    latencies = []

    for _ in range(repeats):
        for query in query_list:
            t0 = time.perf_counter()
            search(query, top_k=TOP_K, model_name=model_name)
            latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    log(
        f"[LATENCY] p50={p50:.4f}s, p95={p95:.4f}s, "
        f"queries={len(query_list)}, repeats={repeats}, warmup={warmup}"
    )

    return {"p50": float(p50), "p95": float(p95)}


# ============================================================
# FIDELIDAD ANN
# ============================================================

def evaluate_ann_fidelity(query_list, model_name="e5_small"):
    """
    Compara resultados ANN (Qdrant) con un oráculo exacto.
    El oráculo exacto se calcula comparando la consulta con todos los productos.
    """

    # El oráculo debe usar el mismo catálogo final que el índice evaluado.
    df = pd.read_csv(CATALOGO_COMPLETO)
    texts = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    ).tolist()
    product_vectors = embed_products(texts, model_name=model_name)
    record_ids = df["record_id"].tolist()

    fidelities = []

    for q in query_list:
        # Embedding de consulta
        q_vec = embed_query(q, model_name=model_name)

        # ANN (Qdrant)
        ann_results = search(q, top_k=TOP_K, model_name=model_name)
        ann_ids = [r["record_id"] for r in ann_results]

        # Oráculo exacto sobre los 15.000 productos del catálogo final.
        scores = product_vectors @ q_vec
        top_indices = np.argsort(-scores)[:TOP_K]
        oracle_ids = [record_ids[index] for index in top_indices]

        # Fidelidad = intersección / TOP_K
        fidelity = len(set(ann_ids) & set(oracle_ids)) / TOP_K
        fidelities.append(fidelity)

    fidelity_mean = float(np.mean(fidelities))

    log(f"[ANN] Fidelidad ANN={fidelity_mean:.4f}")

    return fidelity_mean
