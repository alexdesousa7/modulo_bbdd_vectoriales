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
from src.embeddings import embed_query, embed_product
from src.config import ESCI_MAPPING, TOP_K, log


# ============================================================
# MÉTRICAS DE RANKING
# ============================================================

def dcg_at_k(relevances, k=10):
    """Calcula DCG@k."""
    relevances = np.array(relevances[:k])
    return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))


def ndcg_at_k(relevances, k=10):
    """Calcula nDCG@k."""
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / (ideal + 1e-9)


def recall_at_k(relevances, k=10):
    """Recall@k: proporción de relevantes recuperados."""
    relevances = np.array(relevances[:k])
    relevant = np.sum(relevances > 0)
    total_relevant = np.sum(relevances)
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

        # Construir vector de relevancias
        relevances = []
        for res in results:
            product_id = res["product_id"]
            rel_row = df_rel[df_rel["product_id"] == product_id]

            if len(rel_row) == 0:
                relevances.append(0)
            else:
                esc = rel_row.iloc[0]["esci_label"]
                relevances.append(ESCI_MAPPING[esc])

        # Métricas
        ndcgs.append(ndcg_at_k(relevances, TOP_K))
        recalls.append(recall_at_k(relevances, TOP_K))
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

def measure_latency(query_list, model_name="e5_small"):
    """
    Mide latencia p50 y p95 sobre una lista de consultas.
    """

    latencies = []

    for q in query_list:
        t0 = time.time()
        _ = search(q, top_k=TOP_K, model_name=model_name)
        t1 = time.time()
        latencies.append(t1 - t0)

    latencies = np.array(latencies)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    log(f"[LATENCY] p50={p50:.4f}s, p95={p95:.4f}s")

    return {"p50": float(p50), "p95": float(p95)}


# ============================================================
# FIDELIDAD ANN
# ============================================================

def evaluate_ann_fidelity(query_list, model_name="e5_small"):
    """
    Compara resultados ANN (Qdrant) con un oráculo exacto.
    El oráculo exacto se calcula comparando la consulta con todos los productos.
    """

    # Cargar catálogo completo
    df = pd.read_csv("../data/catalogo_muestra.csv")

    fidelities = []

    for q in query_list:
        # Embedding de consulta
        q_vec = embed_query(q, model_name=model_name)

        # ANN (Qdrant)
        ann_results = search(q, top_k=TOP_K, model_name=model_name)
        ann_ids = [r["record_id"] for r in ann_results]

        # Oráculo exacto
        scores = []
        for _, row in df.iterrows():
            text = row["title"] + " " + row["text"]
            vec = embed_product(text, model_name=model_name)
            score = np.dot(q_vec, vec)
            scores.append((row["record_id"], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        oracle_ids = [x[0] for x in scores[:TOP_K]]

        # Fidelidad = intersección / TOP_K
        fidelity = len(set(ann_ids) & set(oracle_ids)) / TOP_K
        fidelities.append(fidelity)

    fidelity_mean = float(np.mean(fidelities))

    log(f"[ANN] Fidelidad ANN={fidelity_mean:.4f}")

    return fidelity_mean
