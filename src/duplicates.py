"""
duplicates.py

Regla de duplicados para Aurum Market:
- Calibración del umbral con altas_desarrollo.csv
- Evaluación (precision, recall, F1)
- Aplicación a altas_evaluacion.csv
"""

import pandas as pd
import numpy as np
import sqlite3

from src.embeddings import embed_product
from src.search_engine import search
from src.config import DB_PATH, log


# ============================================================
# UTILIDAD: recuperar producto del catálogo por product_id
# ============================================================

def get_catalog_product(product_id: str):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT product_id, title, text
        FROM catalog
        WHERE product_id = ?
        AND active = 1
    """
    row = conn.execute(query, (product_id,)).fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "product_id": row[0],
        "title": row[1],
        "text": row[2]
    }


# ============================================================
# SIMILITUD ENTRE EMBEDDINGS
# ============================================================

def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Similitud mediante dot product (E5-small)."""
    return float(np.dot(vec1, vec2))


# ============================================================
# CALIBRACIÓN DEL UMBRAL
# ============================================================

def calibrate_threshold(csv_path: str, model_name: str = "e5_small"):
    """
    Calibra el umbral de duplicados usando altas_desarrollo.csv.
    Devuelve el umbral que maximiza F1.
    """

    df = pd.read_csv(csv_path)

    similarities = []
    labels = []

    for _, row in df.iterrows():

        incoming_text = f"{row['title']} {row['text']}"
        incoming_vec = embed_product(incoming_text, model_name=model_name)

        if row["is_duplicate"]:
            # comparar con el producto real del catálogo
            ref_product = get_catalog_product(row["reference_product_id"])
            if ref_product is None:
                continue

            ref_text = f"{ref_product['title']} {ref_product['text']}"
            ref_vec = embed_product(ref_text, model_name=model_name)

            sim = compute_similarity(incoming_vec, ref_vec)
            similarities.append(sim)
            labels.append(1)

        else:
            # buscar el más parecido del catálogo
            result = search(incoming_text, top_k=1, model_name=model_name)[0]
            sim = result["score"]

            similarities.append(sim)
            labels.append(0)

    similarities = np.array(similarities)
    labels = np.array(labels)

    # thresholds entre percentiles 50 y 99
    candidate_thresholds = np.percentile(similarities, np.linspace(50, 99, 200))

    best_f1 = -1
    best_threshold = None

    for th in candidate_thresholds:
        preds = (similarities >= th).astype(int)

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    log(f"[DUPLICATES] Umbral óptimo: {best_threshold:.4f} (F1={best_f1:.4f})")
    return best_threshold


# ============================================================
# EVALUACIÓN DE LA REGLA
# ============================================================

def evaluate_rule(csv_path: str, threshold: float, model_name: str = "e5_small"):
    """
    Evalúa precision, recall y F1 sobre altas_desarrollo.csv.
    """

    df = pd.read_csv(csv_path)

    preds = []
    labels = []

    for _, row in df.iterrows():

        incoming_text = f"{row['title']} {row['text']}"
        incoming_vec = embed_product(incoming_text, model_name=model_name)

        if row["is_duplicate"]:
            ref_product = get_catalog_product(row["reference_product_id"])
            if ref_product is None:
                continue

            ref_text = f"{ref_product['title']} {ref_product['text']}"
            ref_vec = embed_product(ref_text, model_name=model_name)

            sim = compute_similarity(incoming_vec, ref_vec)
            pred = int(sim >= threshold)

            preds.append(pred)
            labels.append(1)

        else:
            result = search(incoming_text, top_k=1, model_name=model_name)[0]
            sim = result["score"]

            pred = int(sim >= threshold)

            preds.append(pred)
            labels.append(0)

    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    log(f"[DUPLICATES] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    return precision, recall, f1


# ============================================================
# APLICAR LA REGLA A EVALUACIÓN
# ============================================================

def apply_rule_to_evaluation(csv_path: str, threshold: float, model_name: str = "e5_small"):
    """
    Aplica la regla de duplicados a altas_evaluacion.csv.
    Genera resultados_duplicados.csv en results/.
    """

    df = pd.read_csv(csv_path)

    results = []

    for _, row in df.iterrows():

        incoming_text = f"{row['title']} {row['text']}"
        incoming_vec = embed_product(incoming_text, model_name=model_name)

        # buscar el más parecido del catálogo
        result = search(incoming_text, top_k=1, model_name=model_name)[0]
        sim = result["score"]

        pred = int(sim >= threshold)

        results.append({
            "incoming_id": row["incoming_id"],
            "predicted_duplicate": bool(pred),
            "matched_product_id": result["product_id"] if pred else "",
            "score": sim
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv("../results/resultados_duplicados.csv", index=False)

    log("[DUPLICATES] Archivo resultados_duplicados.csv generado correctamente.")
