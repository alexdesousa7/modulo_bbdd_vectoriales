"""
search_engine.py

Responsable de:
- Ejecutar búsquedas vectoriales en Qdrant
- Generar embeddings de consulta
- Aplicar filtros nativos (marca, color, active)
- Normalizar resultados
- Exponer una interfaz limpia para el resto del sistema
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.embeddings import embed_query
from src.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    TOP_K,
    SEARCH_PARAMS,
    log
)


# ============================================================
# CONEXIÓN A QDRANT
# ============================================================

def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ============================================================
# CONSTRUCCIÓN DE FILTROS
# ============================================================

def build_filter(brand=None, color=None, active=None):
    """
    Construye un filtro Qdrant para búsqueda vectorial.
    - brand: filtrar por marca exacta
    - color: filtrar por color exacto
    - active: 1 o 0
    """

    conditions = []

    if brand is not None:
        conditions.append(
            FieldCondition(key="brand", match=MatchValue(value=brand))
        )

    if color is not None:
        conditions.append(
            FieldCondition(key="color", match=MatchValue(value=color))
        )

    if active is not None:
        conditions.append(
            FieldCondition(key="active", match=MatchValue(value=int(active)))
        )

    if len(conditions) == 0:
        return None

    return Filter(must=conditions)


# ============================================================
# NORMALIZACIÓN DE RESULTADOS
# ============================================================

def normalize_result(hit, rank):
    """
    Convierte un resultado Qdrant en un diccionario limpio.
    """

    payload = hit.payload

    return {
        "rank": rank,
        "record_id": hit.id,
        "product_id": payload.get("product_id"),
        "title": payload.get("title"),
        "brand": payload.get("brand"),
        "color": payload.get("color"),
        "active": payload.get("active"),
        "score": hit.score,
    }


# ============================================================
# BÚSQUEDA VECTORIAL
# ============================================================

def search(query_text: str, top_k: int = TOP_K, model_name: str = "e5_small", filters=None):
    """
    Ejecuta una búsqueda vectorial en Qdrant.
    - query_text: texto de la consulta
    - top_k: número de resultados
    - model_name: modelo de embeddings a usar
    - filters: filtro Qdrant (marca, color, active)
    """

    client = get_qdrant_client()
    collection_name = f"aurum_{model_name}"

    # Generar embedding de consulta
    query_vector = embed_query(query_text, model_name=model_name)

    # Ejecutar búsqueda
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        search_params=SEARCH_PARAMS,
        query_filter=filters
    )

    # En versiones recientes de qdrant_client, query_points devuelve QueryResponse.
    hits = response.points if hasattr(response, "points") else response

    # Normalizar resultados
    results = []
    for i, hit in enumerate(hits, start=1):
        results.append(normalize_result(hit, rank=i))

    return results
