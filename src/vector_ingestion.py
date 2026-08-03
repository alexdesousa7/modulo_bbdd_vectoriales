"""
vector_ingestion.py

Responsable de:
- Crear colecciones Qdrant por modelo de embeddings
- Configurar HNSW (ANN)
- Ingerir vectores con payloads (idempotente)
- Eliminar vectores (idempotente)
- Mantener una colección por modelo de embeddings
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    HnswConfigDiff,
    PointStruct
)

from src.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_MODELS,
    HNSW_CONFIG,
    SEARCH_PARAMS,
    log
)


# ============================================================
# CONEXIÓN A QDRANT
# ============================================================

def get_qdrant_client():
    """Devuelve un cliente Qdrant conectado al host local."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ============================================================
# CREACIÓN DE COLECCIONES POR MODELO
# ============================================================

def create_qdrant_collection(model_name: str, vector_dim: int, metric: str):
    """
    Crea una colección Qdrant para un modelo de embeddings específico.
    - model_name: nombre del modelo (e5_small, bge_m3, mpnet)
    - vector_dim: dimensión del embedding
    - metric: 'dot' o 'cosine'
    """

    client = get_qdrant_client()
    collection_name = f"aurum_{model_name}"

    # Selección de métrica
    if metric == "dot":
        distance = Distance.DOT
    else:
        distance = Distance.COSINE

    # Crear colección (recreate garantiza idempotencia)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=distance
        ),
        hnsw_config=HnswConfigDiff(
            m=HNSW_CONFIG["m"],
            ef_construct=HNSW_CONFIG["ef_construct"],
            full_scan_threshold=HNSW_CONFIG["full_scan_threshold"]  # ← FIX OBLIGATORIO
        )
    )

    log(f"[QDRANT] Colección creada: {collection_name} (dim={vector_dim}, metric={metric})")


# ============================================================
# INGESTA VECTORIAL IDEMPOTENTE
# ============================================================

def upsert_vector(model_name: str, record_id: str, vector, payload: dict):
    """
    Inserta o actualiza un vector en Qdrant de forma idempotente.
    - model_name: nombre del modelo (e5_small, bge_m3, mpnet)
    - record_id: ID del punto
    - vector: embedding generado
    - payload: metadatos del producto
    """

    client = get_qdrant_client()
    collection_name = f"aurum_{model_name}"

    point = PointStruct(
        id=record_id,
        vector=vector,
        payload=payload
    )

    client.upsert(
        collection_name=collection_name,
        points=[point]
    )

    return "upsert"


# ============================================================
# BORRADO IDEMPOTENTE
# ============================================================

def delete_vector(model_name: str, record_id: str):
    """
    Elimina un vector de Qdrant de forma idempotente.
    Si no existe, no pasa nada.
    """

    client = get_qdrant_client()
    collection_name = f"aurum_{model_name}"

    client.delete(
        collection_name=collection_name,
        points_selector={"points": [record_id]}
    )

    return "delete"