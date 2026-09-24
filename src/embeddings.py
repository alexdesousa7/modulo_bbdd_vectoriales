"""
embeddings.py

Responsable de:
- Cargar los modelos de embeddings (E5-small, BGE-M3, MPNet)
- Normalizar texto antes de generar embeddings
- Generar embeddings para consultas y productos
- Aplicar prefijos 'query:' y 'passage:' en E5
- Exponer funciones limpias para el resto del sistema
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import (
    EMBEDDING_MODELS,
    NORMALIZATION,
    log
)


# ============================================================
# MODELOS EN MEMORIA (cache)
# ============================================================

_loaded_models = {}  # cache global para evitar recargas


# ============================================================
# NORMALIZACIÓN DE TEXTO
# ============================================================

def normalize_text(text: str) -> str:
    """Normaliza texto según la configuración global."""
    if text is None:
        return ""

    t = text

    if NORMALIZATION["lowercase"]:
        t = t.lower()

    if NORMALIZATION["remove_extra_spaces"]:
        t = " ".join(t.split())

    return t


# ============================================================
# CARGA DE MODELOS
# ============================================================

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Carga un modelo de embeddings en memoria si no está ya cargado.
    Devuelve el modelo listo para generar embeddings.
    """

    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"Modelo desconocido: {model_name}")

    if model_name in _loaded_models:
        return _loaded_models[model_name]

    model_path = EMBEDDING_MODELS[model_name]
    model = SentenceTransformer(model_path)

    _loaded_models[model_name] = model
    log(f"[EMBEDDINGS] Modelo cargado: {model_name} ({model_path})")

    return model


# ============================================================
# DIMENSIÓN DEL VECTOR POR MODELO
# ============================================================

def get_vector_dimension(model_name: str) -> int:
    """Devuelve la dimensión del embedding para un modelo."""
    model = load_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()


# ============================================================
# EMBEDDINGS PARA CONSULTAS
# ============================================================

def embed_query(text: str, model_name: str = "e5_small") -> np.ndarray:
    """
    Genera un embedding para una consulta.
    - Aplica normalización
    - Aplica prefijo 'query:' para E5
    """

    model = load_embedding_model(model_name)
    t = normalize_text(text)

    # Prefijo E5
    if model_name == "e5_small":
        t = f"query: {t}"

    vector = model.encode(t, normalize_embeddings=False)
    return np.array(vector, dtype=np.float32)


# ============================================================
# EMBEDDINGS PARA PRODUCTOS
# ============================================================

def embed_product(text: str, model_name: str = "e5_small") -> np.ndarray:
    """
    Genera un embedding para un producto del catálogo.
    - Aplica normalización
    - Aplica prefijo 'passage:' para E5
    """

    model = load_embedding_model(model_name)
    t = normalize_text(text)

    # Prefijo E5
    if model_name == "e5_small":
        t = f"passage: {t}"

    vector = model.encode(t, normalize_embeddings=False)
    return np.array(vector, dtype=np.float32)


def embed_products(texts, model_name: str = "e5_small", batch_size: int = 64) -> np.ndarray:
    """Genera embeddings de productos en lotes con el mismo contrato que embed_product."""
    model = load_embedding_model(model_name)
    normalized = [normalize_text(text) for text in texts]
    if model_name == "e5_small":
        normalized = [f"passage: {text}" for text in normalized]
    vectors = model.encode(
        normalized,
        batch_size=batch_size,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
