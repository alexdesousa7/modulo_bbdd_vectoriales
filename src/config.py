"""
config.py

Configuración central del proyecto Aurum Market.

Este módulo define:
- rutas base del proyecto
- configuración de SQLite (fuente de verdad)
- configuración de Qdrant (motor vectorial)
- modelos de embeddings utilizados
- parámetros ANN recomendados
- opciones de normalización de texto
- constantes globales para el pipeline

Todas las demás capas del sistema importan esta configuración.
"""

from pathlib import Path

# ============================
# RUTAS BASE DEL PROYECTO
# ============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "aurum.db"

# ============================
# ARCHIVOS DE DATOS
# ============================

CATALOGO_MUESTRA = DATA_DIR / "catalogo_muestra.csv"
CATALOGO_COMPLETO = DATA_DIR / "catalogo_productos.csv.gz"

CONSULTAS_DESARROLLO = DATA_DIR / "consultas_desarrollo.csv"
RELEVANCIAS_DESARROLLO = DATA_DIR / "relevancias_desarrollo.csv"

CONSULTAS_EVALUACION = DATA_DIR / "consultas_evaluacion.csv"
CONSULTAS_FILTRADAS = DATA_DIR / "consultas_filtradas.csv"

EVENTOS_CATALOGO = DATA_DIR / "eventos_catalogo.csv"

ALTAS_DESARROLLO = DATA_DIR / "altas_desarrollo.csv"
ALTAS_EVALUACION = DATA_DIR / "altas_evaluacion.csv"

MANIFEST = DATA_DIR / "manifest.json"

# ============================
# CONFIGURACIÓN DE QDRANT
# ============================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "aurum_catalog"

# Métrica recomendada para E5 (dot product)
QDRANT_METRIC = "dot"

# Parámetros ANN recomendados para HNSW
# (los justificaremos en el informe)
HNSW_CONFIG = {
    "m": 32,              # grado del grafo
    "ef_construct": 200,  # calidad del índice
    "full_scan_threshold": 10000,   # # Umbral para usar full-scan vs HNSW (obligatorio en Qdrant >=1.7)
}

# Parámetros de búsqueda ANN
SEARCH_PARAMS = {
    "hnsw_ef": 128  # equilibrio entre latencia y fidelidad
}

# ============================
# MODELOS DE EMBEDDINGS
# ============================

EMBEDDING_MODELS = {
    "e5_small": "intfloat/multilingual-e5-small",     # modelo final
    "bge_m3": "BAAI/bge-m3",                          # modelo alternativo
    "mpnet": "sentence-transformers/all-mpnet-base-v2" # baseline vectorial
}

# Modelo final por defecto
DEFAULT_MODEL = "e5_small"

# ============================
# NORMALIZACIÓN DE TEXTO
# ============================

NORMALIZATION = {
    "lowercase": True,
    "strip_accents": False,   # lo justificaremos en el informe
    "remove_extra_spaces": True,
}

# ============================
# CAMPOS DEL CATÁLOGO
# ============================

CATALOGO_FIELDS = [
    "record_id",
    "product_id",
    "title",
    "brand",
    "color",
    "locale",
    "text",
    "catalog_version",
    "active",
]

# ============================
# PARÁMETROS DEL PIPELINE
# ============================

TOP_K = 10  # tamaño del ranking solicitado por el enunciado

# Etiquetas ESCI → relevancia graduada
ESCI_MAPPING = {
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0
}

# ============================
# LOGGING SIMPLE
# ============================

def log(msg: str):
    print(f"[AURUM] {msg}")
