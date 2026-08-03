"""
utils.py

Funciones auxiliares para:
- limpieza y normalización de texto
- normalización de marca y color
- medición de tiempos
- logging estructurado
- lectura/escritura segura de CSV
- validación de campos
"""

import time
import pandas as pd
from src.config import log


# ============================================================
# LIMPIEZA DE TEXTO
# ============================================================

def clean_text(text: str) -> str:
    """Limpia texto eliminando espacios extra y normalizando."""
    if text is None:
        return ""
    t = str(text).strip()
    t = " ".join(t.split())
    return t


def normalize_brand(brand: str) -> str:
    """Normaliza marca a formato consistente."""
    if brand is None:
        return ""
    return clean_text(brand).upper()


def normalize_color(color: str) -> str:
    """Normaliza color a formato consistente."""
    if color is None:
        return ""
    return clean_text(color).upper()


# ============================================================
# MEDICIÓN DE TIEMPO
# ============================================================

def timeit(func):
    """
    Decorador para medir duración de funciones.
    Útil para notebooks y análisis de rendimiento.
    """

    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        log(f"[TIME] {func.__name__} tardó {t1 - t0:.4f}s")
        return result

    return wrapper


# ============================================================
# LOGGING ESTRUCTURADO
# ============================================================

def log_section(title: str):
    """Imprime una sección visual en logs."""
    log("")
    log("=" * 60)
    log(f"{title}")
    log("=" * 60)


def log_error(msg: str):
    """Log de error."""
    log(f"[ERROR] {msg}")


# ============================================================
# LECTURA/ESCRITURA SEGURA DE CSV
# ============================================================

def safe_read_csv(path: str):
    """Lee un CSV con manejo de errores."""
    try:
        df = pd.read_csv(path)
        log(f"[CSV] Cargado: {path} ({len(df)} filas)")
        return df
    except Exception as e:
        log_error(f"No se pudo leer CSV {path}: {e}")
        return pd.DataFrame()


def safe_write_csv(df: pd.DataFrame, path: str):
    """Escribe un CSV con manejo de errores."""
    try:
        df.to_csv(path, index=False)
        log(f"[CSV] Guardado: {path} ({len(df)} filas)")
    except Exception as e:
        log_error(f"No se pudo escribir CSV {path}: {e}")


# ============================================================
# VALIDACIÓN DE CAMPOS
# ============================================================

def ensure_fields(record: dict, required_fields: list):
    """
    Verifica que un diccionario contiene todos los campos requeridos.
    Lanza ValueError si falta alguno.
    """
    missing = [f for f in required_fields if f not in record]
    if missing:
        raise ValueError(f"Faltan campos requeridos: {missing}")
