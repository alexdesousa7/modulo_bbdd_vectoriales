"""
filters.py

Responsable de:
- Validar parámetros de filtros (marca, color, active)
- Construir filtros Qdrant de forma segura
- Exponer funciones reutilizables para consultas filtradas
"""

from qdrant_client.models import Filter, FieldCondition, MatchValue


# ============================================================
# VALIDACIÓN DE PARÁMETROS
# ============================================================

def validate_filter_params(brand=None, color=None, active=None):
    """
    Valida los parámetros de filtro antes de construir el filtro Qdrant.
    - brand: debe ser str o None
    - color: debe ser str o None
    - active: debe ser 0, 1 o None
    """

    if brand is not None and not isinstance(brand, str):
        raise ValueError("brand debe ser una cadena o None")

    if color is not None and not isinstance(color, str):
        raise ValueError("color debe ser una cadena o None")

    if active is not None and active not in (0, 1):
        raise ValueError("active debe ser 0, 1 o None")


# ============================================================
# CONSTRUCCIÓN DE FILTROS
# ============================================================

def make_filter(brand=None, color=None, active=None):
    """
    Construye un filtro Qdrant combinando marca, color y active.
    Si todos son None, devuelve None (sin filtro).
    """

    validate_filter_params(brand, color, active)

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
# FILTROS ESPECÍFICOS
# ============================================================

def make_brand_filter(brand: str):
    """Filtro por marca."""
    validate_filter_params(brand=brand)
    return Filter(must=[FieldCondition(key="brand", match=MatchValue(value=brand))])


def make_color_filter(color: str):
    """Filtro por color."""
    validate_filter_params(color=color)
    return Filter(must=[FieldCondition(key="color", match=MatchValue(value=color))])


def make_active_filter(active: int):
    """Filtro por estado activo/inactivo."""
    validate_filter_params(active=active)
    return Filter(must=[FieldCondition(key="active", match=MatchValue(value=int(active)))])
