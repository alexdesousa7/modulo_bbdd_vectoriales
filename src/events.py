"""
events.py

Responsable de:
- Aplicar eventos del catálogo (UPSERT / DELETE)
- Garantizar idempotencia mediante events_log
- Sincronizar SQLite y Qdrant
- Verificar visibilidad tras cada evento
"""

import pandas as pd
from src.ingestion import (
    get_connection,
    upsert_catalog_record,
    delete_catalog_record,
    log_event
)
from src.vector_ingestion import (
    get_qdrant_client,
    upsert_vector,
    delete_vector
)
from src.search_engine import search
from src.config import log


# ============================================================
# COMPROBAR SI UN EVENTO YA FUE APLICADO
# ============================================================

def event_already_applied(conn, sequence: int) -> bool:
    """Devuelve True si el evento ya está registrado en events_log."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM events_log WHERE sequence = ?", (sequence,))
    return cur.fetchone() is not None


# ============================================================
# VERIFICACIÓN DE VISIBILIDAD
# ============================================================

def verify_visibility(record_id: str, model_name: str, query_text=None):
    """
    Verifica visibilidad del registro:
    - lectura por ID en SQLite
    - búsqueda vectorial en Qdrant
    """

    conn = get_connection()
    cur = conn.cursor()

    # Lectura por ID en SQLite
    cur.execute("SELECT active FROM catalog WHERE record_id = ?", (record_id,))
    row = cur.fetchone()

    if row is None:
        sqlite_status = "not_found"
    else:
        sqlite_status = "active" if row[0] == 1 else "inactive"

    # Lectura directa por ID, que no depende de que el UUID tenga significado semántico.
    client = get_qdrant_client()
    collection_name = f"aurum_{model_name}"
    points = client.retrieve(collection_name=collection_name, ids=[record_id])
    qdrant_status = "found" if points else "not_found"

    # Consulta semántica opcional para verificar también la ruta de búsqueda.
    search_status = None
    if query_text:
        hits = search(query_text, top_k=5, model_name=model_name)
        search_status = "found" if any(h["record_id"] == record_id for h in hits) else "not_found"

    return {
        "sqlite": sqlite_status,
        "qdrant": qdrant_status,
        "search": search_status,
    }


# ============================================================
# APLICAR UN EVENTO INDIVIDUAL
# ============================================================

def apply_event(row, model_name: str):
    """
    Aplica un evento individual:
    - UPSERT → actualizar SQLite + Qdrant
    - DELETE → marcar inactivo en SQLite + borrar vector en Qdrant
    """

    conn = get_connection()
    sequence = int(row["sequence"])
    event_id = row["event_id"]
    operation = row["operation"]
    record_id = row["record_id"]

    # Idempotencia: si ya está aplicado, no hacer nada
    if event_already_applied(conn, sequence):
        log(f"[EVENTS] Evento {sequence} ya aplicado → ignorado")
        conn.close()
        return "ignored"

    # UPSERT
    if operation.upper() == "UPSERT":
        record = {
            "record_id": row["record_id"],
            "product_id": row["product_id"],
            "title": row["title"],
            "brand": row["brand"],
            "color": row["color"],
            "locale": row["locale"],
            "text": row["text"],
            "catalog_version": int(row["catalog_version"]),
            "active": int(row["active"])
        }

        # SQLite
        action = upsert_catalog_record(conn, record)

        # Qdrant
        payload = {
            "record_id": row["record_id"],
            "product_id": row["product_id"],
            "title": row["title"],
            "brand": row["brand"],
            "color": row["color"],
            "active": int(row["active"]),
            "catalog_version": int(row["catalog_version"])
        }

        # Generar embedding del producto
        from src.embeddings import embed_product
        vector = embed_product(row["title"] + " " + row["text"], model_name=model_name)

        upsert_vector(model_name, record_id, vector, payload)

        log_event(conn, sequence, event_id, "UPSERT", record_id)
        conn.commit()
        conn.close()

        log(f"[EVENTS] UPSERT aplicado: {record_id} ({action})")
        return "upsert"

    # DELETE
    elif operation.upper() == "DELETE":
        delete_catalog_record(conn, record_id)
        delete_vector(model_name, record_id)

        log_event(conn, sequence, event_id, "DELETE", record_id)
        conn.commit()
        conn.close()

        log(f"[EVENTS] DELETE aplicado: {record_id}")
        return "delete"

    else:
        raise ValueError(f"Operación desconocida: {operation}")


# ============================================================
# APLICAR TODOS LOS EVENTOS DEL CSV
# ============================================================

def apply_catalog_events(csv_path, model_name: str):
    """
    Aplica todos los eventos del catálogo en orden.
    """

    df = pd.read_csv(csv_path)

    for _, row in df.sort_values("sequence").iterrows():
        apply_event(row, model_name=model_name)

    log("[EVENTS] Todos los eventos aplicados correctamente.")
