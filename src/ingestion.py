"""
ingestion.py

Responsable de:
- Crear el esquema SQLite (fuente de verdad del catálogo)
- Ingerir el catálogo inicial desde CSV
- Garantizar idempotencia en la ingesta
- Manejar versiones de ficha
- Registrar eventos aplicados
"""

import sqlite3
from pathlib import Path
from src.config import DB_PATH, CATALOGO_FIELDS, log


# ============================================================
# CONEXIÓN A SQLITE
# ============================================================

def get_connection():
    """Devuelve una conexión SQLite con foreign keys activadas."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ============================================================
# CREACIÓN DEL ESQUEMA
# ============================================================

def create_schema():
    """Crea las tablas necesarias en SQLite si no existen."""
    conn = get_connection()
    cur = conn.cursor()

    # Tabla principal del catálogo
    cur.execute("""
        CREATE TABLE IF NOT EXISTS catalog (
            record_id TEXT PRIMARY KEY,
            product_id TEXT,
            title TEXT,
            brand TEXT,
            color TEXT,
            locale TEXT,
            text TEXT,
            catalog_version INTEGER,
            active INTEGER
        );
    """)

    # Tabla de eventos aplicados
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events_log (
            sequence INTEGER PRIMARY KEY,
            event_id TEXT,
            operation TEXT,
            record_id TEXT,
            timestamp TEXT
        );
    """)

    # Índices recomendados
    cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_product_id ON catalog(product_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_brand ON catalog(brand);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_active ON catalog(active);")

    conn.commit()
    conn.close()
    log("Esquema SQLite creado correctamente.")


# ============================================================
# INGESTA IDEMPOTENTE
# ============================================================

def upsert_catalog_record(conn, record):
    """
    Inserta o actualiza un registro del catálogo de forma idempotente.
    - Si record_id no existe → INSERT
    - Si existe:
        - Si catalog_version es mayor → UPDATE
        - Si es igual → ignorar
        - Si es menor → ignorar (protección contra regresiones)
    """

    cur = conn.cursor()

    record_id = record["record_id"]
    new_version = int(record["catalog_version"])

    # Comprobar si existe
    cur.execute("SELECT catalog_version FROM catalog WHERE record_id = ?", (record_id,))
    row = cur.fetchone()

    if row is None:
        # INSERT
        cur.execute("""
            INSERT INTO catalog (record_id, product_id, title, brand, color, locale, text, catalog_version, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["record_id"],
            record["product_id"],
            record["title"],
            record["brand"],
            record["color"],
            record["locale"],
            record["text"],
            new_version,
            int(record["active"])
        ))
        return "insert"

    else:
        current_version = row[0]

        if new_version > current_version:
            # UPDATE
            cur.execute("""
                UPDATE catalog
                SET product_id = ?, title = ?, brand = ?, color = ?, locale = ?, text = ?, catalog_version = ?, active = ?
                WHERE record_id = ?
            """, (
                record["product_id"],
                record["title"],
                record["brand"],
                record["color"],
                record["locale"],
                record["text"],
                new_version,
                int(record["active"]),
                record_id
            ))
            return "update"

        else:
            # Ignorar (idempotencia)
            return "ignore"


# ============================================================
# DELETE IDEMPOTENTE
# ============================================================

def delete_catalog_record(conn, record_id):
    """
    Marca un registro como inactivo.
    Si no existe, no hace nada (idempotente).
    """
    cur = conn.cursor()
    cur.execute("UPDATE catalog SET active = 0 WHERE record_id = ?", (record_id,))
    return cur.rowcount  # 0 si no existía, 1 si se marcó


# ============================================================
# REGISTRO DE EVENTOS
# ============================================================

def log_event(conn, sequence, event_id, operation, record_id):
    """Registra un evento aplicado para garantizar idempotencia."""
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO events_log (sequence, event_id, operation, record_id, timestamp)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (sequence, event_id, operation, record_id))
    conn.commit()


# ============================================================
# APLICACIÓN DE EVENTOS (se completará en events.py)
# ============================================================

# Aquí más adelante conectaremos con events.py
