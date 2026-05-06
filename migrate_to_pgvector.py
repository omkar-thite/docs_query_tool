"""
load_to_pgvector.py
────────────────────────────────────────────────────────────────────────────
Loads children.json  →  document_chunks   (child rows + embeddings)
Loads parents.json   →  document_parents  (parent rows, plain text)

Prerequisites
─────────────
    pip install psycopg2-binary pgvector tqdm

Environment variables (or edit DB_CONFIG below)
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
from pathlib import Path

import psycopg
from tqdm import tqdm

from config import settings
from contextlib import contextmanager

from typing import Generator
# ── Config ────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     settings.db_host,
    "port":     settings.db_port,
    "dbname":   settings.app_db,
    "user":     settings.app_user,
    "password": settings.app_password.get_secret_value(),
}

CHILDREN_JSON = "data/children.json"
PARENTS_JSON  = "data/parents.json"

BATCH_SIZE = 200        
EMBEDDING_DIM = 768      


# ── DB helpers ────────────────────────────────────────────────────────────────
@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:
    """
    Establishes a connection to the database using a URL.
    The connection is automatically closed after the generator is exhausted.
    """
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            yield conn
    except psycopg.Error:
        raise


def setup_schema(conn):
    """Create pgvector extension + both tables (idempotent)."""
    with conn.cursor() as cur:
        # pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # ── Parents table ─────────────────────────────────────────────────────
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_parents (
                id   TEXT PRIMARY KEY,
                content     TEXT        NOT NULL,
                source_path TEXT,
                branch      TEXT
            );
        """)

        # ── Children table ────────────────────────────────────────────────────
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          TEXT PRIMARY KEY,
                parent_id   TEXT        NOT NULL
                                REFERENCES document_parents(id)
                                ON DELETE CASCADE,
                content     TEXT        NOT NULL,
                embedding   vector({EMBEDDING_DIM})
            );
        """)

        # HNSW index for fast ANN search on the child embeddings
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON document_chunks
            USING hnsw (embedding vector_ip_ops);
        """)

    conn.commit()
    print("✓  Schema ready (document_parents + document_chunks)")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_parents(conn, path: str):
    """
    Insert rows into document_parents.
    Skips duplicates via ON CONFLICT DO NOTHING.
    """
    raw = json.loads(Path(path).read_text())
    print(f"→  {len(raw):,} parent records found in {path}")

    rows = []
    for p in raw:
        rows.append((
            p["parent_id"],
            p["content"],
            p["metadata"]["source_path"],
            p["metadata"]["branch"],
        ))

    sql = """
                INSERT INTO document_parents (id, content, source_path, branch)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
    """
    inserted = 0
    with conn.cursor() as cur:
        for batch_start in tqdm(range(0, len(rows), BATCH_SIZE),
                                desc="  parents", unit="batch"):
            batch = rows[batch_start : batch_start + BATCH_SIZE]

            with conn.pipeline():
                cur.executemany(sql, batch)
            
            inserted += cur.rowcount

    conn.commit()
    print(f"✓  Parents inserted / updated: {inserted:,}  (skipped duplicates)")


def load_children(conn, path: str):
    """
    Insert rows into document_chunks.
    Embeddings are stored as pgvector vectors.
    Skips duplicates via ON CONFLICT DO NOTHING.
    """
    raw = json.loads(Path(path).read_text())
    print(f"→  {len(raw):,} child records found in {path}")

    # Validate embedding dimension on first record
    if raw:
        first_emb = raw[0].get("embedding", [])
        if first_emb and len(first_emb) != EMBEDDING_DIM:
            print(
                f"⚠  WARNING: embedding dim mismatch — "
                f"expected {EMBEDDING_DIM}, got {len(first_emb)}.\n"
                f"   Update EMBEDDING_DIM at the top of this script.",
                file=sys.stderr,
            )

    rows = [
        (
            c["id"],
            c["metadata"]["parent_id"],
            c["content"],
            c["embedding"],                           
        )
        for c in raw
    ]

    sql = """
        INSERT INTO document_chunks
            (id, parent_id, content, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """

    inserted = 0
    with conn.cursor() as cur:
        for batch_start in tqdm(range(0, len(rows), BATCH_SIZE),
                                desc="  children", unit="batch"):
            batch = rows[batch_start : batch_start + BATCH_SIZE]
            with conn.pipeline():
                cur.executemany(sql, batch)
            inserted += cur.rowcount
 

    conn.commit()
    print(f"✓  Children done — last batch rowcount: {inserted:,}")


# ── Sanity-check query ────────────────────────────────────────────────────────

 
def verify(conn: psycopg.Connection) -> None:
    n_parents  = conn.execute("SELECT COUNT(*) FROM document_parents;").fetchone()[0]
    n_children = conn.execute("SELECT COUNT(*) FROM document_chunks;").fetchone()[0]
    n_with_emb = conn.execute(
        "SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL;"
    ).fetchone()[0]
 
    print(
        f"\n── Verification ────────────────────────────────\n"
        f"   document_parents    : {n_parents:>8,} rows\n"
        f"   document_chunks     : {n_children:>8,} rows\n"
        f"   chunks w/ embeddings: {n_with_emb:>8,} rows\n"
        f"────────────────────────────────────────────────"
    )



# ── Entry point ───────────────────────────────────────────────────────────────
 
def main() -> None:
    print("Connecting to PostgreSQL …")
 
    # psycopg3 connections work as context managers — autoclose on exit
    with get_conn() as conn:
        try:
            setup_schema(conn)
            print()
 
            # Parents MUST be inserted first (FK constraint)
            load_parents(conn,  PARENTS_JSON)
            print()
            load_children(conn, CHILDREN_JSON)
            print()
 
            verify(conn)
            print(RETRIEVAL_EXAMPLE)
 
        except Exception as exc:
            print(f"\n✗  Error: {exc}", file=sys.stderr)
            raise
 
 
if __name__ == "__main__":
    main()