"""Normalize official Iowa invoice lines with an auditable outcome per row."""

from __future__ import annotations

import duckdb

REQUIRED = {
    "invoice_id",
    "ordered_on",
    "item_no",
    "category_code",
    "sales_bottles",
    "sales_liters",
    "sales_dollars",
}


def validate(con: duckdb.DuckDBPyConnection) -> None:
    """Validate quantities and deduplicate both line IDs and order IDs."""
    con.execute("""
        CREATE TEMP TABLE normalized AS
        WITH typed AS (
            SELECT *, try_cast(ordered_on AS DATE) AS day,
                try_cast(sales_liters AS DOUBLE) AS reported_liters,
                try_cast(sales_dollars AS DOUBLE) AS dollars,
                try_cast(sales_bottles AS DOUBLE) AS bottles,
                try_cast(bottle_volume_ml AS DOUBLE) * try_cast(sales_bottles AS DOUBLE)
                    / 1000 AS packaged_liters,
                coalesce(regexp_full_match(invoice_id, 'R?INV-[0-9]{6}'), false) AS order_id
            FROM raw
        ),
        corrected AS (
            SELECT *, CASE WHEN order_id AND day >= DATE '2025-11-02'
                       AND isfinite(packaged_liters) AND trunc(packaged_liters)=reported_liters
                       AND (sign(packaged_liters)=sign(bottles) OR bottles=0)
                    THEN packaged_liters ELSE reported_liters END AS liters
            FROM typed
        )
        SELECT *,
            CASE
                WHEN coalesce(trim(invoice_id), '')='' THEN 'missing invoice ID'
                WHEN order_id AND coalesce(trim(store_no), '')='' THEN 'missing store on order'
                WHEN coalesce(trim(item_no), '')='' THEN 'missing item'
                WHEN day IS NULL THEN 'invalid date'
                WHEN liters IS NULL OR NOT isfinite(liters) THEN 'invalid volume'
                WHEN bottles IS NULL OR NOT isfinite(bottles) OR bottles != trunc(bottles)
                     THEN 'invalid bottle count'
                WHEN sign(liters) != sign(bottles) THEN 'inconsistent signs'
                WHEN nullif(trim(sales_dollars), '') IS NOT NULL AND
                     (dollars IS NULL OR NOT isfinite(dollars)) THEN 'invalid amount'
            END AS invalid
        FROM corrected
    """)
    con.execute("""
        CREATE TEMP TABLE classified AS
        SELECT *, NULL::VARCHAR AS outcome FROM normalized WHERE false
    """)
    # Partition both identity schemes by invoice so copies always meet in one shard.
    # This keeps the sort and hash aggregation working sets small on statewide archives.
    count = con.execute("SELECT count(*) FROM normalized").fetchone()
    shards = max(1, ((count[0] if count else 0) + 249_999) // 250_000)
    query = """
        INSERT INTO classified
        WITH shard AS (
            SELECT * FROM normalized WHERE hash(invoice_id) % ? = ?
        ),
        identities AS (
            SELECT invoice_id, count(DISTINCT (ordered_on, item_no, category_code,
                sales_bottles, sales_liters, sales_dollars, store_no, bottle_volume_ml))>1
                AS conflict
            FROM shard WHERE NOT order_id GROUP BY invoice_id
        ),
        order_copies AS (
            SELECT *, row_number() OVER (
                PARTITION BY source_part, invoice_id, ordered_on, store_no, item_no, category_code,
                    bottle_volume_ml, sales_bottles, sales_liters, sales_dollars
            ) AS occurrence
            FROM shard WHERE order_id
        )
        SELECT n.*, CASE WHEN invalid IS NOT NULL THEN invalid
             WHEN conflict THEN 'conflicting invoice'
             WHEN row_number() OVER (PARTITION BY n.invoice_id ORDER BY source_part)>1
             THEN 'duplicate' ELSE 'accepted' END AS outcome
        FROM shard n LEFT JOIN identities i ON n.invoice_id=i.invoice_id
        WHERE NOT n.order_id
        UNION ALL
        SELECT * EXCLUDE (occurrence), CASE WHEN invalid IS NOT NULL THEN invalid
             WHEN row_number() OVER (
                PARTITION BY invoice_id, ordered_on, store_no, item_no, category_code,
                    bottle_volume_ml, sales_bottles, sales_liters, sales_dollars, occurrence
                ORDER BY source_part
             )>1 THEN 'duplicate' ELSE 'accepted' END AS outcome
        FROM order_copies
    """
    for shard in range(shards):
        con.execute(query, [shards, shard])
