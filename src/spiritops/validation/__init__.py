"""Normalize official Iowa invoice lines with an auditable outcome per row."""

from __future__ import annotations

import duckdb

REQUIRED = {
    "invoice_id", "ordered_on", "item_no", "category_code", "sales_bottles",
    "sales_liters", "sales_dollars",
}


def validate(con: duckdb.DuckDBPyConnection) -> None:
    """Disambiguate invoice IDs across files. Conflicting IDs go entirely to quarantine."""
    con.execute("""
        CREATE TEMP TABLE normalized AS
        SELECT *, try_cast(ordered_on AS DATE) AS day,
            try_cast(sales_liters AS DOUBLE) AS liters,
            try_cast(sales_dollars AS DOUBLE) AS dollars,
            try_cast(sales_bottles AS DOUBLE) AS bottles,
            CASE
                WHEN coalesce(trim(invoice_id), '')='' THEN 'missing invoice ID'
                WHEN coalesce(trim(item_no), '')='' THEN 'missing item'
                WHEN try_cast(ordered_on AS DATE) IS NULL THEN 'invalid date'
                WHEN try_cast(sales_liters AS DOUBLE) IS NULL OR
                     NOT isfinite(try_cast(sales_liters AS DOUBLE)) THEN 'invalid volume'
                WHEN try_cast(sales_bottles AS DOUBLE) IS NULL OR
                     NOT isfinite(try_cast(sales_bottles AS DOUBLE)) OR
                     try_cast(sales_bottles AS DOUBLE) != trunc(try_cast(sales_bottles AS DOUBLE))
                     THEN 'invalid bottle count'
                WHEN sign(try_cast(sales_liters AS DOUBLE)) !=
                     sign(try_cast(sales_bottles AS DOUBLE)) THEN 'inconsistent signs'
                WHEN nullif(trim(sales_dollars), '') IS NOT NULL AND
                     (try_cast(sales_dollars AS DOUBLE) IS NULL OR
                      NOT isfinite(try_cast(sales_dollars AS DOUBLE))) THEN 'invalid amount'
            END AS invalid
        FROM raw
    """)
    con.execute("""
        CREATE TEMP TABLE classified AS
        WITH identities AS (
            SELECT invoice_id, count(DISTINCT (ordered_on, item_no, category_code,
                sales_bottles, sales_liters, sales_dollars))>1 AS conflict
            FROM normalized GROUP BY invoice_id
        )
        SELECT n.*, CASE WHEN invalid IS NOT NULL THEN invalid
             WHEN conflict THEN 'conflicting invoice'
             WHEN row_number() OVER (PARTITION BY n.invoice_id ORDER BY n.invoice_id)>1
             THEN 'duplicate' ELSE 'accepted' END AS outcome
        FROM normalized n LEFT JOIN identities i ON n.invoice_id=i.invoice_id
    """)
