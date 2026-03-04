"""
Step 1: Fetch Ethereum Transaction Data from Google BigQuery

DATA SOURCE: bigquery-public-data.crypto_ethereum.transactions
- Contains ALL Ethereum mainnet transactions since genesis
- Free tier: 1 TB/month query processing
- Each query below processes ~50-100 GB depending on date range

SETUP:
1. Create a Google Cloud project: https://console.cloud.google.com
2. Enable BigQuery API
3. Install: pip install google-cloud-bigquery google-cloud-bigquery-storage
4. Authenticate: gcloud auth application-default login
5. Set GCP_PROJECT_ID in config.py

OUTPUT:
- data/raw/eth_edges_2024.csv  (hourly edge lists for main sample)
- data/raw/eth_edges_covid.csv (hourly edge lists for COVID case study)

Each row = one (from_address, to_address) pair per hour, with:
  - hour_utc: timestamp truncated to hour
  - from_address, to_address
  - tx_count: number of transactions between the pair in that hour
  - total_eth: total ETH transferred between the pair in that hour
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from google.cloud import bigquery
import pandas as pd
import config


def get_bigquery_client():
    return bigquery.Client(project=config.GCP_PROJECT_ID)


def build_hourly_edgelist_query(start_date, end_date):
    """
    SQL query to extract hourly aggregated edge lists from Ethereum transactions.

    Logic:
    - Filter: value > 0 (actual ETH transfers), to_address IS NOT NULL (not contract creation)
    - Group by: hour, from_address, to_address
    - Aggregate: count of transactions, sum of ETH value
    - Filter out dust: only include edges with total_eth >= MIN_ETH_VALUE
    """
    query = f"""
    SELECT
        TIMESTAMP_TRUNC(block_timestamp, HOUR) AS hour_utc,
        from_address,
        to_address,
        COUNT(*) AS tx_count,
        SUM(CAST(value AS FLOAT64) / 1e18) AS total_eth
    FROM
        `{config.BQ_ETH_TABLE}`
    WHERE
        block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        AND to_address IS NOT NULL          -- exclude contract creation txs
        AND value > {config.MIN_ETH_VALUE_WEI}  -- exclude zero/dust transfers
        AND receipt_status = 1              -- only successful transactions
    GROUP BY
        hour_utc, from_address, to_address
    HAVING
        total_eth >= {config.MIN_ETH_VALUE}
    ORDER BY
        hour_utc, total_eth DESC
    """
    return query


def fetch_and_save(client, query, output_path, description=""):
    """Run BigQuery query and save results to CSV."""
    print(f"Running query: {description}")
    print(f"  Estimated scan: check BigQuery console for dry-run estimate")
    print(f"  Output: {output_path}")

    # Optional: dry run to check cost
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_run_job = client.query(query, job_config=job_config)
    gb_processed = dry_run_job.total_bytes_processed / (1024 ** 3)
    print(f"  Estimated data processed: {gb_processed:.1f} GB")

    proceed = input(f"  Proceed? (y/n): ")
    if proceed.lower() != "y":
        print("  Skipped.")
        return

    # Actual query
    df = client.query(query).to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} rows to {output_path}")
    return df


def main():
    client = get_bigquery_client()

    # ---- Main sample: 2024 full year ----
    query_2024 = build_hourly_edgelist_query(config.MAIN_START, config.MAIN_END)
    fetch_and_save(
        client, query_2024,
        config.RAW_DIR / "eth_edges_2024.csv",
        description="2024 full year ETH hourly edge lists"
    )

    # ---- COVID case study ----
    query_covid = build_hourly_edgelist_query(config.COVID_START, config.COVID_END)
    fetch_and_save(
        client, query_covid,
        config.RAW_DIR / "eth_edges_covid.csv",
        description="COVID period ETH hourly edge lists"
    )


# ============================================================
# ALTERNATIVE: If you prefer to run the SQL directly in BigQuery Console
# and download CSVs manually, use these queries:
# ============================================================

MANUAL_SQL_2024 = """
-- Run in BigQuery Console, then Export Results > CSV to Google Cloud Storage
-- Estimated: ~80-120 GB scan

SELECT
    TIMESTAMP_TRUNC(block_timestamp, HOUR) AS hour_utc,
    from_address,
    to_address,
    COUNT(*) AS tx_count,
    SUM(CAST(value AS FLOAT64) / 1e18) AS total_eth
FROM
    `bigquery-public-data.crypto_ethereum.transactions`
WHERE
    block_timestamp >= '2024-01-01'
    AND block_timestamp < '2025-01-01'
    AND to_address IS NOT NULL
    AND value > '1000000000000000'   -- > 0.001 ETH in Wei
    AND receipt_status = 1
GROUP BY
    hour_utc, from_address, to_address
HAVING
    total_eth >= 0.001
ORDER BY
    hour_utc, total_eth DESC
"""

MANUAL_SQL_COVID = """
-- Run in BigQuery Console for COVID period
-- Estimated: ~25-40 GB scan

SELECT
    TIMESTAMP_TRUNC(block_timestamp, HOUR) AS hour_utc,
    from_address,
    to_address,
    COUNT(*) AS tx_count,
    SUM(CAST(value AS FLOAT64) / 1e18) AS total_eth
FROM
    `bigquery-public-data.crypto_ethereum.transactions`
WHERE
    block_timestamp >= '2020-01-15'
    AND block_timestamp < '2020-05-01'
    AND to_address IS NOT NULL
    AND value > '1000000000000000'
    AND receipt_status = 1
GROUP BY
    hour_utc, from_address, to_address
HAVING
    total_eth >= 0.001
ORDER BY
    hour_utc, total_eth DESC
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Fetch Ethereum Transaction Data from BigQuery")
    print("=" * 60)
    print()
    print("Option A: Run via Python (requires gcloud auth)")
    print("Option B: Copy SQL from this file, run in BigQuery Console,")
    print("          export to GCS, download CSV to data/raw/")
    print()

    choice = input("Run via Python? (y/n): ")
    if choice.lower() == "y":
        main()
    else:
        print("\nSQL queries for manual execution:\n")
        print("--- 2024 Full Year ---")
        print(MANUAL_SQL_2024)
        print("\n--- COVID Period ---")
        print(MANUAL_SQL_COVID)
        print("\nSave outputs as:")
        print(f"  {config.RAW_DIR / 'eth_edges_2024.csv'}")
        print(f"  {config.RAW_DIR / 'eth_edges_covid.csv'}")
