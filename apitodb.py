import requests
import json
import logging
from typing import Callable, Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -------------------- CONFIGURATION --------------------
DATABASE_URI = "sqlite:///temp.db"  # Replace with PostgreSQL/MySQL URI in prod
TEMP_TABLE_NAME = "temp_data"
FLAG_TABLE_NAME = "processing_flags"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- CORE ENGINE SETUP --------------------
def get_engine(uri: str = DATABASE_URI) -> Engine:
    return create_engine(uri, future=True)

# -------------------- API FETCHER --------------------
def fetch_json(api_url: str, headers: dict = None) -> List[dict]:
    logger.info(f"Fetching data from API: {api_url}")
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        data = [data]
    return data

# -------------------- TEMP TABLE LOADER --------------------
def load_json_to_temp_table(engine: Engine, data: List[dict]) -> None:
    logger.info("Loading JSON data into temporary table.")
    with engine.begin() as conn:
        # Create temp table
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TEMP_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data JSON
            )
        """))

        # Insert JSON rows
        for row in data:
            conn.execute(
                text(f"INSERT INTO {TEMP_TABLE_NAME} (data) VALUES (:data)"),
                {"data": json.dumps(row)}
            )

# -------------------- DATA PROCESSOR --------------------
def process_temp_data(engine: Engine, process_fn: Callable[[Any], None]) -> None:
    logger.info("Processing temporary data.")
    with engine.begin() as conn:
        results = conn.execute(text(f"SELECT id, data FROM {TEMP_TABLE_NAME}"))
        for row in results:
            record = json.loads(row["data"])
            process_fn(record)

        # Set completion flag
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {FLAG_TABLE_NAME} (done BOOLEAN)
        """))
        conn.execute(text(f"INSERT INTO {FLAG_TABLE_NAME} (done) VALUES (1)"))

# -------------------- CLEANUP --------------------
def clean_up(engine: Engine) -> None:
    logger.info("Checking cleanup flag.")
    with engine.begin() as conn:
        result = conn.execute(text(f"""
            SELECT done FROM {FLAG_TABLE_NAME} ORDER BY ROWID DESC LIMIT 1
        """)).scalar()

        if result:
            logger.info("Flag detected. Cleaning up temporary data.")
            conn.execute(text(f"DROP TABLE IF EXISTS {TEMP_TABLE_NAME}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {FLAG_TABLE_NAME}"))
        else:
            logger.info("Processing not complete. Cleanup skipped.")

# -------------------- CONTROLLER --------------------
def run_pipeline(
    api_url: str,
    headers: dict = None,
    engine_uri: str = DATABASE_URI,
    process_fn: Callable[[dict], None] = lambda record: print("Processed:", record)
) -> None:
    try:
        engine = get_engine(engine_uri)
        data = fetch_json(api_url, headers)
        load_json_to_temp_table(engine, data)
        process_temp_data(engine, process_fn)
        clean_up(engine)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")