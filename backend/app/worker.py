"""Ingestion worker: python -m app.worker. Polls ingestion_jobs and processes one job at a time."""

import logging
import signal
import threading

import psycopg

from app.config import get_settings
from app.db import check_db, open_pool
from app.ingest import claim_job, process_job, reset_stale

log = logging.getLogger("app.worker")

POLL_SECONDS = 2.0
DB_RETRY_SECONDS = 5.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    pool = open_pool(settings)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())  # finish the running job, then exit
    with pool.connection() as conn:
        check_db(conn, settings.embedding_dim)  # same refusal as the API: no superuser, RLS on, dimension matches
        reset = reset_stale(conn)
    log.info("worker started (%d stale job(s) reset)", reset)
    while not stop.is_set():
        try:
            with pool.connection() as conn:
                job = claim_job(conn)
            if job is None:
                stop.wait(POLL_SECONDS)
                continue
            log.info("job %s: document %s (attempt %d)", job["id"], job["document_id"], job["attempts"])
            process_job(pool, job, settings)
        except psycopg.OperationalError:
            log.exception("database unavailable; retrying")
            stop.wait(DB_RETRY_SECONDS)
    pool.close()
    log.info("worker stopped")


if __name__ == "__main__":
    main()
