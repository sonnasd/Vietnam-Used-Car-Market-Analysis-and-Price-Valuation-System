import argparse
import csv
import json
import math
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://gateway.chotot.com/v1/public/ad-listing/"
TERMINAL = {"ok", "removed", "empty", "unavailable"}
STATUSES = ("pending", "ok", "removed", "empty", "unavailable", "http_error",
            "network_error", "invalid_response", "rate_limited")
AUDIT_FIELDS = ("list_id", "status", "http_status", "attempts", "checked_at",
                "source_url", "error")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_ids(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "list_id" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain list_id")
        ids = [row["list_id"].strip() for row in reader]
    if not ids or any(not value.isascii() or not value.isdigit() or int(value) < 1 for value in ids):
        raise ValueError("Input must contain positive integer list_ids")
    ids = [str(int(value)) for value in ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate list_ids in input; resolve before collecting")
    return ids


def retry_after_seconds(value, now=None):
    try:
        seconds = float(value)
        if math.isfinite(seconds):
            return max(0, seconds)
    except (TypeError, ValueError):
        pass
    try:
        deadline = parsedate_to_datetime(value)
        return max(0, (deadline - (now or datetime.now(timezone.utc))).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 60


class RateLimiter:
    """One request-start interval and one Retry-After pause across all workers."""

    def __init__(self, delay, deadline=None):
        self.delay = delay
        self.deadline = deadline
        self.next_start = 0.0
        self.blocked_until = 0.0
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def acquire(self):
        while not self.stopped.is_set():
            with self.lock:
                now = time.monotonic()
                if self.deadline is not None and now >= self.deadline:
                    return False
                remaining = max(self.next_start, self.blocked_until) - now
                if remaining <= 0:
                    self.next_start = now + self.delay
                    return True
            self.stopped.wait(min(remaining, 0.5))
        return False

    def pause(self, seconds):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


def parse_ad(payload, list_id):
    ad = payload.get("ad") if isinstance(payload, dict) else None
    if not isinstance(ad, dict) or str(ad.get("list_id")) != list_id:
        return "invalid_response", "", "Missing ad or mismatched list_id"
    status = ad.get("status")
    if not isinstance(status, str) or not status:
        return "invalid_response", "", "Missing listing status"
    if status != "active":
        return "unavailable", "", f"Listing status: {status} (cause not inferred)"
    body = ad.get("body")
    if body is None or (isinstance(body, str) and not body.strip()):
        return "empty", "", "Active detail has no nonblank body"
    if not isinstance(body, str):
        return "invalid_response", "", "body is not text"
    return "ok", body, ""


def fetch_one(list_id, limiter, timeout=20, max_attempts=3, opener=urlopen):
    url = ENDPOINT + list_id
    result = dict(list_id=list_id, status="pending", description="", http_status="",
                  attempts=0, checked_at="", source_url=url, error="")
    for attempt in range(1, max_attempts + 1):
        if not limiter.acquire():
            break
        result.update(attempts=attempt, checked_at=utc_now(), http_status="")
        backoff = min(30, 2 ** attempt)
        try:
            request = Request(url, headers={"User-Agent": "UsedCarResearch/1.0",
                                            "Accept": "application/json"})
            with opener(request, timeout=timeout) as response:
                result["http_status"] = response.status
                payload = json.load(response)
            status, body, error = parse_ad(payload, list_id)
            result.update(status=status, description=body, error=error)
            return result
        except HTTPError as exc:
            result.update(http_status=exc.code, error=f"HTTP {exc.code}")
            if exc.code in (404, 410):
                result["status"] = "removed"
                exc.close()
                return result
            if exc.code == 429:
                result["status"] = "rate_limited"
                backoff = max(60, retry_after_seconds(exc.headers.get("Retry-After")))
                limiter.pause(backoff)
            else:
                result["status"] = "http_error"
            exc.close()
            if exc.code != 429 and not 500 <= exc.code < 600:
                return result
        except (URLError, TimeoutError, OSError, HTTPException) as exc:
            result.update(status="network_error", error=f"{type(exc).__name__}: {exc}"[:500])
        except (ValueError, UnicodeError) as exc:
            result.update(status="invalid_response", error=f"Invalid JSON: {exc}"[:500])
            return result
        if attempt < max_attempts:
            # A 429 pause is shared by every worker; other transient retries are local.
            if result["status"] != "rate_limited":
                remaining = backoff
                if limiter.deadline is not None:
                    remaining = min(remaining, max(0, limiter.deadline - time.monotonic()))
                limiter.stopped.wait(remaining)
    return result


def open_checkpoint(path):
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS descriptions (
        list_id TEXT PRIMARY KEY, status TEXT NOT NULL, description TEXT NOT NULL,
        http_status TEXT NOT NULL, attempts INTEGER NOT NULL,
        checked_at TEXT NOT NULL, source_url TEXT NOT NULL, error TEXT NOT NULL)""")
    return db


def store_result(db, row):
    if row["status"] == "pending":
        return
    with db:
        db.execute("""INSERT INTO descriptions VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(list_id) DO UPDATE SET status=excluded.status,
            description=excluded.description, http_status=excluded.http_status,
            attempts=descriptions.attempts+excluded.attempts,
            checked_at=excluded.checked_at, source_url=excluded.source_url, error=excluded.error""",
                   tuple(row[key] for key in ("list_id", "status", "description", "http_status",
                                               "attempts", "checked_at", "source_url", "error")))


def checkpoint_rows(db):
    fields = ("list_id", "status", "description", "http_status", "attempts",
              "checked_at", "source_url", "error")
    return {row[0]: dict(zip(fields, row)) for row in db.execute("SELECT * FROM descriptions")}


def atomic_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_snapshot(db, ids, out, run):
    saved = checkpoint_rows(db)
    rows = [saved.get(value, dict(list_id=value, status="pending", description="",
                                http_status="", attempts=0, checked_at="",
                                source_url=ENDPOINT + value, error="")) for value in ids]
    atomic_csv(out / "descriptions.csv", ("list_id", "description"),
               (row for row in rows if row["status"] == "ok"))
    atomic_csv(out / "description_status.csv", AUDIT_FIELDS, rows)
    counts = Counter(row["status"] for row in rows)
    checked_dates = [row["checked_at"] for row in rows if row["checked_at"]]
    report = dict(total_ids=len(ids), checked_ids=len(ids) - counts["pending"],
                  counts={status: counts[status] for status in STATUSES},
                  description_coverage=counts["ok"] / len(ids),
                  all_ids_attempted=counts["pending"] == 0,
                  unresolved_ids=sum(counts[s] for s in STATUSES if s not in TERMINAL),
                  first_checked_at=min(checked_dates, default=None),
                  last_checked_at=max(checked_dates, default=None),
                  exported_at=utc_now(), endpoint=ENDPOINT + "{list_id}", run=run,
                  limitations=[
                      "removed means detail endpoint returned HTTP 404/410 at checked_at; reason unknown.",
                      "Access failures and pending IDs are not counted as removed.",
                      "Bodies are collected later than cars.csv; listing content/status may have changed.",
                      "Missing descriptions remain missing; use subject fallback explicitly in notebook 04."])
    path = out / "descriptions_report.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/cars.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--delay", type=float, default=1, help="Global seconds between request starts (>=0.25)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent requests (1..4); global delay still applies")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, help="Optional time budget; omitted means process all IDs")
    parser.add_argument("--limit", type=int, help="Optional number of IDs to process this run")
    parser.add_argument("--refresh", action="store_true", help="Recheck terminal results too")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.delay) or args.delay < 0.25 or not 1 <= args.workers <= 4
            or not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_attempts < 1
            or (args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0))
            or (args.limit is not None and args.limit < 1)):
        parser.error("Invalid delay, workers, timeout, max-attempts, max-seconds or limit")
    ids = read_ids(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    db = open_checkpoint(args.out / "descriptions.sqlite3")
    saved = checkpoint_rows(db)
    todo = [value for value in ids if args.refresh or saved.get(value, {}).get("status") not in TERMINAL]
    if args.limit is not None:
        todo = todo[:args.limit]
    started = time.monotonic()
    limiter = RateLimiter(args.delay, started + args.max_seconds if args.max_seconds else None)
    run = dict(started_at=utc_now(), input=str(args.input.resolve()), delay=args.delay,
               workers=args.workers, timeout=args.timeout, max_attempts=args.max_attempts,
               max_seconds=args.max_seconds, limit=args.limit, refresh=args.refresh,
               processed_this_run=0, stop_reason="running")
    iterator = iter(todo)
    pending = {}
    failures = 0
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def submit_next():
        if limiter.stopped.is_set():
            return
        value = next(iterator, None)
        if value is not None:
            pending[pool.submit(fetch_one, value, limiter, args.timeout, args.max_attempts)] = value

    print(f"Input: {len(ids)} IDs; scheduled: {len(todo)}; checkpoint: {len(saved)}", flush=True)
    export_snapshot(db, ids, args.out, run)
    try:
        for _ in range(args.workers):
            submit_next()
        while pending:
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            if limiter.deadline is not None and time.monotonic() >= limiter.deadline:
                run["stop_reason"] = "time_budget"
                limiter.stopped.set()
            for future in done:
                pending.pop(future)
                result = future.result()
                store_result(db, result)
                if result["attempts"]:
                    run["processed_this_run"] += 1
                    failures = 0 if result["status"] in TERMINAL else failures + 1
                    if failures >= 10:
                        run["stop_reason"] = "consecutive_errors"
                        limiter.stopped.set()
                    if run["processed_this_run"] % 100 == 0:
                        report = export_snapshot(db, ids, args.out, run)
                        print(json.dumps(dict(processed=run["processed_this_run"],
                                              elapsed_seconds=round(time.monotonic() - started),
                                              counts=report["counts"])), flush=True)
                submit_next()
        if run["stop_reason"] == "running":
            run["stop_reason"] = "limit" if args.limit and len(todo) == args.limit else "completed"
    except KeyboardInterrupt:
        run["stop_reason"] = "interrupted"
        limiter.stopped.set()
    except Exception:
        run["stop_reason"] = "unexpected_error"
        limiter.stopped.set()
        raise
    finally:
        limiter.stopped.set()
        pool.shutdown(wait=True, cancel_futures=True)
        # Save requests that finished while stopping, without counting cancelled tasks.
        for future in pending:
            if future.done() and not future.cancelled() and future.exception() is None:
                result = future.result()
                store_result(db, result)
                run["processed_this_run"] += bool(result["attempts"])
        run.update(ended_at=utc_now(), elapsed_seconds=round(time.monotonic() - started, 2))
        report = export_snapshot(db, ids, args.out, run)
        db.close()
        print(json.dumps(report, ensure_ascii=True), flush=True)
    return 0 if report["unresolved_ids"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
