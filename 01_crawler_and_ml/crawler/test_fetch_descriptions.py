"""Offline regression checks: python -m unittest discover -s 01_crawler_and_ml/crawler."""

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError


SPEC = importlib.util.spec_from_file_location(
    "fetch_descriptions", Path(__file__).with_name("fetch_descriptions.py")
)
crawler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(crawler)


class Response(io.BytesIO):
    status = 200


def json_response(payload):
    return Response(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def immediate_limiter():
    # No network, real time delays, or worker races are needed by these tests.
    limiter = Mock()
    limiter.acquire.return_value = True
    limiter.deadline = None
    return limiter


def result(list_id, status="ok", description="Mô tả xe", attempts=1):
    return dict(
        list_id=list_id,
        status=status,
        description=description if status == "ok" else "",
        http_status=200 if status in ("ok", "empty", "unavailable") else 403,
        attempts=attempts,
        checked_at="2026-09-28T01:02:03+00:00",
        source_url=crawler.ENDPOINT + list_id,
        error="" if status == "ok" else status,
    )


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class FetchTests(unittest.TestCase):
    def test_unicode_newlines_and_whitespace_are_kept_without_truncation(self):
        body = '  Xe chính chủ, "không ngập nước".\r\n☎ Liên hệ 🚗\n' + "Đầy đủ lịch sử bảo dưỡng.\n" * 600
        opener = Mock(return_value=json_response({
            "ad": {"list_id": 123, "status": "active", "body": body}
        }))
        actual = crawler.fetch_one("123", immediate_limiter(), opener=opener)
        self.assertEqual(actual["status"], "ok")
        self.assertEqual(actual["description"], body)
        self.assertEqual(actual["attempts"], 1)
        self.assertEqual(actual["http_status"], 200)
        self.assertEqual(actual["source_url"], crawler.ENDPOINT + "123")

    def test_only_404_and_410_are_removed(self):
        for code, expected in ((404, "removed"), (410, "removed"), (403, "http_error"), (401, "http_error")):
            with self.subTest(code=code):
                opener = Mock(side_effect=HTTPError(crawler.ENDPOINT + "1", code, "error", {}, None))
                actual = crawler.fetch_one("1", immediate_limiter(), opener=opener)
                self.assertEqual(actual["status"], expected)
                self.assertEqual(actual["http_status"], code)
                self.assertEqual(actual["description"], "")
                self.assertEqual(opener.call_count, 1)

    def test_rate_limit_retries_and_pauses_without_becoming_removed(self):
        limiter = immediate_limiter()
        opener = Mock(side_effect=[
            HTTPError(crawler.ENDPOINT + "1", 429, "slow down", {"Retry-After": "90"}, None)
            for _ in range(3)
        ])
        actual = crawler.fetch_one("1", limiter, opener=opener)
        self.assertEqual(actual["status"], "rate_limited")
        self.assertEqual(actual["http_status"], 429)
        self.assertEqual(actual["attempts"], 3)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(limiter.pause.call_count, 3)
        for call in limiter.pause.call_args_list:
            self.assertEqual(call.args, (90,))

    def test_network_and_server_failures_remain_retryable(self):
        failures = (
            (URLError("connection reset"), "network_error"),
            (TimeoutError("timed out"), "network_error"),
            (HTTPError(crawler.ENDPOINT + "1", 503, "unavailable", {}, None), "http_error"),
        )
        for failure, expected in failures:
            with self.subTest(failure=type(failure).__name__, status=expected):
                opener = Mock(side_effect=failure)
                actual = crawler.fetch_one("1", immediate_limiter(), opener=opener)
                self.assertEqual(actual["status"], expected)
                self.assertEqual(actual["description"], "")
                self.assertEqual(actual["attempts"], 3)
                self.assertEqual(opener.call_count, 3)

    def test_malformed_json_is_invalid_response(self):
        actual = crawler.fetch_one(
            "1", immediate_limiter(), opener=Mock(return_value=Response(b"<html>maintenance</html>"))
        )
        self.assertEqual(actual["status"], "invalid_response")
        self.assertEqual(actual["description"], "")
        self.assertEqual(actual["http_status"], 200)

    def test_truncated_body_retries_as_network_error(self):
        class TruncatedResponse(Response):
            def read(self, *args, **kwargs):
                raise IncompleteRead(b'{"ad":', 200)

        responses = [TruncatedResponse() for _ in range(3)]
        opener = Mock(side_effect=responses)
        actual = crawler.fetch_one("1", immediate_limiter(), opener=opener)
        self.assertEqual(actual["status"], "network_error")
        self.assertEqual(actual["description"], "")
        self.assertEqual(actual["http_status"], 200)
        self.assertEqual(actual["attempts"], 3)
        self.assertEqual(opener.call_count, 3)
        self.assertIn("IncompleteRead", actual["error"])
        self.assertTrue(all(response.closed for response in responses))

    def test_other_listing_body_is_never_attached_to_requested_id(self):
        actual = crawler.fetch_one("1", immediate_limiter(), opener=Mock(return_value=json_response({
            "ad": {"list_id": 2, "status": "active", "body": "Description for another car"}
        })))
        self.assertEqual(actual["status"], "invalid_response")
        self.assertEqual(actual["description"], "")

    def test_empty_inactive_and_invalid_bodies_are_distinct(self):
        variants = (
            ({"status": "active", "body": None}, "empty"),
            ({"status": "active", "body": " \n\t"}, "empty"),
            ({"status": "active", "body": 123}, "invalid_response"),
            ({"status": "hidden", "body": "Old body"}, "unavailable"),
            ({"body": "Missing status"}, "invalid_response"),
        )
        for fields, expected in variants:
            with self.subTest(fields=fields):
                payload = {"ad": {"list_id": 1, "subject": "Title is not a description", **fields}}
                status, body, _ = crawler.parse_ad(payload, "1")
                self.assertEqual(status, expected)
                self.assertEqual(body, "")


class RetryAfterTests(unittest.TestCase):
    def test_seconds_and_http_date(self):
        now = datetime(2026, 9, 28, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(crawler.retry_after_seconds("90"), 90)
        self.assertEqual(crawler.retry_after_seconds("1.5"), 1.5)
        self.assertEqual(crawler.retry_after_seconds("-3"), 0)
        self.assertEqual(crawler.retry_after_seconds("Mon, 28 Sep 2026 00:02:00 GMT", now), 120)
        self.assertEqual(crawler.retry_after_seconds("Sun, 27 Sep 2026 23:59:00 GMT", now), 0)

    def test_unusable_values_have_safe_default(self):
        for value in (None, "invalid", "NaN", "Infinity"):
            with self.subTest(value=value):
                self.assertEqual(crawler.retry_after_seconds(value), 60)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.db = crawler.open_checkpoint(self.directory / "descriptions.sqlite3")
        self.addCleanup(self.db.close)

    def test_export_matches_full_input_universe_and_preserves_csv_body(self):
        body = 'Xe tốt, "chính chủ"\r\nTiếng Việt 🚗\n'
        statuses = ["ok", "removed", "empty", "unavailable", "http_error", "network_error", "invalid_response", "rate_limited"]
        for index, status in enumerate(statuses, start=1):
            crawler.store_result(self.db, result(str(index), status, description=body))
        # A checkpoint from a wider/older input must not pollute the current export.
        crawler.store_result(self.db, result("999"))
        ids = [str(value) for value in range(1, 10)]
        report = crawler.export_snapshot(self.db, ids, self.directory, {"stop_reason": "test"})
        descriptions = read_csv(self.directory / "descriptions.csv")
        audit = read_csv(self.directory / "description_status.csv")
        self.assertEqual(descriptions, [{"list_id": "1", "description": body}])
        self.assertEqual([row["list_id"] for row in audit], ids)
        self.assertEqual([row["status"] for row in audit], statuses + ["pending"])
        self.assertEqual(set(descriptions[0]), {"list_id", "description"})
        self.assertEqual(set(audit[0]), set(crawler.AUDIT_FIELDS))
        self.assertEqual({row["list_id"] for row in descriptions}, {row["list_id"] for row in audit if row["status"] == "ok"})
        self.assertEqual(report["total_ids"], 9)
        self.assertEqual(report["checked_ids"], 8)
        self.assertEqual(sum(report["counts"].values()), 9)
        self.assertEqual(report["unresolved_ids"], 5)
        self.assertEqual(report["description_coverage"], 1 / 9)
        self.assertFalse(report["all_ids_attempted"])
        self.assertEqual(json.loads((self.directory / "descriptions_report.json").read_text(encoding="utf-8")), report)

    def test_resume_skips_terminal_rows_and_retries_errors(self):
        statuses = ["ok", "removed", "empty", "unavailable", "http_error", "network_error", "invalid_response", "rate_limited"]
        for index, status in enumerate(statuses, start=1):
            crawler.store_result(self.db, result(str(index), status, attempts=2))
        input_path = self.directory / "cars.csv"
        input_path.write_text("list_id\n" + "\n".join(str(value) for value in range(1, 10)), encoding="utf-8")
        # main creates its own connection, like a resumed CLI process would.
        with patch.object(crawler, "fetch_one", side_effect=lambda list_id, *args: result(list_id)) as fetch, contextlib.redirect_stdout(io.StringIO()):
            exit_code = crawler.main(["--input", str(input_path), "--out", str(self.directory)])
        self.assertEqual(exit_code, 0)
        self.assertEqual([call.args[0] for call in fetch.call_args_list], ["5", "6", "7", "8", "9"])
        saved = crawler.checkpoint_rows(self.db)
        self.assertEqual(saved["1"]["attempts"], 2)
        self.assertEqual(saved["5"]["attempts"], 3)
        self.assertEqual(saved["9"]["attempts"], 1)
        audit = read_csv(self.directory / "description_status.csv")
        self.assertEqual(len(audit), 9)
        self.assertTrue(all(row["status"] in crawler.TERMINAL for row in audit))

    def test_stopped_pending_result_does_not_erase_previous_error(self):
        crawler.store_result(self.db, result("1", "network_error", attempts=3))
        crawler.store_result(self.db, result("1", "pending", attempts=0))
        self.assertEqual(crawler.checkpoint_rows(self.db)["1"]["status"], "network_error")
        self.assertEqual(crawler.checkpoint_rows(self.db)["1"]["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
