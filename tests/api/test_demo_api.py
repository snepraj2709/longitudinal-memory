from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.app import create_app


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "results/demo/demo-v1/bundle.json"


class DemoApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        web = Path(self.temp.name)
        (web / "index.html").write_text("<html>demo</html>", encoding="utf-8")
        self.client = TestClient(create_app(bundle_path=BUNDLE, web_root=web))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_health_is_artifact_replay(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "mode": "artifact_replay"})
        self.assertEqual(response.headers["x-robots-tag"], "noindex, nofollow, noarchive")

    def test_case_endpoints(self) -> None:
        listing = self.client.get("/api/demo/cases")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(len(listing.json()), 4)
        detail = self.client.get("/api/demo/cases/temporal_003")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["answer"]["body"], "May 18, 2026")
        self.assertEqual(self.client.get("/api/demo/cases/missing").status_code, 404)

    def test_run_and_scorecard_endpoints(self) -> None:
        runs = self.client.get("/api/runs")
        self.assertEqual([item["series_id"] for item in runs.json()], [
            "openai-gpt41-v1", "qwen35-27b-fp8-v1",
        ])
        scorecards = self.client.get("/api/scorecards")
        self.assertEqual([item["baseline_id"] for item in scorecards.json()], [
            "B1", "B2", "B3", "B4", "B5", "B6", "B7",
        ])

    def test_write_methods_are_rejected(self) -> None:
        for method in ("post", "put", "patch", "delete"):
            response = getattr(self.client, method)("/api/demo/cases")
            self.assertEqual(response.status_code, 405, method)

    def test_private_paths_and_docs_are_not_routes(self) -> None:
        for path in ("/api/gold", "/api/oracle", "/api/review", "/api/secrets", "/openapi.json"):
            self.assertEqual(self.client.get(path).status_code, 404, path)


if __name__ == "__main__":
    unittest.main()
