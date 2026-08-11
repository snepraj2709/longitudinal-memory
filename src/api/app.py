"""Read-only demo API and compiled React host."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "results/demo/demo-v1/bundle.json"
DEFAULT_WEB = ROOT / "web/dist"


class DemoStore:
    def __init__(self, bundle_path: Path) -> None:
        value = json.loads(bundle_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != "demo_bundle_v1":
            raise ValueError("demo bundle has an unsupported schema")
        self.bundle = value
        self.cases = {str(item["case_id"]): item for item in value["cases"]}

    def case_summaries(self) -> list[dict[str, Any]]:
        fields = ("case_id", "title", "summary", "capability", "difficulty", "as_of")
        return [{field: item[field] for field in fields} for item in self.bundle["cases"]]


def create_app(*, bundle_path: Path | None = None, web_root: Path | None = None) -> FastAPI:
    selected_bundle = bundle_path or Path(os.environ.get("DEMO_BUNDLE", DEFAULT_BUNDLE))
    selected_web = web_root or Path(os.environ.get("DEMO_WEB", DEFAULT_WEB))
    store = DemoStore(selected_bundle)
    app = FastAPI(
        title="Longitudinal Memory Demo",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def demo_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "artifact_replay"}

    @app.get("/api/demo/cases")
    def list_cases() -> list[dict[str, Any]]:
        return store.case_summaries()

    @app.get("/api/demo/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        if case_id not in store.cases:
            raise HTTPException(status_code=404, detail="case not found")
        return store.cases[case_id]

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return store.bundle["runs"]

    @app.get("/api/scorecards")
    def list_scorecards() -> list[dict[str, Any]]:
        return store.bundle["scorecards"]

    assets = selected_web / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):  # type: ignore[no-untyped-def]
        if path.startswith("api/") or path in {"docs", "redoc", "openapi.json"}:
            return JSONResponse(status_code=404, content={"detail": "not found"})
        index = selected_web / "index.html"
        if not index.is_file():
            return JSONResponse(
                status_code=503,
                content={"detail": "frontend is not built; run npm run build in web"},
            )
        return FileResponse(index)

    return app


app = create_app()
