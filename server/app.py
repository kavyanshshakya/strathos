"""
FastAPI server exposing the Strathos environment for HF Spaces.

Binds to 0.0.0.0:7860 (HF Spaces default port). Supports concurrent sessions
per TRL's requirement that max_concurrent_envs >= generation_batch_size.

The OpenEnv contract (/reset, /step, /health, /state) is mounted by
create_fastapi_app(). We additionally:
  - serve a showcase landing page at GET /
  - mount /static for the showcase's images and any auxiliary files

Run locally:
    python -m server.app

Or via uvicorn directly:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_fastapi_app

from .models import StrathosAction, StrathosObservation
from .robo_advisor import RoboAdvisorTaskEnv
from .wrapper import StrathosEnvironment


# Max concurrent sessions. Must be >= generation_batch_size for TRL GRPO.
MAX_CONCURRENT_ENVS = int(os.environ.get("STRATHOS_MAX_CONCURRENT", "64"))

# Path to the static assets (showcase HTML + figures).
STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


def _env_factory() -> StrathosEnvironment:
    """Factory called by OpenEnv for each new session."""
    task_env = RoboAdvisorTaskEnv()
    return StrathosEnvironment(task_env=task_env)


app = create_fastapi_app(
    env=_env_factory,
    action_cls=StrathosAction,
    observation_cls=StrathosObservation,
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)

# Mount /static so showcase HTML can reference images and other assets.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """Serve the Strathos showcase landing page at the Space root."""
    if INDEX_HTML.exists():
        return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))
    # Fallback if static file missing (defensive).
    return HTMLResponse(
        content=(
            "<h1>Strathos</h1>"
            "<p>OpenEnv RL environment. See "
            "<a href='https://github.com/kavyanshshakya/strathos'>GitHub</a> "
            "or POST to <code>/reset</code> and <code>/step</code>.</p>"
        ),
        status_code=200,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
