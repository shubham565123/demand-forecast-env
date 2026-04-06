"""
FastAPI application for the Demand Forecast Adjuster Environment.

Endpoints:
    - POST /reset: Reset the environment with a new scenario
    - POST /step: Submit an adjusted forecast for grading
    - GET /state: Get current environment state
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from models import ForecastAction, ForecastObservation
except ImportError:
    from ..models import ForecastAction, ForecastObservation

from .environment import DemandForecastEnvironment


app = create_app(
    DemandForecastEnvironment,
    ForecastAction,
    ForecastObservation,
    env_name="demand_forecast",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
