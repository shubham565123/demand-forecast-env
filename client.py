"""Demand Forecast Adjuster Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import ForecastAction, ForecastObservation
except ImportError:
    from .models import ForecastAction, ForecastObservation


class DemandForecastEnv(EnvClient[ForecastAction, ForecastObservation, State]):
    """
    Client for the Demand Forecast Adjuster Environment.

    Example:
        >>> from demand_forecast_env import DemandForecastEnv, ForecastAction
        >>>
        >>> with DemandForecastEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(difficulty="easy", seed=42)
        ...     print(f"Baseline: {result.observation.baseline_forecast}")
        ...     result = env.step(ForecastAction(adjusted_forecast=13000))
        ...     print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: ForecastAction) -> Dict:
        return {
            "adjusted_forecast": action.adjusted_forecast,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ForecastObservation]:
        obs_data = payload.get("observation", {})
        observation = ForecastObservation(
            product_name=obs_data.get("product_name"),
            product_category=obs_data.get("product_category"),
            baseline_forecast=obs_data.get("baseline_forecast"),
            time_period=obs_data.get("time_period"),
            signals=obs_data.get("signals"),
            difficulty=obs_data.get("difficulty"),
            expected_forecast=obs_data.get("expected_forecast"),
            direction_score=obs_data.get("direction_score"),
            magnitude_score=obs_data.get("magnitude_score"),
            coverage_score=obs_data.get("coverage_score"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
