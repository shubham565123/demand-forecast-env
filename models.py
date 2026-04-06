"""
Data models for the Demand Forecast Adjuster Environment.

The LLM receives a baseline demand forecast along with contextual signals
(promotions, festive seasons, competitor activity, etc.) and must output
an adjusted forecast. The environment grades the adjustment on three
dimensions: directional correctness, magnitude accuracy, and signal coverage.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ForecastAction(Action):
    """Action for the Demand Forecast Adjuster — the agent's adjusted forecast."""

    adjusted_forecast: float = Field(
        ..., description="The agent's adjusted demand forecast (in units)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional: the agent's reasoning for the adjustment",
    )


class ForecastObservation(Observation):
    """Observation from the Demand Forecast Adjuster environment."""

    # --- Provided on reset (the scenario) ---
    product_name: Optional[str] = Field(
        default=None, description="Name of the product being forecast"
    )
    product_category: Optional[str] = Field(
        default=None, description="Category (e.g., shampoo, detergent)"
    )
    baseline_forecast: Optional[float] = Field(
        default=None, description="Statistical baseline forecast in units"
    )
    time_period: Optional[str] = Field(
        default=None, description="Forecast period (e.g., 'November 2025')"
    )
    signals: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Context signals, each with 'signal' name and 'description'",
    )
    difficulty: Optional[str] = Field(
        default=None, description="Task difficulty: easy, medium, or hard"
    )

    # --- Provided on step (the grading result) ---
    expected_forecast: Optional[float] = Field(
        default=None,
        description="The ground-truth adjusted forecast (revealed after step)",
    )
    direction_score: Optional[float] = Field(
        default=None, description="Score for directional correctness (0.0 or 1.0)"
    )
    magnitude_score: Optional[float] = Field(
        default=None, description="Score for magnitude accuracy (0.0 to 1.0)"
    )
    coverage_score: Optional[float] = Field(
        default=None,
        description="Score for signal coverage (0.0 to 1.0)",
    )
