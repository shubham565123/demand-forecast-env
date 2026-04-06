"""
Demand Forecast Adjuster Environment.

A single-step RL environment where the agent receives a demand forecast
scenario with contextual signals and must output an adjusted forecast.

Three difficulty levels:
  - easy:   1 signal, 3 products, straightforward adjustment
  - medium: 2-3 signals (some conflicting), 5 products
  - hard:   4+ signals with interaction effects, 7 products
"""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import ForecastAction, ForecastObservation
except ImportError:
    from ..models import ForecastAction, ForecastObservation

try:
    from server.grader import grade_forecast
except ImportError:
    from .grader import grade_forecast


# ── Product catalog ──────────────────────────────────────────────────────────

PRODUCTS = [
    {"name": "UltraClean Shampoo", "category": "Hair Care", "base_range": (8000, 15000)},
    {"name": "FreshGlow Face Wash", "category": "Skin Care", "base_range": (5000, 12000)},
    {"name": "PowerWhite Detergent", "category": "Home Care", "base_range": (20000, 40000)},
    {"name": "SilkSoft Conditioner", "category": "Hair Care", "base_range": (6000, 10000)},
    {"name": "PureShield Hand Wash", "category": "Hygiene", "base_range": (10000, 25000)},
    {"name": "GlowPlus Body Lotion", "category": "Skin Care", "base_range": (4000, 9000)},
    {"name": "SparkleClean Dishwash", "category": "Home Care", "base_range": (15000, 30000)},
]

# ── Signal templates ─────────────────────────────────────────────────────────
# Each signal has a name, a description template, and an impact range (%).

POSITIVE_SIGNALS = [
    {
        "name": "festive_season",
        "description": "Diwali festive season is approaching — historically drives {pct}% increase in {category} sales",
        "impact_range": (15, 40),
    },
    {
        "name": "promotion_bogo",
        "description": "Buy-One-Get-One promotion planned for this product — expected {pct}% uplift",
        "impact_range": (20, 50),
    },
    {
        "name": "summer_peak",
        "description": "Peak summer season ahead — {category} demand typically rises by {pct}%",
        "impact_range": (10, 30),
    },
    {
        "name": "social_media_trend",
        "description": "Product went viral on social media last week — expected {pct}% boost in short-term demand",
        "impact_range": (15, 35),
    },
    {
        "name": "new_distribution",
        "description": "Expanding distribution to 500 new retail stores — projected {pct}% increase in reach",
        "impact_range": (10, 25),
    },
    {
        "name": "wedding_season",
        "description": "Wedding season starting — {category} gifting demand expected to rise by {pct}%",
        "impact_range": (10, 25),
    },
]

NEGATIVE_SIGNALS = [
    {
        "name": "competitor_launch",
        "description": "Major competitor launched a cheaper alternative — expected {pct}% drop in market share",
        "impact_range": (-25, -5),
    },
    {
        "name": "price_increase",
        "description": "Recent {pct_abs}% price increase due to raw material costs — demand elasticity expected",
        "impact_range": (-20, -5),
    },
    {
        "name": "monsoon_disruption",
        "description": "Heavy monsoon predicted — supply chain and store footfall expected to drop by {pct_abs}%",
        "impact_range": (-20, -10),
    },
    {
        "name": "negative_publicity",
        "description": "Negative media report about product ingredient — estimated {pct_abs}% customer churn",
        "impact_range": (-15, -5),
    },
    {
        "name": "economic_slowdown",
        "description": "Consumer spending declining in the region — discretionary {category} purchases down {pct_abs}%",
        "impact_range": (-15, -5),
    },
]

TIME_PERIODS = [
    "October 2025", "November 2025", "December 2025",
    "January 2026", "February 2026", "March 2026",
    "April 2026", "May 2026", "June 2026",
]


def _make_signal(template: dict, category: str, rng: random.Random) -> dict:
    """Instantiate a signal template with a concrete impact percentage."""
    lo, hi = template["impact_range"]
    pct = rng.randint(min(lo, hi), max(lo, hi))
    description = template["description"].format(
        pct=abs(pct), pct_abs=abs(pct), category=category
    )
    return {
        "signal": template["name"],
        "description": description,
        "impact_pct": pct,
    }


def generate_scenario(
    difficulty: str, rng: random.Random
) -> dict:
    """
    Generate a forecast scenario for the given difficulty level.

    Returns a dict with product info, baseline, signals, and time period.
    """
    product = rng.choice(PRODUCTS)
    lo, hi = product["base_range"]
    baseline = round(rng.uniform(lo, hi) / 100) * 100  # round to nearest 100
    time_period = rng.choice(TIME_PERIODS)

    if difficulty == "easy":
        # 1 signal, clear direction
        pool = POSITIVE_SIGNALS + NEGATIVE_SIGNALS
        template = rng.choice(pool)
        signals = [_make_signal(template, product["category"], rng)]

    elif difficulty == "medium":
        # 2-3 signals, same direction (all positive OR all negative)
        # Challenge is computing the combined magnitude, not the direction
        n_signals = rng.randint(2, 3)
        go_positive = rng.choice([True, False])

        if go_positive:
            chosen = rng.sample(POSITIVE_SIGNALS, min(n_signals, len(POSITIVE_SIGNALS)))
        else:
            chosen = rng.sample(NEGATIVE_SIGNALS, min(n_signals, len(NEGATIVE_SIGNALS)))

        signals = []
        for t in chosen:
            signals.append(_make_signal(t, product["category"], rng))
        rng.shuffle(signals)

    elif difficulty == "hard":
        # 4-5 signals with interaction effects
        n_signals = rng.randint(4, 5)
        pos_count = rng.randint(2, n_signals - 1)
        neg_count = n_signals - pos_count

        chosen_pos = rng.sample(POSITIVE_SIGNALS, min(pos_count, len(POSITIVE_SIGNALS)))
        chosen_neg = rng.sample(NEGATIVE_SIGNALS, min(neg_count, len(NEGATIVE_SIGNALS)))

        signals = []
        for t in chosen_pos:
            signals.append(_make_signal(t, product["category"], rng))
        for t in chosen_neg:
            signals.append(_make_signal(t, product["category"], rng))
        rng.shuffle(signals)

    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    return {
        "product_name": product["name"],
        "product_category": product["category"],
        "baseline_forecast": baseline,
        "time_period": time_period,
        "signals": signals,
        "difficulty": difficulty,
    }


class DemandForecastEnvironment(Environment):
    """
    Demand Forecast Adjuster environment.

    Single-step episodes: reset() provides a scenario, step() grades the answer.
    Supports 3 difficulty levels: easy, medium, hard.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_scenario = None
        self._rng = random.Random(42)
        self._difficulty = "easy"

    def reset(
        self,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> ForecastObservation:
        """
        Reset and generate a new forecast scenario.

        Args:
            difficulty: "easy", "medium", or "hard" (default: "easy")
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier
        """
        if difficulty is not None:
            if difficulty not in ("easy", "medium", "hard"):
                raise ValueError(
                    f"difficulty must be 'easy', 'medium', or 'hard', got '{difficulty}'"
                )
            self._difficulty = difficulty

        if seed is not None:
            self._rng = random.Random(seed)

        self._current_scenario = generate_scenario(self._difficulty, self._rng)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Build signals for observation (without leaking impact_pct)
        agent_signals = [
            {"signal": s["signal"], "description": s["description"]}
            for s in self._current_scenario["signals"]
        ]

        return ForecastObservation(
            product_name=self._current_scenario["product_name"],
            product_category=self._current_scenario["product_category"],
            baseline_forecast=self._current_scenario["baseline_forecast"],
            time_period=self._current_scenario["time_period"],
            signals=agent_signals,
            difficulty=self._current_scenario["difficulty"],
            # Grading fields are None on reset
            expected_forecast=None,
            direction_score=None,
            magnitude_score=None,
            coverage_score=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: ForecastAction) -> ForecastObservation:
        """
        Grade the agent's adjusted forecast.

        Args:
            action: ForecastAction with the adjusted_forecast value

        Returns:
            ForecastObservation with scores and done=True
        """
        self._state.step_count += 1

        if self._current_scenario is None:
            return ForecastObservation(
                done=True,
                reward=0.0,
            )

        result = grade_forecast(
            baseline=self._current_scenario["baseline_forecast"],
            predicted=action.adjusted_forecast,
            signals=self._current_scenario["signals"],
        )

        return ForecastObservation(
            product_name=self._current_scenario["product_name"],
            product_category=self._current_scenario["product_category"],
            baseline_forecast=self._current_scenario["baseline_forecast"],
            time_period=self._current_scenario["time_period"],
            difficulty=self._current_scenario["difficulty"],
            signals=None,  # Don't repeat signals in response
            expected_forecast=result["expected_forecast"],
            direction_score=result["direction_score"],
            magnitude_score=result["magnitude_score"],
            coverage_score=result["coverage_score"],
            done=True,
            reward=result["reward"],
        )

    @property
    def state(self) -> State:
        return self._state
