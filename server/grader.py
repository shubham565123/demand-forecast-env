"""
Grading logic for the Demand Forecast Adjuster Environment.

Scores an agent's adjusted forecast on three dimensions:
  1. Direction  — Did the agent adjust in the correct direction? (0.0 or 1.0)
  2. Magnitude  — How close is the adjustment to the expected value? (0.0–1.0)
  3. Coverage   — Did the agent account for all signals? (0.0–1.0)

Final reward = weighted combination of the three scores.
"""

from typing import Dict, List


# Weights for the three grading dimensions
W_DIRECTION = 0.35
W_MAGNITUDE = 0.40
W_COVERAGE = 0.25


def compute_expected_forecast(
    baseline: float,
    signals: List[Dict],
) -> float:
    """
    Compute the ground-truth adjusted forecast from baseline + signals.

    Each signal has a known `impact_pct` (e.g., +30 means +30%).
    Signals are applied multiplicatively to the baseline.

    Example:
        baseline = 10000
        signals = [{"impact_pct": 30}, {"impact_pct": -5}]
        expected = 10000 * (1 + 0.30) * (1 + (-0.05)) = 12350
    """
    adjusted = baseline
    for signal in signals:
        pct = signal["impact_pct"]
        adjusted *= 1 + (pct / 100.0)
    return round(adjusted, 2)


def score_direction(
    baseline: float,
    predicted: float,
    expected: float,
) -> float:
    """
    Score whether the agent adjusted in the correct direction.

    Returns:
        1.0 if direction is correct (or no adjustment was needed)
        0.0 if direction is wrong
    """
    expected_delta = expected - baseline
    predicted_delta = predicted - baseline

    # If no change was needed (expected == baseline), any answer is fine
    if abs(expected_delta) < 0.01:
        return 1.0

    # Check if both deltas have the same sign
    if expected_delta > 0 and predicted_delta > 0:
        return 1.0
    if expected_delta < 0 and predicted_delta < 0:
        return 1.0

    return 0.0


def score_magnitude(
    predicted: float,
    expected: float,
) -> float:
    """
    Score how close the predicted forecast is to the expected value.

    Uses a tolerance-based approach:
      - Within 5% of expected  → 1.0
      - Within 10%             → 0.8
      - Within 20%             → 0.6
      - Within 35%             → 0.4
      - Within 50%             → 0.2
      - Beyond 50%             → 0.0

    This gives a smooth gradient for partial credit.
    """
    if expected == 0:
        return 1.0 if abs(predicted) < 0.01 else 0.0

    pct_error = abs(predicted - expected) / abs(expected)

    if pct_error <= 0.05:
        return 1.0
    elif pct_error <= 0.10:
        return 0.8
    elif pct_error <= 0.20:
        return 0.6
    elif pct_error <= 0.35:
        return 0.4
    elif pct_error <= 0.50:
        return 0.2
    else:
        return 0.0


def score_coverage(
    predicted: float,
    baseline: float,
    signals: List[Dict],
) -> float:
    """
    Score whether the agent accounted for all signals.

    Heuristic: for each signal, check if the adjustment direction is
    consistent with that signal being considered. A signal is considered
    "covered" if removing it from the calculation would move the expected
    value further from the prediction.

    Returns: fraction of signals covered (0.0 to 1.0)
    """
    if not signals:
        return 1.0

    covered = 0
    predicted_delta = predicted - baseline

    for signal in signals:
        pct = signal["impact_pct"]

        if abs(pct) < 0.01:
            # Negligible signal, count as covered
            covered += 1
            continue

        # Compute expected WITHOUT this signal
        other_signals = [s for s in signals if s is not signal]
        expected_without = compute_expected_forecast(baseline, other_signals)
        expected_with_all = compute_expected_forecast(baseline, signals)

        # The signal's contribution to the expected forecast
        signal_contribution = expected_with_all - expected_without

        # If the prediction moved in the direction of this signal's
        # contribution (even partially), count it as covered
        if abs(signal_contribution) < 0.01:
            covered += 1
        elif signal_contribution > 0 and predicted_delta > 0:
            covered += 1
        elif signal_contribution < 0 and predicted_delta < 0:
            covered += 1
        elif abs(predicted_delta) < 0.01:
            # Agent didn't adjust at all — none of the signals are covered
            pass
        else:
            # For conflicting signals: if the agent's direction matches
            # the NET effect, give credit for the dominant signal
            net_effect = expected_with_all - baseline
            if (net_effect > 0 and predicted_delta > 0) or (
                net_effect < 0 and predicted_delta < 0
            ):
                covered += 0.5

    return round(covered / len(signals), 4)


def grade_forecast(
    baseline: float,
    predicted: float,
    signals: List[Dict],
) -> Dict[str, float]:
    """
    Grade a forecast adjustment on all three dimensions.

    Args:
        baseline: The original statistical baseline forecast
        predicted: The agent's adjusted forecast
        signals: List of signal dicts, each with 'impact_pct'

    Returns:
        Dictionary with:
          - expected_forecast: ground truth
          - direction_score: 0.0 or 1.0
          - magnitude_score: 0.0 to 1.0
          - coverage_score: 0.0 to 1.0
          - reward: weighted final score 0.0 to 1.0
    """
    expected = compute_expected_forecast(baseline, signals)

    d_score = score_direction(baseline, predicted, expected)
    m_score = score_magnitude(predicted, expected)
    c_score = score_coverage(predicted, baseline, signals)

    reward = round(
        W_DIRECTION * d_score + W_MAGNITUDE * m_score + W_COVERAGE * c_score,
        4,
    )

    # Clamp to strictly within (0, 1) — validators reject exactly 0.0 and 1.0
    reward = max(0.01, min(0.99, reward))

    reward = max(0.0001, min(0.9999, reward))
    return {
        "expected_forecast": expected,
        "direction_score": d_score,
        "magnitude_score": m_score,
        "coverage_score": c_score,
        "reward": reward,
    }
