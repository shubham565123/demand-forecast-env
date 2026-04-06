"""
Inference Script — Demand Forecast Adjuster
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import json
import os
import re
import textwrap
from typing import Optional

from openai import OpenAI

# ── Environment imports ──────────────────────────────────────────────────────
from server.environment import DemandForecastEnvironment
from models import ForecastAction

# ── Configuration from environment variables ─────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

TEMPERATURE = 0.2
MAX_TOKENS = 300
EPISODES_PER_TASK = 5  # Number of episodes per difficulty level

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert demand planning analyst. You will receive a product's
    baseline statistical demand forecast along with contextual market signals
    (promotions, festive seasons, competitor activity, etc.).

    Your job is to output an ADJUSTED forecast that accounts for all the signals.

    Rules:
    - Each signal has a directional impact (positive or negative) on demand
    - Consider ALL signals, not just the strongest one
    - Signals interact multiplicatively (a +30% and a -10% don't simply net to +20%)
    - Your output must be a JSON object with exactly this format:
      {"adjusted_forecast": <number>, "reasoning": "<brief explanation>"}
    - The adjusted_forecast must be a positive number representing units
    - Do NOT include any text outside the JSON object
""").strip()


def build_user_prompt(observation) -> str:
    """Build the user prompt from the environment observation."""
    signals_text = ""
    if observation.signals:
        for i, s in enumerate(observation.signals, 1):
            signals_text += f"  {i}. [{s['signal']}] {s['description']}\n"
    else:
        signals_text = "  (none)\n"

    return textwrap.dedent(f"""
        Product: {observation.product_name} ({observation.product_category})
        Time Period: {observation.time_period}
        Baseline Statistical Forecast: {observation.baseline_forecast} units
        Difficulty: {observation.difficulty}

        Context Signals:
        {signals_text}
        Based on these signals, provide your adjusted demand forecast.
        Respond with ONLY a JSON object: {{"adjusted_forecast": <number>, "reasoning": "<explanation>"}}
    """).strip()


def parse_forecast(response_text: str, baseline: float) -> float:
    """
    Parse the LLM's response to extract the adjusted forecast.

    Falls back to baseline if parsing fails.
    """
    if not response_text:
        return baseline

    # Try to parse JSON directly
    try:
        data = json.loads(response_text.strip())
        if "adjusted_forecast" in data:
            val = float(data["adjusted_forecast"])
            if val > 0:
                return val
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to find JSON in the response
    json_match = re.search(r'\{[^}]*"adjusted_forecast"\s*:\s*([\d.]+)[^}]*\}', response_text)
    if json_match:
        try:
            val = float(json_match.group(1))
            if val > 0:
                return val
        except ValueError:
            pass

    # Try to find any number that looks like a forecast
    numbers = re.findall(r'[\d,]+\.?\d*', response_text)
    for num_str in numbers:
        try:
            val = float(num_str.replace(",", ""))
            # Only accept values in a reasonable range relative to baseline
            if baseline * 0.3 < val < baseline * 5:
                return val
        except ValueError:
            continue

    return baseline


def run_episode(
    client: OpenAI,
    env: DemandForecastEnvironment,
    difficulty: str,
    seed: int,
) -> dict:
    """Run a single episode and return the results."""
    observation = env.reset(difficulty=difficulty, seed=seed)

    user_prompt = build_user_prompt(observation)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  Model request failed ({exc}). Using baseline as fallback.")
        response_text = ""

    adjusted_forecast = parse_forecast(response_text, observation.baseline_forecast)
    action = ForecastAction(adjusted_forecast=adjusted_forecast)
    result = env.step(action)

    return {
        "difficulty": difficulty,
        "seed": seed,
        "product": observation.product_name,
        "baseline": observation.baseline_forecast,
        "predicted": adjusted_forecast,
        "expected": result.expected_forecast,
        "reward": result.reward,
        "direction_score": result.direction_score,
        "magnitude_score": result.magnitude_score,
        "coverage_score": result.coverage_score,
    }


def main() -> None:
    """Run baseline inference across all three difficulty levels."""
    print("[START]")
    print("=" * 60)
    print("Demand Forecast Adjuster — Baseline Inference")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Episodes per difficulty: {EPISODES_PER_TASK}")
    print()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DemandForecastEnvironment()

    all_results = []

    for difficulty in ["easy", "medium", "hard"]:
        print(f"--- {difficulty.upper()} ---")
        task_rewards = []

        for ep in range(EPISODES_PER_TASK):
            seed = 100 + ep  # Deterministic seeds for reproducibility
            result = run_episode(client, env, difficulty, seed)
            all_results.append(result)
            task_rewards.append(result["reward"])

            print(f"[STEP] {diff} episode {ep+1}")
            print(
                f"  Episode {ep+1}: "
                f"product={result['product']}, "
                f"baseline={result['baseline']:.0f}, "
                f"predicted={result['predicted']:.0f}, "
                f"expected={result['expected']:.0f}, "
                f"reward={result['reward']:.4f} "
                f"(dir={result['direction_score']:.1f}, "
                f"mag={result['magnitude_score']:.1f}, "
                f"cov={result['coverage_score']:.2f})"
            )

        avg_reward = sum(task_rewards) / len(task_rewards)
        print(f"  Average reward ({difficulty}): {avg_reward:.4f}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for difficulty in ["easy", "medium", "hard"]:
        task_results = [r for r in all_results if r["difficulty"] == difficulty]
        avg = sum(r["reward"] for r in task_results) / len(task_results)
        print(f"  {difficulty:6s}: avg_reward = {avg:.4f}")

    overall_avg = sum(r["reward"] for r in all_results) / len(all_results)
    print("[END]")
    print(f"  {'OVERALL':6s}: avg_reward = {overall_avg:.4f}")
    print()


if __name__ == "__main__":
    main()
