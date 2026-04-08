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
from typing import List, Optional

from openai import OpenAI

from server.environment import DemandForecastEnvironment
from models import ForecastAction

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TEMPERATURE = 0.2
MAX_TOKENS = 300
EPISODES_PER_TASK = 5

TASK_IDS = ["task_easy", "task_medium", "task_hard"]
TASK_DIFFICULTY = {
    "task_easy": "easy",
    "task_medium": "medium",
    "task_hard": "hard",
}

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


# ── Structured logging (matches validator format exactly) ────────────────────

def emit_log_line(prefix: str, fields: list) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields)
    print(f"{prefix} {payload}", flush=True)


def log_start(task: str, env: str, model: str) -> None:
    emit_log_line("[START]", [
        ("task", task),
        ("env", env),
        ("model", model),
    ])


def log_step(step: int, action: str, reward: float, done: bool) -> None:
    emit_log_line("[STEP]", [
        ("step", str(step)),
        ("action", action),
        ("reward", f"{reward:.2f}"),
        ("done", str(done).lower()),
    ])


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    emit_log_line("[END]", [
        ("success", str(success).lower()),
        ("steps", str(steps)),
        ("score", f"{score:.3f}"),
        ("rewards", rewards_str),
    ])


# ── Prompt building ─────────────────────────────────────────────────────────

def build_user_prompt(observation) -> str:
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
    if not response_text:
        return baseline

    try:
        data = json.loads(response_text.strip())
        if "adjusted_forecast" in data:
            val = float(data["adjusted_forecast"])
            if val > 0:
                return val
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    json_match = re.search(r'\{[^}]*"adjusted_forecast"\s*:\s*([\d.]+)[^}]*\}', response_text)
    if json_match:
        try:
            val = float(json_match.group(1))
            if val > 0:
                return val
        except ValueError:
            pass

    numbers = re.findall(r'[\d,]+\.?\d*', response_text)
    for num_str in numbers:
        try:
            val = float(num_str.replace(",", ""))
            if baseline * 0.3 < val < baseline * 5:
                return val
        except ValueError:
            continue

    return baseline


# ── Run one task (one [START]...[END] block per task) ────────────────────────

def run_task(
    client: OpenAI,
    env: DemandForecastEnvironment,
    task_id: str,
) -> dict:
    difficulty = TASK_DIFFICULTY[task_id]
    rewards: List[float] = []
    score = 0.0
    success = False

    log_start(
        task=task_id,
        env="demand_forecast",
        model=MODEL_NAME,
    )

    try:
        for ep in range(EPISODES_PER_TASK):
            seed = 100 + ep
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
                print(f"  Model request failed ({exc}). Using baseline as fallback.", flush=True)
                response_text = ""

            adjusted_forecast = parse_forecast(response_text, observation.baseline_forecast)
            action = ForecastAction(adjusted_forecast=adjusted_forecast)
            result = env.step(action)

            reward = result.reward
            rewards.append(reward)

            log_step(
                step=ep + 1,
                action=json.dumps({"adjusted_forecast": adjusted_forecast}),
                reward=reward,
                done=True,
            )

        score = sum(rewards) / len(rewards) if rewards else 0.0
        # Clamp score to strictly (0, 1)
        score = max(0.001, min(0.999, score))
        success = score > 0.5

    except Exception as exc:
        print(f"  Task {task_id} failed: {exc}", flush=True)
    finally:
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DemandForecastEnvironment()

    results = []
    for task_id in TASK_IDS:
        result = run_task(client, env, task_id)
        results.append(result)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  {r['task_id']:12s}: score={r['score']:.3f} success={r['success']}", flush=True)
    overall = sum(r["score"] for r in results) / len(results)
    print(f"  {'OVERALL':12s}: score={overall:.3f}", flush=True)


if __name__ == '__main__':
    main()
