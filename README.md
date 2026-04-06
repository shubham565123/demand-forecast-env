---
title: Demand Forecast Adjuster
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Demand Forecast Adjuster

An OpenEnv environment that trains AI agents to adjust statistical demand forecasts
using contextual market signals — the core task in Supply Chain S&OP (Sales & Operations Planning).

## Why This Matters

Demand forecasting drives billions in inventory, production, and supply chain decisions.
Statistical models produce a baseline forecast, but real-world factors — festive seasons,
promotions, competitor activity, economic shifts — require human analysts to make
contextual adjustments. This environment teaches LLMs to perform that adjustment task.

**No supply chain or S&OP environment currently exists in the OpenEnv Hub.**

## How It Works

```
reset(difficulty="easy")
  → Observation: product info, baseline forecast, context signals

step(ForecastAction(adjusted_forecast=13000))
  → Observation: scores, expected value, done=True
```

Each episode is single-step:
1. **reset()** generates a scenario: a product, baseline forecast, and 1-5 context signals
2. The agent reads the signals and submits an **adjusted forecast** as a number
3. **step()** grades the adjustment on three dimensions and returns a 0.0–1.0 reward

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `adjusted_forecast` | `float` | The agent's adjusted demand forecast in units |
| `reasoning` | `str` (optional) | The agent's explanation for the adjustment |

## Observation Space

| Field | Type | When Available |
|-------|------|----------------|
| `product_name` | `str` | reset |
| `product_category` | `str` | reset |
| `baseline_forecast` | `float` | reset |
| `time_period` | `str` | reset |
| `signals` | `list[dict]` | reset |
| `difficulty` | `str` | reset |
| `expected_forecast` | `float` | step |
| `direction_score` | `float` | step |
| `magnitude_score` | `float` | step |
| `coverage_score` | `float` | step |

## Grading (Reward Function)

The reward is a weighted combination of three deterministic scores:

| Dimension | Weight | Score Range | What It Measures |
|-----------|--------|-------------|-----------------|
| Direction | 35% | 0.0 or 1.0 | Did the agent adjust up/down correctly? |
| Magnitude | 40% | 0.0 – 1.0 | How close to the expected value? (5 tiers) |
| Coverage | 25% | 0.0 – 1.0 | Did the agent account for all signals? |

**Magnitude scoring tiers:**
- Within 5% of expected → 1.0
- Within 10% → 0.8
- Within 20% → 0.6
- Within 35% → 0.4
- Within 50% → 0.2
- Beyond 50% → 0.0

## Tasks (3 Difficulty Levels)

| Difficulty | Signals | Challenge |
|------------|---------|-----------|
| **Easy** | 1 signal | Straightforward single adjustment |
| **Medium** | 2-3 signals (may conflict) | Must weigh opposing forces |
| **Hard** | 4-5 signals with interactions | Multiplicative effects, conflicting priorities |

## Setup & Usage

### Install
```bash
pip install openenv-core
```

### Run locally
```bash
cd demand_forecast_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Connect via client
```python
from demand_forecast_env import DemandForecastEnv, ForecastAction

with DemandForecastEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(difficulty="easy", seed=42)
    print(f"Baseline: {result.observation.baseline_forecast}")

    result = env.step(ForecastAction(adjusted_forecast=13000))
    print(f"Reward: {result.reward}")
```

### Run baseline inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-token"
python inference.py
```

### Docker
```bash
docker build -t demand-forecast-env -f server/Dockerfile .
docker run -p 8000:8000 demand-forecast-env
```

## Baseline Scores

Scores will be populated after running inference with the target model.

| Difficulty | Avg Reward | Direction | Magnitude | Coverage |
|------------|-----------|-----------|-----------|----------|
| Easy | 1.0000 | 1.0 | 1.0 | 1.0 |
| Medium | 0.8080 | 1.0 | 0.52 | 1.0 |
| Hard | 0.7075 | 0.6 | 0.6 | 0.75 |
