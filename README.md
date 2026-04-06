
# Demand Forecast Adjuster

An OpenEnv environment that trains AI agents to adjust statistical demand forecasts using contextual market signals — the core task in Supply Chain S&OP (Sales & Operations Planning).

## Why This Matters

Demand forecasting drives billions in inventory, production, and supply chain decisions globally. Statistical models produce a baseline forecast, but real-world factors — festive seasons, promotions, competitor activity, economic shifts — require human analysts to make contextual adjustments. This is exactly what demand planners at companies like Unilever, P&G, and Nestle do every planning cycle.

This environment teaches LLMs to perform that adjustment task, bridging the gap between statistical forecasting and human judgment.

**No supply chain or S&OP environment currently exists in the OpenEnv Hub — this is a novel domain.**

## How It Works

Each episode is single-step:
1. `reset(difficulty="easy")` generates a scenario with a product, baseline forecast, and context signals
2. The agent reads the signals and submits an adjusted forecast number
3. `step()` grades the adjustment on three dimensions and returns a 0.0-1.0 reward

```python
from demand_forecast_env import DemandForecastEnv, ForecastAction

with DemandForecastEnv(base_url="https://shubhamyeole565-demand-forecast-env.hf.space").sync() as env:
    result = env.reset(difficulty="easy", seed=42)
    print(f"Product: {result.observation.product_name}")
    print(f"Baseline: {result.observation.baseline_forecast} units")
    result = env.step(ForecastAction(adjusted_forecast=5600))
    print(f"Reward: {result.reward}")
```

## Example Scenario

**Easy task:**
```
Product: GlowPlus Body Lotion (Skin Care)
Baseline: 4,600 units
Period: February 2026
Signal: "Product went viral on social media - expected 22% boost"
Expected adjustment: 4600 * 1.22 = 5,612 units
```

**Hard task:**
```
Product: GlowPlus Body Lotion (Skin Care)
Baseline: 4,600 units
Signals:
  1. Diwali festive season - +28% increase
  2. BOGO promotion - +38% uplift
  3. Competitor launched cheaper alternative - -25% drop
  4. Economic slowdown - -15% decline
Expected: 4600 * 1.28 * 1.38 * 0.75 * 0.85 = ~5,180 units
```

The LLM must reason about how these signals interact multiplicatively, not just add them up.

## Action Space

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `adjusted_forecast` | `float` | Yes | The agent's adjusted demand forecast in units |
| `reasoning` | `str` | No | The agent's explanation for the adjustment |

## Observation Space

**On reset (the scenario):**

| Field | Type | Description |
|-------|------|-------------|
| `product_name` | `str` | Name of the product |
| `product_category` | `str` | Category (e.g., Hair Care, Home Care) |
| `baseline_forecast` | `float` | Statistical baseline forecast in units |
| `time_period` | `str` | Forecast period (e.g., November 2025) |
| `signals` | `list[dict]` | Context signals with name and description |
| `difficulty` | `str` | Task difficulty: easy, medium, or hard |

**On step (the grading result):**

| Field | Type | Description |
|-------|------|-------------|
| `expected_forecast` | `float` | Ground-truth adjusted forecast |
| `direction_score` | `float` | Directional correctness (0.0 or 1.0) |
| `magnitude_score` | `float` | Magnitude accuracy (0.0 to 1.0) |
| `coverage_score` | `float` | Signal coverage (0.0 to 1.0) |
| `reward` | `float` | Weighted final score (0.0 to 1.0) |

## Grading (Reward Function)

The reward is a weighted combination of three deterministic, reproducible scores:

### 1. Direction Score (35% weight)
Did the agent adjust in the correct direction? Correct = 1.0, Wrong = 0.0.

### 2. Magnitude Score (40% weight)
How close is the prediction to the expected value?
- Within 5%: 1.0
- Within 10%: 0.8
- Within 20%: 0.6
- Within 35%: 0.4
- Within 50%: 0.2
- Beyond 50%: 0.0

### 3. Coverage Score (25% weight)
Did the agent account for all signals? Computed as fraction of signals that influenced the adjustment.

**Final reward** = 0.35 x direction + 0.40 x magnitude + 0.25 x coverage

## Tasks (3 Difficulty Levels)

| Difficulty | Signals | Challenge |
|------------|---------|-----------|
| **Easy** | 1 signal | Straightforward single adjustment |
| **Medium** | 2-3 signals (same direction) | Compute combined magnitude |
| **Hard** | 4-5 signals (conflicting) | Weigh opposing forces with multiplicative effects |

## Signal Types

**Positive signals:** Festive season (Diwali), BOGO promotions, peak summer season, social media trends, new distribution channels, wedding/gifting season.

**Negative signals:** Competitor launches, price increases, monsoon disruption, negative publicity, economic slowdown.

## Project Structure

```
demand_forecast_env/
├── __init__.py              # Module exports
├── models.py                # Pydantic Action and Observation models
├── client.py                # WebSocket EnvClient subclass
├── inference.py             # Baseline inference script (OpenAI client)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── Dockerfile               # Container definition
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI server (create_app)
    ├── environment.py       # Core logic (reset/step/state)
    └── grader.py            # 3-dimension scoring engine
```

## Setup and Usage

### Run locally
```bash
pip install openenv-core openai pydantic uvicorn fastapi
cd demand_forecast_env
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run baseline inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-token-here"
PYTHONPATH=. python inference.py
```

### Docker
```bash
docker build -t demand-forecast-env .
docker run -p 8000:8000 demand-forecast-env
```

## Baseline Scores

Model: meta-llama/Llama-3.1-8B-Instruct via HuggingFace Inference API (5 episodes per difficulty)

| Difficulty | Avg Reward | Direction | Magnitude | Coverage |
|------------|-----------|-----------|-----------|----------|
| Easy | 1.0000 | 1.0 | 1.0 | 1.0 |
| Medium | 0.8080 | 1.0 | 0.52 | 1.0 |
| Hard | 0.7075 | 0.6 | 0.6 | 0.75 |
| **Overall** | **0.8385** | | | |

## Design Decisions

1. **Single-step episodes**: Each scenario is independent, making the environment fast to run and easy to scale for RL training.

2. **Multiplicative signal composition**: Signals are applied multiplicatively (not additively) to the baseline, matching real-world demand planning where a 30% promo lift on top of a 20% seasonal lift gives 1.3 x 1.2 = 1.56x, not 1.5x.

3. **Three-dimension grading**: Rather than binary correct/incorrect, the grader rewards partial progress. Getting the direction right but magnitude wrong still earns partial credit, providing richer gradient signal for RL training.

4. **No external dependencies**: The grading logic is pure Python math with no external APIs or databases, ensuring fast execution and deterministic scoring.

## Use Cases

- **RL Training**: Train LLMs to become better demand planners using GRPO/PPO with the multi-dimensional reward signal
- **Agent Evaluation**: Benchmark how well different LLMs handle numerical reasoning with real-world context
- **Supply Chain AI Research**: Study how LLMs reason about competing market signals

