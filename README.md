# Math Eval Harness

Math evaluation pipeline for **Gemma 3** and **Qwen 3**, based on [Uni-DPO](https://github.com/pspdada/Uni-DPO).

## What Changed from Uni-DPO

1. **Dataset download scripts** — Added `scripts/download_data.py` to automatically fetch `math500`, `minerva_math`, and `aime24` from HuggingFace into the expected JSONL format. The original repo ships no data.

2. **Field name fixes** — Uni-DPO's parser expects a `solution` field (full solution with `\boxed{}`) for `math500` and `minerva_math`. The downloaded data is converted accordingly.

3. **Gemma 3 support** — Use `--apply_chat_template` with `--prompt_type cot` instead of the default `qwen25-math-cot`. The chat template wraps the prompt as a single user turn with `add_generation_prompt=True`.

4. **Bundled `latex2sympy2`** — The grader depends on `latex2sympy2` which is included in `Math/evaluation/latex2sympy/`. Install it explicitly (see setup below).

5. **Blackwell GPU fix** — RTX 5090 (sm_120, CUDA 13.0) requires a CUDA-matched vLLM wheel. Install via `uv pip install vllm --torch-backend=auto` instead of plain `pip install vllm`.

## Setup

```bash
# 1. Clone
git clone https://github.com/cpsu00/math-eval-harness.git
cd math-eval-harness

# 2. Create env (uv recommended)
uv venv .venv --python 3.11
source .venv/bin/activate

# 3. Install dependencies
uv pip install vllm --torch-backend=auto   # picks correct CUDA wheel automatically
uv pip install transformers accelerate datasets tqdm sympy word2number Pebble timeout-decorator
uv pip install Math/evaluation/latex2sympy/  # bundled latex2sympy2

# 4. HuggingFace login (Gemma 3 is gated)
huggingface-cli login

# 5. Download datasets
python scripts/download_data.py --data_dir Math/evaluation/data
```

## Evaluate Gemma 3

### Instruct model (gemma-3-4b-it / gemma-3-12b-it / gemma-3-27b-it)

```bash
cd Math/evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/math_eval.py \
    --model_name_or_path google/gemma-3-4b-it \
    --data_name gsm8k,math500,minerva_math,aime24 \
    --data_dir ./data \
    --output_dir ./results/gemma3-4b-it \
    --split test \
    --prompt_type cot \
    --num_test_sample -1 \
    --max_tokens_per_call 3000 \
    --seed 0 --temperature 0 --n_sampling 1 --top_p 1 \
    --start 0 --end -1 \
    --use_vllm \
    --apply_chat_template \
    --save_outputs
```

### Pre-trained base model (gemma-3-4b-pt)

> Note: base models score very low with 0-shot CoT since they don't follow the `\boxed{}` output format. The official report uses 8-shot for PT models (38.4% GSM8K). Add `--num_shots 8` for a fairer comparison.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/math_eval.py \
    --model_name_or_path google/gemma-3-4b-pt \
    --data_name gsm8k,math500,minerva_math,aime24 \
    --data_dir ./data \
    --output_dir ./results/gemma3-4b-pt \
    --split test \
    --prompt_type cot \
    --num_test_sample -1 \
    --max_tokens_per_call 3000 \
    --seed 0 --temperature 0 --n_sampling 1 --top_p 1 \
    --start 0 --end -1 \
    --use_vllm \
    --save_outputs
```

## Evaluate Qwen 3

### Instruct model (Qwen3-4B / Qwen3-8B / Qwen3-14B / ...)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/math_eval.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --data_name gsm8k,math500,minerva_math,aime24 \
    --data_dir ./data \
    --output_dir ./results/qwen3-4b \
    --split test \
    --prompt_type qwen25-math-cot \
    --num_test_sample -1 \
    --max_tokens_per_call 3000 \
    --seed 0 --temperature 0 --n_sampling 1 --top_p 1 \
    --start 0 --end -1 \
    --use_vllm \
    --apply_chat_template \
    --save_outputs
```

## Benchmark Results

### Gemma 3 4B — IT vs PT (0-shot CoT, max_tokens=3000)

| Model | GSM8K | MATH-500 | MATH (full 5k) | AIME24 |
|---|---|---|---|---|
| **Gemma 3 4B-IT** | **88.9%** | **74.2%** | **75.4%** | **6.7%** |
| Gemma 3 4B-PT (0-shot) | 10.3% | 9.0% | 9.1% | 0.0% |
| Gemma 3 4B-IT (official report) | 89.2% | ~75.6% | 75.6% | — |
| Gemma 3 4B-PT (official, 8-shot) | 38.4% | 24.2% | — | — |

> PT model 0-shot is expected to be low — base models need few-shot examples to learn the answer format.

### MATH (full 5k) by Subject — Gemma 3 4B-IT vs PT

| Subject | IT | PT (0-shot) |
|---|---|---|
| Algebra | 91.2% | 11.8% |
| Prealgebra | 85.4% | 15.3% |
| Number Theory | 81.1% | 7.2% |
| Counting & Probability | 72.6% | 7.6% |
| Geometry | 66.4% | 7.7% |
| Intermediate Algebra | 58.5% | 5.4% |
| Precalculus | 57.7% | 4.2% |

## Supported Datasets

| Dataset | Source | Auto-download |
|---|---|---|
| `gsm8k` | HuggingFace (`gsm8k`) | ✅ automatic |
| `math500` | `HuggingFaceH4/MATH-500` | ✅ via download script |
| `minerva_math` | `EleutherAI/hendrycks_math` (all subjects) | ✅ via download script |
| `aime24` | `Maxwell-Jia/AIME_2024` | ✅ via download script |
| `olympiadbench` | not public | ❌ |
| `college_math` | not public | ❌ |
| `amc23` | not public | ❌ |
| `gaokao2023en` | not public | ❌ |

## Credits

Based on [Uni-DPO](https://github.com/pspdada/Uni-DPO) by pspdada et al.
