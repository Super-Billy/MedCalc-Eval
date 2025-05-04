## Environment Setup Instructions



### 1.  Prepare a clean Python interpreter

| Option                | Recommendation                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Conda**             | `bash\nconda create -n medcalc_llm python=3.10 -y\nconda activate medcalc_llm\n`                                        |
| **virtualenv / venv** | `bash\npython3 -m venv .venv\nsource .venv/bin/activate  # Linux / macOS\n.\.venv\\Scripts\\activate      # Windows\n` |

> All listed libraries are officially tested against ≥3.9; we validated the full pipeline with 3.10 to avoid incompatibilities that may arise in 3.11+.



### 2.  Install CUDA‑matching PyTorch (GPU users only)

If you **plan to run local vLLM models on GPU**, first install the PyTorch build that matches your CUDA driver.
Replace `<cuXX>` with your CUDA version (e.g. `cu121`) or use `cpu` if you have no GPU.

```bash
pip install --index-url https://download.pytorch.org/whl/<cuXX> torch==2.6.0
```


### 3.  Install project dependencies

```bash
# at repo root
pip install -r requirements.txt
```

---

### 4.  Configure environment variables

Create a .env file in the root dir and edit it:

Add any of the following keys that you intend to use:

| Variable               | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `OPENAI_API_KEY`       | Required for OpenAI models               |
| `DEEPSEEK_API_KEY`     | Required for LLM-Eval                    |
| `VLLM_ENDPOINT`        | Override default localhost vLLM endpoint |
| `CUDA_VISIBLE_DEVICES` | (Optional) Select GPUs for vLLM serving  |


### 5.  (Optional) Launch a local vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
       --model deepseek-ai/deepseek-llm-14b-chat \
       --port 8000
```

Confirm the server responds at `http://localhost:8000/v1/chat/completions` before running the generation script.


### 6.  Verify installation

```bash
python - <<'PY'
import torch, transformers, vllm, openai, pandas
print("✅  Torch:", torch.__version__)
print("✅  Transformers:", transformers.__version__)
print("✅  vLLM:", vllm.__version__)
print("✅  OpenAI:", openai.__version__)
print("✅  Pandas:", pandas.__version__)
PY
```

You should see the exact versions listed below.

---

## Full Dependency List (locked versions)

| Package             | Version    |
| ------------------- | ---------- |
| aiofiles            | **24.1.0** |
| langchain           | **0.3.25** |
| langchain‑community | **0.3.23** |
| langchain‑core      | **0.3.58** |
| langchain‑openai    | **0.3.16** |
| nest-asyncio        | **1.6.0**  |
| numpy               | **2.2.5**  |
| openai              | **1.77.0** |
| pandas              | **2.2.3**  |
| pydantic            | **2.11.4** |
| python‑dotenv       | **1.1.0**  |
| tenacity            | **9.0.0**  |
| tiktoken            | **0.9.0**  |
| torch               | **2.6.0**  |
| tqdm                | **4.67.1** |
| transformers        | **4.51.2** |
| vllm                | **0.8.3**  |

**Tip:** Keep these exact versions to guarantee reproducibility.
If you must upgrade, retest the pipeline end‑to‑end.

## 7. Run the Evaluation Code

```bash
python plain.py
```