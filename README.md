# pocket-genie

Quantization-aware edge agent runtime for macOS and Android Termux.

## Architecture
- `registry/` & `middlewares/`: Tool registry, skills, and guardrail chain
- `retrieval/`: Alpha-blended BM25 + dense workspace store
- `quant/`: Kurtosis mixed-precision profiles, logit-margin token router, and BOS-sink INT4 KV cache
- `core/`: *(in progress: specialist delegation & execution graph)*

```bash
pip install -e .
pytest -q
```
