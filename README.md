# pocket-genie

Quantization-aware edge agent runtime for macOS and Android Termux.

## Current Stack
- `registry/` & `middlewares/`: Tool registry, skills, and guardrail chain
- `retrieval/`: Alpha-blended BM25 + dense workspace store
- `quant/`: SmoothQuant scale migration, fake-quant hooks, and FSM JSON schema parser

```bash
pip install -e .
pytest -q
```
