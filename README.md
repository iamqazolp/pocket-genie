# pocket-genie

Edge agent runtime for macOS and Android Termux.

## Modules
- `registry/`: Decorator tool registry & YAML skill loader
- `middlewares/`: HITL gate, SQL CTE allowlist, refusal detection, rolling summary
- `retrieval/`: Sequential markdown/code chunker & BM25 lexical index
- `quant/`: *(pending: SmoothQuant & mixed-precision activation hooks)*

```bash
pip install -e .
pytest -q
```
