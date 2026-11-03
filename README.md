# pocket-genie

Edge agent runtime for macOS and Android Termux with hybrid workspace retrieval.

## Modules
- `registry/`: Decorator tool registry & YAML skill loader
- `middlewares/`: 8-stage guardrail & summarization pipeline
- `retrieval/`: Alpha-blended BM25 + dense vector store with cross-feature reranking
- `quant/`: *(pending: SmoothQuant scale search & JSON schema parser)*

```bash
pip install -e .
pytest -q
```
