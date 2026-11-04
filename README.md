# pocket-genie

Quantization-aware multi-agent supervisor for macOS and Android Termux.

## Architecture
- `core/`: Tiered supervisor, specialist delegation wrappers, and execution graph
- `quant/`: SmoothQuant, kurtosis mixed-precision profiles, and BOS-sink INT4 KV cache
- `tools/` & `mcp/`: Workspace SQL, DevOps, Termux device tools, and MCP bridge

```bash
pip install -e .
pytest -q
```
