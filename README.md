# Local Model Router for Linux

Most local multi-model routing tutorials assume Apple Silicon or NVIDIA CUDA. This one runs on an AMD Radeon Pro with 8GB VRAM on Linux.

## What It Does

Routes queries through a **hot classifier** to **cold-loaded specialist** models — all running locally on your GPU via [llama.cpp](https://github.com/ggml-org/llama.cpp) with Vulkan.

```
User Query → Classifier (always loaded, ~1.7GB VRAM)
                    ↓
         ┌─────────┼─────────┐
         ↓         ↓         ↓
      Code      Reasoning   General
    Specialist  Specialist  Specialist
   (cold-load) (cold-load) (cold-load)
```

**Why route instead of using one model?** A 1.7B classifier takes ~100ms to determine whether your query needs a code model, a reasoning model, or a general-purpose model. Then only the right specialist loads. On 8GB VRAM you can't fit multiple large models simultaneously — routing lets you use the best model for each task while staying within your VRAM budget.

## Benchmarks (AMD Radeon Pro WX 7100, 8GB, Vulkan)

| Component | Model | VRAM | Speed | Cold Load |
|-----------|-------|------|-------|-----------|
| Classifier | SmolLM2-1.7B Q8 | 1.7 GB | 78-160ms per classification | 17s (first load) |
| Code Specialist | LFM2.5-1.2B Q8 | 1.2 GB | ~102 tok/s | 12s |
| General Specialist | LFM2.5-1.2B Q8 | 1.2 GB | ~102 tok/s | 12s |
| Reasoning Specialist | Nemotron-8B Q4 | 4.7 GB | ~1.8 tok/s | 48s |

**Key finding**: 1-2B models are the sweet spot for Polaris/GCN GPUs at ~100 tok/s. 8B models work but are impractically slow (~2 tok/s). On RDNA 2/3 GPUs (RX 6000/7000), 7-8B models will be significantly faster.

Full benchmark data: [`benchmarks/results/wx7100_vulkan.json`](benchmarks/results/wx7100_vulkan.json)

## Quick Start

### Prerequisites

- Linux with AMD GPU (or NVIDIA — works with CUDA build too)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with Vulkan: `cmake -DGGML_VULKAN=ON ..`
- Python 3.10+
- GGUF model files (see [Model Selection](#model-selection))

### Setup

```bash
git clone https://github.com/wayneColt/local-model-router-linux.git
cd local-model-router-linux

# Verify your environment
bash scripts/setup_vulkan.sh

# Copy and edit config
cp config.example.json config.json
# Edit config.json: set llama_server_path and model paths
```

### Run

```bash
# Interactive chat with automatic routing
python3 examples/interactive_chat.py

# Batch routing demo
python3 examples/batch_routing.py

# Run benchmarks on your hardware
python3 benchmarks/run_benchmarks.py
```

## How It Works

1. **Classifier** (System 1): A small model (~1.7B) stays loaded in VRAM at all times. It reads every incoming query and classifies it as `code`, `reasoning`, or `general` in under 200ms.

2. **Specialist Manager** (System 2): Based on the classification, the router cold-loads the appropriate specialist model. If the same specialist is already loaded from a previous query, it stays hot — no reload penalty.

3. **Routing**: The query is forwarded to the specialist's llama-server instance via the OpenAI-compatible chat completions API. The response is returned along with timing metadata.

The hot/cold pattern means you only pay the cold-load penalty (~12s for a 1.2B model) when switching between task types. Sequential queries of the same type are fast.

## Model Selection

The router is model-agnostic — any GGUF model works. Here are practical recommendations by VRAM budget:

### 8GB VRAM (RX 580, WX 7100, RX 6600)

| Role | Model | VRAM | Notes |
|------|-------|------|-------|
| Classifier | SmolLM2-1.7B Q8_0 | ~1.7 GB | Fast, accurate classification |
| Code | Qwen2.5-Coder-3B Q8_0 | ~3.5 GB | Or any code-tuned model ≤3B |
| Reasoning | Qwen2.5-3B Q8_0 | ~3.5 GB | Or Phi-3-mini-4k |
| General | SmolLM2-1.7B Q8_0 | ~1.7 GB | Reuse the classifier model |

### 12-16GB VRAM (RX 6700 XT, RX 7800 XT)

| Role | Model | VRAM | Notes |
|------|-------|------|-------|
| Classifier | SmolLM2-1.7B Q8_0 | ~1.7 GB | Same fast classifier |
| Code | Qwen2.5-Coder-7B Q4_K_M | ~4.4 GB | Strong code generation |
| Reasoning | Qwen2.5-7B Q4_K_M | ~4.4 GB | Good reasoning at 7B |
| General | Llama-3.2-3B Q8_0 | ~3.5 GB | Conversational |

### 24GB VRAM (RX 7900 XTX)

With 24GB you can keep multiple specialists loaded simultaneously. Modify `specialist_manager.py` to skip unloading.

## Configuration

Edit `config.json`:

```json
{
    "llama_server_path": "/path/to/llama-server",
    "classifier": {
        "path": "/path/to/SmolLM2-1.7B-Instruct-Q8_0.gguf",
        "vram_mb": 1700,
        "context_size": 512
    },
    "specialists": {
        "code": {
            "path": "/path/to/code-model.gguf",
            "vram_mb": 3500,
            "context_size": 2048
        }
    }
}
```

Key settings:
- `vram_total_mb`: Your GPU's VRAM (8192 for 8GB)
- `vram_reserved_mb`: VRAM reserved for display/compositor (2048 is conservative)
- `gpu_device`: Vulkan device index (usually 0)
- `context_size`: Smaller = less VRAM, faster. 512 is enough for classification.

## Project Structure

```
local-model-router-linux/
├── router/
│   ├── config.py              # Configuration and model specs
│   ├── classifier.py          # Hot classifier (always loaded)
│   ├── specialist_manager.py  # Cold-load specialist lifecycle
│   └── router.py              # Main routing orchestration
├── examples/
│   ├── interactive_chat.py    # Try it yourself
│   └── batch_routing.py       # Route a batch, see decisions
├── benchmarks/
│   ├── run_benchmarks.py      # Automated benchmark suite
│   └── results/               # Hardware-specific benchmark data
├── scripts/
│   ├── setup_vulkan.sh        # Environment verification
│   └── download_models.sh     # Model download helper
├── docs/
│   └── hardware_notes.md      # AMD Vulkan specifics, VRAM math
└── config.example.json        # Template configuration
```

## NVIDIA Support

This router also works with NVIDIA GPUs. Build llama.cpp with `-DGGML_CUDA=ON` instead of `-DGGML_VULKAN=ON`. The routing architecture is identical — only the GPU backend changes.

## Contributing

Benchmark results from other hardware are welcome. Run `benchmarks/run_benchmarks.py` on your setup and submit the JSON file from `benchmarks/results/`.

## License

MIT — Wayne Colt, 2026
