#!/bin/bash
# Download recommended GGUF models for the local model router.
# Requires: curl or wget, ~8GB disk space
#
# These are suggested models. You can use any GGUF-compatible model.
# Adjust config.json paths after downloading.

set -e

MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR"

echo "=== Model Download Helper ==="
echo "Download directory: $MODEL_DIR"
echo

# HuggingFace base URL for GGUF downloads
HF_BASE="https://huggingface.co"

download_model() {
    local name="$1"
    local url="$2"
    local filename="$3"
    local target="$MODEL_DIR/$filename"

    if [ -f "$target" ]; then
        echo "  Already exists: $filename ($(du -h "$target" | cut -f1))"
        return
    fi

    echo "  Downloading: $name"
    echo "  URL: $url"
    echo "  Target: $target"

    if command -v curl &>/dev/null; then
        curl -L -o "$target" "$url" --progress-bar
    elif command -v wget &>/dev/null; then
        wget -O "$target" "$url" --show-progress
    else
        echo "  ERROR: Neither curl nor wget found."
        return 1
    fi

    echo "  Done: $(du -h "$target" | cut -f1)"
}

echo "=== Recommended Models ==="
echo
echo "1. Classifier: SmolLM2-1.7B-Instruct (Q8_0, ~1.7GB)"
echo "   Fast intent classification. Always loaded."
echo
echo "2. Code Specialist: Qwen2.5-Coder-7B-Instruct (Q4_K_M, ~4.4GB)"
echo "   Strong code generation. Cold-loaded for code tasks."
echo
echo "3. Reasoning Specialist: Qwen2.5-7B-Instruct (Q4_K_M, ~4.4GB)"
echo "   Analysis and reasoning. Cold-loaded for complex queries."
echo
echo "4. General Specialist: SmolLM2-1.7B-Instruct (Q8_0, ~1.7GB)"
echo "   Light general model. Or use the classifier model itself."
echo

echo "To download, visit huggingface.co and search for GGUF quantized versions."
echo "Popular GGUF providers: TheBloke, bartowski, unsloth"
echo
echo "Example with huggingface-cli:"
echo "  pip install huggingface-hub"
echo "  huggingface-cli download HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF \\"
echo "    smollm2-1.7b-instruct-q8_0.gguf --local-dir $MODEL_DIR"
echo
echo "After downloading, update config.json with the correct model paths."
