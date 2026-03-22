#!/bin/bash
# Verify Vulkan + llama.cpp setup for the local model router.
# Run this first to confirm your environment is ready.

set -e

echo "=== Vulkan Environment Check ==="
echo

# 1. Check for Vulkan runtime
echo "1. Vulkan loader..."
if command -v vulkaninfo &>/dev/null; then
    DEVICE=$(vulkaninfo --summary 2>/dev/null | grep deviceName | head -1 | cut -d= -f2 | xargs)
    DRIVER=$(vulkaninfo --summary 2>/dev/null | grep driverName | head -1 | cut -d= -f2 | xargs)
    echo "   Device: $DEVICE"
    echo "   Driver: $DRIVER"
elif [ -f /run/opengl-driver/lib/libvulkan_radeon.so ]; then
    echo "   RADV driver found (NixOS). vulkaninfo not in PATH."
    echo "   Install with: nix-shell -p vulkan-tools"
else
    echo "   WARNING: vulkaninfo not found. Install vulkan-tools."
fi

# 2. Check for AMD GPU via sysfs
echo
echo "2. AMD GPU (sysfs)..."
if [ -f /sys/class/drm/card1/device/mem_info_vram_total ]; then
    VRAM_TOTAL=$(cat /sys/class/drm/card1/device/mem_info_vram_total)
    VRAM_USED=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
    VRAM_FREE=$((VRAM_TOTAL - VRAM_USED))
    echo "   VRAM Total: $((VRAM_TOTAL / 1024 / 1024)) MB"
    echo "   VRAM Used:  $((VRAM_USED / 1024 / 1024)) MB"
    echo "   VRAM Free:  $((VRAM_FREE / 1024 / 1024)) MB"
else
    echo "   INFO: AMD sysfs VRAM info not at /sys/class/drm/card1/"
    echo "   Check card0 or use radeontop for VRAM monitoring."
fi

# 3. Check llama-server
echo
echo "3. llama-server..."
LLAMA_SERVER="${LLAMA_SERVER_PATH:-llama-server}"
if [ -f "$LLAMA_SERVER" ]; then
    echo "   Found: $LLAMA_SERVER"
    # Check Vulkan linkage
    if ldd "$LLAMA_SERVER" 2>/dev/null | grep -q vulkan; then
        echo "   Vulkan support: YES"
    else
        echo "   WARNING: llama-server not linked against Vulkan!"
        echo "   Rebuild with: cmake -DGGML_VULKAN=ON .."
    fi
elif command -v llama-server &>/dev/null; then
    LLAMA_SERVER=$(which llama-server)
    echo "   Found in PATH: $LLAMA_SERVER"
else
    echo "   NOT FOUND. Build llama.cpp with Vulkan:"
    echo "     git clone https://github.com/ggml-org/llama.cpp"
    echo "     cd llama.cpp && mkdir build && cd build"
    echo "     cmake -DGGML_VULKAN=ON .."
    echo "     cmake --build . --config Release -j$(nproc)"
fi

# 4. Check GGUF models
echo
echo "4. GGUF models..."
CONFIG="${1:-config.json}"
if [ -f "$CONFIG" ]; then
    echo "   Config: $CONFIG"
    # Extract model paths from JSON (basic parsing)
    MODELS=$(grep '"path"' "$CONFIG" | sed 's/.*"path".*:.*"\(.*\)".*/\1/')
    for model in $MODELS; do
        expanded=$(eval echo "$model")
        if [ -f "$expanded" ]; then
            SIZE=$(du -h "$expanded" | cut -f1)
            echo "   OK  $SIZE  $(basename "$expanded")"
        else
            echo "   MISSING: $expanded"
        fi
    done
else
    echo "   No config.json found. Create one from config.example.json"
fi

echo
echo "=== Setup check complete ==="
