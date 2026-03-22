# Hardware Notes — AMD Vulkan on Linux

## Tested Hardware

| Component | Specification |
|-----------|--------------|
| GPU | AMD Radeon Pro WX 7100, 8GB GDDR5 |
| Driver | Mesa RADV (Polaris10) via Vulkan |
| CPU | AMD Ryzen 7 5800XT (8C/16T) |
| RAM | 64GB DDR4 |
| OS | NixOS 26.05 (Linux 6.18) |
| llama.cpp | Built from source with `-DGGML_VULKAN=ON` |

## VRAM Budget on 8GB Cards

With an 8GB VRAM card and a Linux desktop compositor running, expect:

| Consumer | VRAM |
|----------|------|
| Desktop compositor (Wayland) | ~200-500MB |
| Display framebuffers (4K + ultrawide) | ~300-500MB |
| **Available for inference** | **~5-7GB** |

### Practical VRAM Layout

```
Total VRAM:     8192 MB
├── Display:    ~800 MB  (dual monitor, Wayland compositor)
├── Classifier: ~1700 MB (SmolLM2-1.7B Q8_0, always loaded)
├── Specialist: ~1200 MB (1.2B model) or ~4700 MB (8B Q4_K_M model)
└── Overhead:   ~500 MB  (context buffers, Vulkan allocations)
```

**Key constraint**: Only one specialist loaded at a time. The router unloads the current specialist before loading a new one.

### Model Size vs VRAM

| Model | Quantization | File Size | VRAM Usage |
|-------|-------------|-----------|------------|
| SmolLM2-1.7B | Q8_0 | 1.7 GB | ~1.7 GB |
| LFM2.5-1.2B | Q8_0 | 1.2 GB | ~1.2 GB |
| 7B model | Q4_K_M | ~4.4 GB | ~4.7 GB |
| 8B model | Q4_K_M | ~4.7 GB | ~5.0 GB |

Rule of thumb: VRAM usage ≈ file size + 10-20% overhead for context and Vulkan buffers.

## Building llama.cpp with Vulkan

### Prerequisites

**Debian/Ubuntu**:
```bash
sudo apt install cmake build-essential libvulkan-dev vulkan-tools
```

**NixOS**:
```bash
nix-shell -p cmake gcc vulkan-headers vulkan-loader vulkan-tools shaderc
```

**Arch**:
```bash
sudo pacman -S cmake base-devel vulkan-headers vulkan-icd-loader vulkan-tools
```

### Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake -DGGML_VULKAN=ON ..
cmake --build . --config Release -j$(nproc)
```

The binary is at `build/bin/llama-server`. Verify Vulkan linkage:

```bash
ldd build/bin/llama-server | grep vulkan
# Should show: libvulkan.so.1 => ...
```

### Verify Vulkan Device

```bash
# Check llama-server sees your GPU
./build/bin/llama-server --version
# Should print: ggml_vulkan: Found 1 Vulkan devices: ... AMD ...
```

## AMD-Specific Notes

### RADV vs AMDVLK

Mesa RADV (the default on most Linux distributions) works well with llama.cpp. AMDVLK is AMD's proprietary Vulkan driver — it also works but RADV is generally preferred for compute workloads on consumer/pro cards.

### Polaris (WX 7100, RX 480/580)

- GCN 4th gen, no hardware ray tracing
- Vulkan compute works well via RADV
- `fp16: 0` in llama.cpp — no native FP16, but quantized inference (Q4/Q8) works at full speed
- No matrix cores — relies on shader compute for GEMM

### RDNA 1/2/3 (RX 5000/6000/7000 series)

- Better Vulkan compute performance than Polaris
- Some RDNA cards have better FP16 support
- RX 7900 XTX (24GB) is the AMD flagship for local inference

### Monitoring VRAM

```bash
# Real-time VRAM usage (AMD only)
cat /sys/class/drm/card1/device/mem_info_vram_used
cat /sys/class/drm/card1/device/mem_info_vram_total

# Or use radeontop
radeontop -d /dev/dri/card1
```

### NixOS Specifics

- Vulkan drivers are at `/run/opengl-driver/lib/` (RADV: `libvulkan_radeon.so`)
- ICD files at `/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json`
- `vulkaninfo` requires `vulkan-tools` in your environment
- Building llama.cpp: use `nix-shell` with Vulkan development packages (see above)
- llama-server built in nix-shell links against the Nix store Vulkan loader

## Performance Expectations

Based on WX 7100 (Polaris, 8GB GDDR5) benchmarks:

| Metric | 1.2B Q8 | 1.7B Q8 | 8B Q4_K_M |
|--------|---------|---------|-----------|
| Cold load | ~3-5s | ~5-8s | ~15-25s |
| Generation tok/s | ~100-130 | ~80-110 | ~15-25 |
| Classification | — | ~200-400ms | — |

These are rough estimates. Actual numbers depend on context length, prompt complexity, and system load. See `benchmarks/results/` for measured values.

## Other AMD GPUs

This router should work with any AMD GPU that supports Vulkan via Mesa RADV:

| GPU | VRAM | Expected Performance |
|-----|------|---------------------|
| RX 580 | 8GB | Similar to WX 7100 |
| RX 6600 | 8GB | ~1.5-2x faster than Polaris |
| RX 6700 XT | 12GB | Larger models + faster |
| RX 7800 XT | 16GB | Two specialists simultaneously |
| RX 7900 XTX | 24GB | All models loaded at once |

## NVIDIA Users

This router also works with NVIDIA GPUs via llama.cpp CUDA. Build with `-DGGML_CUDA=ON` instead of `-DGGML_VULKAN=ON`. NVIDIA generally offers better tok/s at equivalent VRAM sizes, but the routing architecture is the same.
