#!/usr/bin/env python3
"""Benchmark suite for the local model router.

Measures:
- Classifier tok/s and classification latency
- Specialist cold-load times
- Specialist generation tok/s
- End-to-end routed query latency
- VRAM utilization at each stage
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.config import RouterConfig
from router.classifier import Classifier
from router.specialist_manager import SpecialistManager
from router.router import ModelRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_vram_usage() -> dict:
    """Read VRAM usage from sysfs (AMD GPUs)."""
    try:
        with open("/sys/class/drm/card1/device/mem_info_vram_used") as f:
            used = int(f.read().strip())
        with open("/sys/class/drm/card1/device/mem_info_vram_total") as f:
            total = int(f.read().strip())
        return {
            "used_mb": round(used / 1024 / 1024),
            "total_mb": round(total / 1024 / 1024),
            "free_mb": round((total - used) / 1024 / 1024),
            "utilization_pct": round(used / total * 100, 1),
        }
    except (FileNotFoundError, ValueError):
        return {"error": "Could not read VRAM info from sysfs"}


def get_gpu_info() -> dict:
    """Gather GPU information."""
    info = {"vram": get_vram_usage()}

    # Try vulkaninfo
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "deviceName" in line:
                    info["device_name"] = line.split("=")[-1].strip()
                elif "driverName" in line:
                    info["driver"] = line.split("=")[-1].strip()
                elif "apiVersion" in line and "api" not in info:
                    info["vulkan_api"] = line.split("=")[-1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def benchmark_classifier(config: RouterConfig, n_rounds: int = 5) -> dict:
    """Benchmark the classifier model."""
    logger.info("=== Benchmarking Classifier ===")
    results = {"model": os.path.basename(config.classifier.path)}

    vram_before = get_vram_usage()
    classifier = Classifier(config)

    start = time.monotonic()
    classifier.start()
    startup_time = time.monotonic() - start
    results["startup_seconds"] = round(startup_time, 2)

    vram_after = get_vram_usage()
    results["vram_delta_mb"] = vram_after.get("used_mb", 0) - vram_before.get("used_mb", 0)

    # Classification latency
    test_queries = [
        "Write a Python function to reverse a string",
        "Explain quantum computing in simple terms",
        "What is the weather like today?",
        "Debug this JavaScript: const x = [1,2,3; console.log(x)",
        "Compare supervised and unsupervised learning approaches",
    ]

    latencies = []
    categories = []
    for _ in range(n_rounds):
        for query in test_queries:
            t0 = time.monotonic()
            cat = classifier.classify(query)
            latencies.append((time.monotonic() - t0) * 1000)
            categories.append(cat)

    results["classification"] = {
        "queries_tested": len(latencies),
        "avg_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "categories_seen": list(set(categories)),
    }

    classifier.stop()
    return results


def benchmark_specialists(config: RouterConfig) -> dict:
    """Benchmark each specialist model: cold-load time + generation."""
    logger.info("=== Benchmarking Specialists ===")
    results = {}

    manager = SpecialistManager(config)

    for name, spec in config.specialists.items():
        logger.info("Benchmarking specialist: %s", name)
        vram_before = get_vram_usage()

        try:
            load_time = manager.load(name)

            vram_loaded = get_vram_usage()
            vram_delta = vram_loaded.get("used_mb", 0) - vram_before.get("used_mb", 0)

            # Generation benchmark
            prompt = "Explain in two sentences what a binary search tree is."
            gen_times = []
            token_counts = []

            for _ in range(3):
                t0 = time.monotonic()
                resp = manager.infer(
                    [{"role": "user", "content": prompt}],
                    max_tokens=128,
                    temperature=0.7,
                )
                gen_time = (time.monotonic() - t0) * 1000
                tokens = resp.get("usage", {}).get("completion_tokens", 0)
                gen_times.append(gen_time)
                token_counts.append(tokens)

            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0
            tok_per_sec = (avg_tokens / (avg_gen_time / 1000)) if avg_gen_time > 0 else 0

            results[name] = {
                "model": os.path.basename(spec.path),
                "file_size_mb": round(os.path.getsize(spec.path) / 1024 / 1024),
                "cold_load_seconds": round(load_time, 2),
                "vram_delta_mb": vram_delta,
                "generation": {
                    "avg_tokens": round(avg_tokens, 1),
                    "avg_time_ms": round(avg_gen_time, 1),
                    "tok_per_sec": round(tok_per_sec, 1),
                },
            }

            manager.unload()
            time.sleep(1)  # Allow VRAM to settle

        except Exception as e:
            logger.error("Failed to benchmark '%s': %s", name, e)
            results[name] = {"error": str(e)}
            manager.unload()

    return results


def benchmark_routing(config: RouterConfig) -> dict:
    """End-to-end routing benchmark."""
    logger.info("=== Benchmarking End-to-End Routing ===")

    queries = [
        ("Write a function to check if a number is prime", "code"),
        ("What are the main causes of inflation?", "reasoning"),
        ("Tell me a joke about programmers", "general"),
        ("Implement a linked list in Python", "code"),
        ("Compare REST and GraphQL", "reasoning"),
    ]

    results_list = []

    with ModelRouter(config) as router:
        for query, expected_cat in queries:
            try:
                result = router.route(query, max_tokens=128)
                results_list.append({
                    "query": query[:50],
                    "expected": expected_cat,
                    "actual": result.category,
                    "specialist": result.specialist,
                    "classify_ms": round(result.classify_time_ms, 1),
                    "load_ms": round(result.load_time_ms, 1),
                    "inference_ms": round(result.inference_time_ms, 1),
                    "total_ms": round(result.total_time_ms, 1),
                    "cold_load": result.was_cold_load,
                })
            except Exception as e:
                results_list.append({"query": query[:50], "error": str(e)})

        stats = router.get_stats()

    return {"queries": results_list, "stats": stats}


def main():
    config_path = os.environ.get("ROUTER_CONFIG", "config.json")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = RouterConfig.from_file(config_path)

    print("=" * 60)
    print("Local Model Router — Benchmark Suite")
    print(f"GPU: AMD (reading from sysfs)")
    print(f"Classifier: {os.path.basename(config.classifier.path)}")
    print(f"Specialists: {', '.join(config.specialists.keys())}")
    print("=" * 60)

    benchmark = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": get_gpu_info(),
        "vram_baseline": get_vram_usage(),
    }

    # Run benchmarks
    benchmark["classifier"] = benchmark_classifier(config)
    benchmark["specialists"] = benchmark_specialists(config)
    benchmark["routing"] = benchmark_routing(config)
    benchmark["vram_final"] = get_vram_usage()

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks", "results"
    )
    os.makedirs(output_dir, exist_ok=True)

    # GPU-specific filename
    gpu_name = benchmark["gpu"].get("device_name", "unknown_gpu")
    gpu_slug = gpu_name.lower().replace(" ", "_").replace("/", "_")[:30]
    output_path = os.path.join(output_dir, f"{gpu_slug}.json")

    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(json.dumps(benchmark, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
