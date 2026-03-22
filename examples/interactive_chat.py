#!/usr/bin/env python3
"""Interactive chat with the local model router.

Queries are classified and routed to the best specialist automatically.
Type 'quit' to exit, 'stats' to see routing statistics.
"""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.config import RouterConfig
from router.router import ModelRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def main():
    config_path = os.environ.get("ROUTER_CONFIG", "config.json")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Set ROUTER_CONFIG env var or run from repo root.")
        sys.exit(1)

    config = RouterConfig.from_file(config_path)
    issues = config.validate()
    if issues:
        print("Configuration warnings:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    print("=" * 60)
    print("Local Model Router — Interactive Chat")
    print(f"Classifier: {os.path.basename(config.classifier.path)}")
    print(f"Specialists: {', '.join(config.specialists.keys())}")
    print(f"VRAM budget: {config.vram_available_mb}MB")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'stats' for statistics")
    print()

    with ModelRouter(config) as router:
        while True:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not query:
                continue
            if query.lower() == "quit":
                break
            if query.lower() == "stats":
                stats = router.get_stats()
                print(json.dumps(stats, indent=2))
                continue

            try:
                result = router.route(query)
                print(f"\n[{result.category} → {result.specialist}]")
                print(f"  classify={result.classify_time_ms:.0f}ms "
                      f"load={result.load_time_ms:.0f}ms "
                      f"infer={result.inference_time_ms:.0f}ms "
                      f"total={result.total_time_ms:.0f}ms"
                      f"{' (cold)' if result.was_cold_load else ' (hot)'}")
                print(f"\n{result.response}")
            except Exception as e:
                print(f"\nError: {e}")

        print("\nFinal stats:")
        print(json.dumps(router.get_stats(), indent=2))


if __name__ == "__main__":
    main()
