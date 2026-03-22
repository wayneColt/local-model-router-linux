#!/usr/bin/env python3
"""Batch routing demo — route a set of queries and display routing decisions."""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.config import RouterConfig
from router.router import ModelRouter

logging.basicConfig(level=logging.WARNING)

# Sample queries spanning all categories
SAMPLE_QUERIES = [
    "Write a Python function to sort a list of dictionaries by a key",
    "What are the main differences between TCP and UDP?",
    "Tell me a short story about a robot learning to paint",
    "Debug this code: for i in range(10) print(i)",
    "Explain the tradeoffs between microservices and monoliths",
    "How do I set up SSH key authentication on Linux?",
    "What is the capital of France?",
    "Implement a binary search tree in Python with insert and search methods",
    "Compare and contrast REST and GraphQL APIs for a mobile application",
    "Summarize the concept of supply and demand in economics",
]


def main():
    config_path = os.environ.get("ROUTER_CONFIG", "config.json")
    config = RouterConfig.from_file(config_path)

    print("=" * 70)
    print("Batch Routing Demo")
    print(f"Routing {len(SAMPLE_QUERIES)} queries through classifier → specialist")
    print("=" * 70)

    with ModelRouter(config) as router:
        results = []
        for i, query in enumerate(SAMPLE_QUERIES, 1):
            print(f"\n[{i}/{len(SAMPLE_QUERIES)}] {query[:60]}...")
            try:
                result = router.route(query, max_tokens=256)
                results.append(result.to_dict())
                cold = "COLD" if result.was_cold_load else "HOT"
                print(f"  → {result.category} → {result.specialist} "
                      f"[{cold}] {result.total_time_ms:.0f}ms")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"query": query[:60], "error": str(e)})

        print("\n" + "=" * 70)
        print("ROUTING SUMMARY")
        print("=" * 70)
        stats = router.get_stats()
        print(json.dumps(stats, indent=2))

        # Save results
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks", "results", "batch_routing_results.json"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"queries": results, "stats": stats}, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
