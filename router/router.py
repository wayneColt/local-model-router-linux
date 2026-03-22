"""Main model router — classifies queries and routes to specialists."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from router.classifier import Classifier
from router.specialist_manager import SpecialistManager
from router.config import RouterConfig

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of a routed query."""
    query: str
    category: str
    specialist: str
    response: str
    classify_time_ms: float
    load_time_ms: float
    inference_time_ms: float
    total_time_ms: float
    was_cold_load: bool
    tokens_generated: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query[:100],
            "category": self.category,
            "specialist": self.specialist,
            "response": self.response,
            "timing": {
                "classify_ms": round(self.classify_time_ms, 1),
                "load_ms": round(self.load_time_ms, 1),
                "inference_ms": round(self.inference_time_ms, 1),
                "total_ms": round(self.total_time_ms, 1),
            },
            "was_cold_load": self.was_cold_load,
            "tokens_generated": self.tokens_generated,
        }


class ModelRouter:
    """Routes queries through a hot classifier to cold-loaded specialists.

    Architecture:
        Query → Classifier (always hot) → Category → Load Specialist → Infer → Response

    The classifier is a small, fast model that stays loaded in VRAM.
    Specialists are larger models cold-loaded on demand — only one at a time.
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.classifier = Classifier(config)
        self.specialist_manager = SpecialistManager(config)
        self._history: list[RoutingResult] = []
        self._category_to_specialist = self._build_routing_table()

    def _build_routing_table(self) -> dict[str, str]:
        """Map classification categories to specialist names.

        Falls back to first available specialist if a category has no match.
        """
        table = {}
        specialists = list(self.config.specialists.keys())

        for name in specialists:
            # Match specialist name to category
            name_lower = name.lower()
            if "code" in name_lower:
                table["code"] = name
            elif "reason" in name_lower:
                table["reasoning"] = name
            elif "general" in name_lower:
                table["general"] = name

        # Fill gaps with first specialist as fallback
        fallback = specialists[0] if specialists else None
        for category in ("code", "reasoning", "general"):
            if category not in table and fallback:
                table[category] = fallback

        logger.info("Routing table: %s", table)
        return table

    def start(self) -> None:
        """Start the router (loads the classifier)."""
        issues = self.config.validate()
        if issues:
            for issue in issues:
                logger.warning("Config issue: %s", issue)

        self.classifier.start()
        logger.info("Router started. Classifier hot. %d specialists available.",
                     len(self.config.specialists))

    def stop(self) -> None:
        """Stop the router (unloads everything)."""
        self.specialist_manager.unload()
        self.classifier.stop()
        logger.info("Router stopped.")

    def route(self, query: str, system_prompt: Optional[str] = None,
              **kwargs) -> RoutingResult:
        """Route a query through classification to specialist inference.

        Args:
            query: The user's question or request
            system_prompt: Optional system prompt for the specialist
            **kwargs: Additional inference parameters

        Returns:
            RoutingResult with response and timing data
        """
        total_start = time.monotonic()

        # Step 1: Classify
        classify_start = time.monotonic()
        category = self.classifier.classify(query)
        classify_time = (time.monotonic() - classify_start) * 1000

        # Step 2: Resolve specialist
        specialist_name = self._category_to_specialist.get(category)
        if not specialist_name:
            raise RuntimeError(f"No specialist for category '{category}'")

        # Step 3: Load specialist (cold-load if needed)
        was_cold = self.specialist_manager.current != specialist_name
        load_start = time.monotonic()
        self.specialist_manager.load(specialist_name)
        load_time = (time.monotonic() - load_start) * 1000

        # Step 4: Infer
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        infer_start = time.monotonic()
        response = self.specialist_manager.infer(messages, **kwargs)
        infer_time = (time.monotonic() - infer_start) * 1000

        total_time = (time.monotonic() - total_start) * 1000

        # Extract response text and token count
        text = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("completion_tokens", 0)

        result = RoutingResult(
            query=query,
            category=category,
            specialist=specialist_name,
            response=text,
            classify_time_ms=classify_time,
            load_time_ms=load_time,
            inference_time_ms=infer_time,
            total_time_ms=total_time,
            was_cold_load=was_cold,
            tokens_generated=tokens,
        )

        self._history.append(result)
        logger.info(
            "Routed → %s via %s | classify=%.0fms load=%.0fms infer=%.0fms total=%.0fms",
            specialist_name, category,
            classify_time, load_time, infer_time, total_time,
        )

        return result

    def get_stats(self) -> dict:
        """Return routing statistics."""
        if not self._history:
            return {"total_queries": 0}

        categories = {}
        for r in self._history:
            categories[r.category] = categories.get(r.category, 0) + 1

        cold_loads = sum(1 for r in self._history if r.was_cold_load)
        avg_classify = sum(r.classify_time_ms for r in self._history) / len(self._history)
        avg_infer = sum(r.inference_time_ms for r in self._history) / len(self._history)
        avg_total = sum(r.total_time_ms for r in self._history) / len(self._history)

        return {
            "total_queries": len(self._history),
            "categories": categories,
            "cold_loads": cold_loads,
            "hot_hits": len(self._history) - cold_loads,
            "avg_classify_ms": round(avg_classify, 1),
            "avg_inference_ms": round(avg_infer, 1),
            "avg_total_ms": round(avg_total, 1),
            "specialist_load_times": self.specialist_manager.get_load_times(),
        }

    def get_history(self) -> list[dict]:
        """Return routing history as list of dicts."""
        return [r.to_dict() for r in self._history]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
