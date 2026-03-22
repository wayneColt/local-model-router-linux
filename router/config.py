"""Configuration for the local model router."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelSpec:
    """Specification for a single model."""
    name: str
    path: str
    vram_mb: int  # Approximate VRAM usage in MB
    description: str = ""
    context_size: int = 2048
    gpu_layers: int = 99  # -1 or 99 = all layers on GPU


@dataclass
class RouterConfig:
    """Configuration for the model router."""

    # llama-server binary path
    llama_server_path: str = "llama-server"

    # Classifier — always loaded
    classifier: ModelSpec = field(default_factory=lambda: ModelSpec(
        name="classifier",
        path="",
        vram_mb=1700,
        description="Hot classifier — always loaded, routes queries to specialists",
        context_size=512,
    ))

    # Specialist models — cold-loaded on demand
    specialists: dict[str, ModelSpec] = field(default_factory=dict)

    # Ports
    classifier_port: int = 9100
    specialist_port: int = 9101

    # GPU settings
    gpu_device: int = 0  # Vulkan device index
    vram_total_mb: int = 8192
    vram_reserved_mb: int = 2048  # Reserved for display compositor, other services

    # Timeouts
    server_startup_timeout: int = 60  # seconds
    inference_timeout: int = 120  # seconds
    health_check_interval: float = 0.5  # seconds

    @property
    def vram_available_mb(self) -> int:
        return self.vram_total_mb - self.vram_reserved_mb

    def validate(self) -> list[str]:
        """Validate configuration, return list of issues."""
        issues = []

        if not self.classifier.path or not Path(self.classifier.path).exists():
            issues.append(f"Classifier model not found: {self.classifier.path}")

        for name, spec in self.specialists.items():
            if not spec.path or not Path(spec.path).exists():
                issues.append(f"Specialist '{name}' model not found: {spec.path}")

            budget = self.vram_available_mb - self.classifier.vram_mb
            if spec.vram_mb > budget:
                issues.append(
                    f"Specialist '{name}' ({spec.vram_mb}MB) exceeds budget "
                    f"({budget}MB = {self.vram_available_mb}MB available - "
                    f"{self.classifier.vram_mb}MB classifier)"
                )

        return issues

    @classmethod
    def from_file(cls, path: str) -> "RouterConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        config = cls()
        config.llama_server_path = data.get("llama_server_path", config.llama_server_path)
        config.classifier_port = data.get("classifier_port", config.classifier_port)
        config.specialist_port = data.get("specialist_port", config.specialist_port)
        config.gpu_device = data.get("gpu_device", config.gpu_device)
        config.vram_total_mb = data.get("vram_total_mb", config.vram_total_mb)
        config.vram_reserved_mb = data.get("vram_reserved_mb", config.vram_reserved_mb)
        config.server_startup_timeout = data.get("server_startup_timeout", config.server_startup_timeout)
        config.inference_timeout = data.get("inference_timeout", config.inference_timeout)

        if "classifier" in data:
            c = data["classifier"]
            config.classifier = ModelSpec(
                name="classifier",
                path=os.path.expanduser(c.get("path", "")),
                vram_mb=c.get("vram_mb", 1700),
                description=c.get("description", ""),
                context_size=c.get("context_size", 512),
                gpu_layers=c.get("gpu_layers", 99),
            )

        if "specialists" in data:
            for name, s in data["specialists"].items():
                config.specialists[name] = ModelSpec(
                    name=name,
                    path=os.path.expanduser(s.get("path", "")),
                    vram_mb=s.get("vram_mb", 2000),
                    description=s.get("description", ""),
                    context_size=s.get("context_size", 2048),
                    gpu_layers=s.get("gpu_layers", 99),
                )

        return config

    def to_dict(self) -> dict:
        """Serialize config to dict."""
        return {
            "llama_server_path": self.llama_server_path,
            "classifier_port": self.classifier_port,
            "specialist_port": self.specialist_port,
            "gpu_device": self.gpu_device,
            "vram_total_mb": self.vram_total_mb,
            "vram_reserved_mb": self.vram_reserved_mb,
            "server_startup_timeout": self.server_startup_timeout,
            "inference_timeout": self.inference_timeout,
            "classifier": {
                "path": self.classifier.path,
                "vram_mb": self.classifier.vram_mb,
                "description": self.classifier.description,
                "context_size": self.classifier.context_size,
                "gpu_layers": self.classifier.gpu_layers,
            },
            "specialists": {
                name: {
                    "path": spec.path,
                    "vram_mb": spec.vram_mb,
                    "description": spec.description,
                    "context_size": spec.context_size,
                    "gpu_layers": spec.gpu_layers,
                }
                for name, spec in self.specialists.items()
            },
        }
