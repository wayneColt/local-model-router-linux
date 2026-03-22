"""Specialist manager — cold-load and unload specialist models on demand."""

import json
import logging
import subprocess
import time
import urllib.request
import urllib.error
from typing import Optional

from router.config import RouterConfig, ModelSpec

logger = logging.getLogger(__name__)


class SpecialistManager:
    """Manages cold-loading specialist models via llama-server.

    Only one specialist is loaded at a time. Loading a new specialist
    automatically unloads the current one to free VRAM.
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._current_specialist: Optional[str] = None
        self._base_url = f"http://127.0.0.1:{config.specialist_port}"
        self._load_times: dict[str, float] = {}

    @property
    def current(self) -> Optional[str]:
        """Name of the currently loaded specialist, or None."""
        return self._current_specialist

    @property
    def is_running(self) -> bool:
        """Check if the specialist server is running and healthy."""
        if self._process is None or self._process.poll() is not None:
            return False
        try:
            req = urllib.request.Request(f"{self._base_url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def load(self, specialist_name: str) -> float:
        """Cold-load a specialist model. Returns load time in seconds.

        If the requested specialist is already loaded, returns 0.0.
        """
        if specialist_name not in self.config.specialists:
            available = list(self.config.specialists.keys())
            raise ValueError(
                f"Unknown specialist '{specialist_name}'. Available: {available}"
            )

        if self._current_specialist == specialist_name and self.is_running:
            logger.info("Specialist '%s' already loaded", specialist_name)
            return 0.0

        # Unload current specialist first
        self.unload()

        spec = self.config.specialists[specialist_name]
        logger.info("Cold-loading specialist '%s' (%s, ~%dMB VRAM)",
                     specialist_name, spec.path, spec.vram_mb)

        start_time = time.monotonic()

        cmd = [
            self.config.llama_server_path,
            "--model", spec.path,
            "--port", str(self.config.specialist_port),
            "--host", "127.0.0.1",
            "--ctx-size", str(spec.context_size),
            "--n-gpu-layers", str(spec.gpu_layers),
            "--threads", "4",
            "--log-disable",
        ]

        env = dict(subprocess.os.environ)
        env["GGML_VK_DEVICE"] = str(self.config.gpu_device)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for healthy
        deadline = time.monotonic() + self.config.server_startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                returncode = self._process.returncode
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                self._process = None
                raise RuntimeError(
                    f"Specialist '{specialist_name}' died during startup "
                    f"(exit {returncode}): {stderr[:500]}"
                )
            if self.is_running:
                load_time = time.monotonic() - start_time
                self._current_specialist = specialist_name
                self._load_times[specialist_name] = load_time
                logger.info("Specialist '%s' loaded in %.1fs", specialist_name, load_time)
                return load_time
            time.sleep(self.config.health_check_interval)

        self.unload()
        raise TimeoutError(
            f"Specialist '{specialist_name}' failed to start within "
            f"{self.config.server_startup_timeout}s"
        )

    def unload(self) -> None:
        """Unload the current specialist to free VRAM."""
        if self._process is not None:
            logger.info("Unloading specialist '%s'", self._current_specialist)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            self._current_specialist = None
            # Brief pause for VRAM to be freed
            time.sleep(0.5)

    def infer(self, messages: list[dict], **kwargs) -> dict:
        """Send an inference request to the loaded specialist.

        Args:
            messages: OpenAI-format chat messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Full response dict from llama-server
        """
        if not self.is_running:
            raise RuntimeError("No specialist is loaded. Call load() first.")

        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "stream": False,
        }

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.config.inference_timeout) as resp:
            return json.loads(resp.read())

    def get_load_times(self) -> dict[str, float]:
        """Return recorded load times for all specialists."""
        return dict(self._load_times)

    def __del__(self):
        self.unload()
