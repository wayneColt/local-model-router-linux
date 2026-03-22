"""Hot classifier — always loaded, routes queries to the right specialist."""

import json
import logging
import subprocess
import time
import urllib.request
import urllib.error
from typing import Optional

from router.config import RouterConfig

logger = logging.getLogger(__name__)

# Classification prompt — instruct the small model to categorize queries
CLASSIFY_SYSTEM_PROMPT = """You are a query classifier. Categorize the user's query into exactly one category.

Categories:
- code: Programming, debugging, code generation, scripting, technical implementation
- reasoning: Analysis, math, logic, planning, comparison, explanation of complex topics
- general: Conversation, creative writing, simple questions, summaries, translations

Respond with ONLY the category name. Nothing else."""


class Classifier:
    """Hot classifier using a small always-loaded model via llama-server."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._base_url = f"http://127.0.0.1:{config.classifier_port}"

    @property
    def is_running(self) -> bool:
        """Check if the classifier server is running and healthy."""
        if self._process is None or self._process.poll() is not None:
            return False
        try:
            req = urllib.request.Request(f"{self._base_url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def start(self) -> None:
        """Start the classifier llama-server process."""
        if self.is_running:
            logger.info("Classifier already running on port %d", self.config.classifier_port)
            return

        # Kill any existing process
        self.stop()

        cmd = [
            self.config.llama_server_path,
            "--model", self.config.classifier.path,
            "--port", str(self.config.classifier_port),
            "--host", "127.0.0.1",
            "--ctx-size", str(self.config.classifier.context_size),
            "--n-gpu-layers", str(self.config.classifier.gpu_layers),
            "--threads", "4",
            "--log-disable",
        ]

        logger.info("Starting classifier: %s", " ".join(cmd))
        env = dict(subprocess.os.environ)
        env["GGML_VK_DEVICE"] = str(self.config.gpu_device)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for server to become healthy
        deadline = time.monotonic() + self.config.server_startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(
                    f"Classifier process died during startup (exit {self._process.returncode}): "
                    f"{stderr[:500]}"
                )
            if self.is_running:
                logger.info("Classifier healthy on port %d", self.config.classifier_port)
                return
            time.sleep(self.config.health_check_interval)

        self.stop()
        raise TimeoutError(
            f"Classifier failed to start within {self.config.server_startup_timeout}s"
        )

    def stop(self) -> None:
        """Stop the classifier server."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            logger.info("Classifier stopped")

    def classify(self, query: str) -> str:
        """Classify a query into a specialist category.

        Returns one of: 'code', 'reasoning', 'general'
        """
        if not self.is_running:
            raise RuntimeError("Classifier is not running. Call start() first.")

        payload = json.dumps({
            "messages": [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 0.0,
            "max_tokens": 10,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.inference_timeout) as resp:
                data = json.loads(resp.read())
                raw = data["choices"][0]["message"]["content"].strip().lower()
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            logger.error("Classification failed: %s", e)
            return "general"  # Fallback

        # Normalize response to valid category
        for category in ("code", "reasoning", "general"):
            if category in raw:
                return category

        logger.warning("Unexpected classification '%s', falling back to 'general'", raw)
        return "general"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
