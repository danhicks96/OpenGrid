"""
OpenGrid worker loop — polls for inference jobs and returns results.
Uses the local vLLM/llama.cpp backend loaded by the opengrid daemon.
"""
import time
import requests
from requests.adapters import HTTPAdapter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ORCHESTRATOR  = "http://193.23.160.39:8080"
LOCAL         = "http://127.0.0.1:8080"
NODE_ID       = "node-32e70945447dae56"
AUTH          = {"Authorization": "Bearer dev-key-change-me"}
POLL_INTERVAL = 2.0   # seconds between polls when idle


def _session():
    """Fresh session per request — avoids stale connection pool timeouts."""
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=0))
    return s


def run_inference(model_id: str, prompt: str, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model_id or "tinyllama",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    log.info("  → sending to local inference (model=%s, max_tokens=%d)", model_id, max_tokens)
    with _session() as s:
        r = s.post(f"{LOCAL}/v1/chat/completions/local", json=payload, headers=AUTH, timeout=120)
    log.info("  → local response: %d", r.status_code)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def post_result(job_id: str, output_text: str, tokens: int, latency_ms: float, error: str = ""):
    payload = {
        "job_id": job_id,
        "node_id": NODE_ID,
        "output_text": output_text,
        "tokens_generated": tokens,
        "latency_ms": latency_ms,
    }
    if error:
        payload["error"] = error
    with _session() as s:
        r = s.post(f"{ORCHESTRATOR}/v1/work/result", headers=AUTH, json=payload, timeout=30)
    log.info("  → result POST: %d %s", r.status_code, r.text[:80])
    r.raise_for_status()


def poll_and_work():
    url = f"{ORCHESTRATOR}/v1/work/poll?node_id={NODE_ID}"
    with _session() as s:
        r = s.get(url, headers=AUTH, timeout=10)
    r.raise_for_status()
    job = r.json()

    if not job.get("has_work"):
        return False

    job_id   = job["job_id"]
    model_id = job["model_id"]
    prompt   = job["prompt"]
    log.info("Job %s | model=%s | %d chars", job_id, model_id, len(prompt))

    t0 = time.time()
    try:
        output_text    = run_inference(model_id, prompt, job["max_tokens"], job["temperature"])
        tokens         = len(output_text.split())
        latency_ms     = (time.time() - t0) * 1000
        post_result(job_id, output_text, tokens, latency_ms)
        log.info("Job %s done | tokens=%d | latency=%.0fms", job_id, tokens, latency_ms)
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        log.error("Job %s failed: %s", job_id, e)
        try:
            post_result(job_id, "", 0, latency_ms, error=str(e))
        except Exception as pe:
            log.error("Failed to POST error result: %s", pe)

    return True


if __name__ == "__main__":
    log.info("Worker loop starting — node %s", NODE_ID)
    while True:
        try:
            had_work = poll_and_work()
            if not had_work:
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as e:
            log.warning("Poll error: %s — retrying in 5s", e)
            time.sleep(5)
