"""Compare Ollama models on the real correction pipeline (prompt + validation).

Usage:
    python bench/run_bench.py --models qwen3.5:9b-mlx,gemma4:26b-mlx -v

Metrics per model:
- accuracy  : output matches one of the acceptable answers
- clean kept: already-correct inputs left untouched (over-editing is the worst UX)
- rejected  : chunks the safety validator threw away (model tried to rewrite)
- latency   : wall time per case after a warm-up call (model already in memory)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grammar_fix.config import load_config  # noqa: E402
from grammar_fix.core import correct_text  # noqa: E402
from grammar_fix.ollama_client import OllamaClient, OllamaError  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_model(client: OllamaClient, cases: list[dict], verbose: bool) -> dict:
    client.warm_up()
    client.correct("warm up the prompt cache")
    hits = clean_total = clean_kept = rejected = 0
    latencies = []
    for case in cases:
        start = time.perf_counter()
        result = correct_text(case["input"], client.correct)
        latencies.append(time.perf_counter() - start)
        out = result.text
        rejected += result.rejected_count
        ok = out in case["expected"]
        hits += ok
        if case["input"] in case["expected"]:
            clean_total += 1
            clean_kept += out == case["input"]
        if verbose and not ok:
            print(f"  ✗ in : {case['input']!r}\n    out: {out!r}")
            for chunk in result.chunks:
                if chunk.status.startswith("rejected"):
                    print(f"    ({chunk.status})")
    return {
        "accuracy": f"{hits}/{len(cases)}",
        "clean_kept": f"{clean_kept}/{clean_total}",
        "rejected": rejected,
        "avg_s": round(statistics.mean(latencies), 2),
        "p95_s": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
    }


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=cfg.model, help="comma-separated Ollama model names")
    parser.add_argument("--cases", default=str(ROOT / "bench" / "cases.jsonl"))
    parser.add_argument("-v", "--verbose", action="store_true", help="print failed cases")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"== {model}")
        client = OllamaClient(cfg.ollama_url, model, cfg.timeout_s, keep_alive="5m")
        try:
            rows.append((model, run_model(client, cases, args.verbose)))
        except OllamaError as e:
            print(f"  error: {e}")

    print(f"\n{'model':32} {'accuracy':>9} {'clean kept':>11} {'rejected':>9} {'avg s':>7} {'p95 s':>7}")
    for model, r in rows:
        print(
            f"{model:32} {r['accuracy']:>9} {r['clean_kept']:>11} {r['rejected']:>9} "
            f"{r['avg_s']:>7} {r['p95_s']:>7}"
        )


if __name__ == "__main__":
    main()
