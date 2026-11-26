"""Compare Ollama models on the real correction pipeline (prompt + validation).

Usage:
    python bench/run_bench.py --models gemma4:12b-mlx,qwen3.5:9b-mlx -v
    python bench/run_bench.py --models gemma4:12b-mlx --langs pl
    python bench/run_bench.py --models gemma4:12b-mlx --langs pl --prompt-lang en

Cases live in bench/cases.jsonl: {"input", "expected": [...], "lang"?: "en"|"pl",
"no_diacritics"?: true}. A missing "lang" means "en".

--prompt-lang:
  auto  each case uses the prompt of its own language (default)
  en    every case goes through the English prompt; for non-English cases this
        shows what the current app does to such text (ideally: leaves it alone)

Metrics per model and language:
- accuracy  : output matches one of the accepted answers
- clean kept: already-correct inputs left untouched (over-editing is the worst UX)
- untouched : output identical to input (the key number for --prompt-lang en)
- diacr.    : of the inputs written without Polish diacritics, how many got them added
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
from grammar_fix.prompts import LANGUAGES  # noqa: E402

POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def load_cases(path: Path, langs: set[str] | None) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        case.setdefault("lang", "en")
    return [c for c in cases if langs is None or c["lang"] in langs]


def _new_stats() -> dict:
    return {"n": 0, "hits": 0, "clean_total": 0, "clean_kept": 0, "untouched": 0,
            "diac_total": 0, "diac_added": 0, "rejected": 0, "latencies": []}


def run_model(client: OllamaClient, cases: list[dict], prompt_lang: str, verbose: bool) -> dict[str, dict]:
    client.warm_up()
    for lang in {c["lang"] if prompt_lang == "auto" else prompt_lang for c in cases}:
        client.correct("warm up the prompt cache", lang)  # each language prefix is cached separately

    stats: dict[str, dict] = {}
    for case in cases:
        lang = case["lang"]
        use_lang = lang if prompt_lang == "auto" else prompt_lang
        s = stats.setdefault(lang, _new_stats())

        start = time.perf_counter()
        result = correct_text(case["input"], lambda text: client.correct(text, use_lang))
        s["latencies"].append(time.perf_counter() - start)

        out = result.text
        ok = out in case["expected"]
        s["n"] += 1
        s["hits"] += ok
        s["untouched"] += out == case["input"]
        s["rejected"] += result.rejected_count
        if case["input"] in case["expected"] and not case.get("no_diacritics"):
            s["clean_total"] += 1
            s["clean_kept"] += out == case["input"]
        if case.get("no_diacritics"):
            s["diac_total"] += 1
            s["diac_added"] += any(ch in POLISH_DIACRITICS for ch in out)

        if verbose and not ok:
            print(f"  ✗ [{lang}] in : {case['input']!r}\n         out: {out!r}")
            for chunk in result.chunks:
                if chunk.status.startswith("rejected"):
                    print(f"         ({chunk.status})")
    return stats


def _row(model: str, lang: str, s: dict) -> str:
    lat = sorted(s["latencies"])
    p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
    diac = f"{s['diac_added']}/{s['diac_total']}" if s["diac_total"] else "-"
    return (
        f"{model:28} {lang:4} {s['hits']:>3}/{s['n']:<3} {s['clean_kept']:>5}/{s['clean_total']:<3} "
        f"{s['untouched']:>5}/{s['n']:<3} {diac:>6} {s['rejected']:>8} "
        f"{statistics.mean(lat):>6.2f} {p95:>6.2f}"
    )


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=cfg.model, help="comma-separated Ollama model names")
    parser.add_argument("--cases", default=str(ROOT / "bench" / "cases.jsonl"))
    parser.add_argument("--langs", default="", help="only these case languages, e.g. 'pl' or 'en,pl'")
    parser.add_argument("--prompt-lang", default="auto", choices=["auto", *LANGUAGES],
                        help="'auto' = prompt per case language; 'en' = everything through the English prompt")
    parser.add_argument("-v", "--verbose", action="store_true", help="print failed cases")
    args = parser.parse_args()

    langs = {x.strip() for x in args.langs.split(",") if x.strip()} or None
    cases = load_cases(Path(args.cases), langs)
    print(f"{len(cases)} cases, prompt: {args.prompt_lang}")

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"== {model}")
        client = OllamaClient(cfg.ollama_url, model, cfg.timeout_s, keep_alive="5m")
        try:
            rows.append((model, run_model(client, cases, args.prompt_lang, args.verbose)))
        except OllamaError as e:
            print(f"  error: {e}")

    print(f"\n{'model':28} {'lang':4} {'accuracy':>7} {'clean kept':>9} {'untouched':>9} "
          f"{'diacr.':>6} {'rejected':>8} {'avg s':>6} {'p95 s':>6}")
    for model, per_lang in rows:
        for lang in sorted(per_lang):
            print(_row(model, lang, per_lang[lang]))


if __name__ == "__main__":
    main()
