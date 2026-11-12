"""Async runner: rate-limited, retrying, resumable, with a cost estimate and a hard budget cap.

Files
  data/scenarios/{run}/{category}.jsonl   generated scenarios (deterministic)
  data/logs/{run}/{model}.jsonl           one RawCall per (scenario, model), append-only
  data/ledger.csv                         cumulative spend across all runs
"""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

from . import prompts
from .providers import KEY_ENV, PROVIDERS, ProviderError, QuotaExhausted, RetryableError
from .scenarios import generate
from .schemas import ROOT, Catalog, RawCall, Scenario

MAX_ATTEMPTS = 6
EST_OUTPUT_TOKENS = 120  # JSON choice + one-sentence reason; refined from the pilot
CHARS_PER_TOKEN = 3.5

LEDGER = ROOT / "data/ledger.csv"


# --------------------------------------------------------------------------- config
@dataclass
class RunConfig:
    name: str
    experiment: str
    categories: list[str]
    models: list[str]
    n_scenarios: int
    seed: int
    prompt: str = "default"
    n_per_model: dict | None = None  # optional per-model override, e.g. {claude-haiku-5-5: 500}

    def n_for(self, model: str) -> int:
        return (self.n_per_model or {}).get(model, self.n_scenarios)

    @property
    def n_max(self) -> int:
        return max([self.n_scenarios, *(self.n_per_model or {}).values()])

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


def load_models() -> dict:
    with open(ROOT / "configs/models.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- scenarios
def build_scenarios(cfg: RunConfig) -> dict[str, tuple[Catalog, list[Scenario]]]:
    out = {}
    for cat_key in cfg.categories:
        cat = Catalog.load(cat_key)
        scen = generate(cfg.experiment, cat, cfg.seed, cfg.n_max, prefix=cfg.name)
        path = ROOT / "data/scenarios" / cfg.name / f"{cat_key}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in scen:
                f.write(s.model_dump_json() + "\n")
        out[cat_key] = (cat, scen)
    return out


def log_path(cfg: RunConfig, model: str) -> Path:
    return ROOT / "data/logs" / cfg.name / f"{model}.jsonl"


def done_ids(path: Path) -> set[str]:
    """Scenario IDs that already have a successful (non-error) call; errors are retried on resume."""
    if not path.exists():
        return set()
    ids = set()
    for line in open(path):
        c = RawCall.model_validate_json(line)
        if not c.error:
            ids.add(c.scenario_id)
    return ids


# --------------------------------------------------------------------------- cost
def spent_usd() -> float:
    """Total spend across all runs, from the call logs (the source of truth; survives interrupted runs)."""
    total = 0.0
    for path in (ROOT / "data/logs").glob("*/*.jsonl"):
        for line in open(path):
            total += RawCall.model_validate_json(line).cost_usd
    return total


def todo_for(cfg: RunConfig, model: str, data: dict) -> list[tuple[Catalog, Scenario]]:
    """The first n_for(model) scenarios per category (shared across models), minus those already done."""
    done = done_ids(log_path(cfg, model))
    n = cfg.n_for(model)
    return [(cat, s) for cat, scen in data.values() for s in scen[:n] if s.scenario_id not in done]


def calibration(model: str) -> tuple[float, float]:
    """(chars per input token, output+thinking tokens per call) from this model's past calls, else defaults."""
    chars = tok_in = tok_out = n = 0
    for path in (ROOT / "data/logs").glob(f"*/{model}.jsonl"):
        for line in open(path):
            c = RawCall.model_validate_json(line)
            if c.error or not c.tokens_in:
                continue
            chars += len(c.prompt)
            tok_in += c.tokens_in
            tok_out += (c.tokens_out or 0) + (c.tokens_thinking or 0)
            n += 1
    if n < 10:
        return CHARS_PER_TOKEN, EST_OUTPUT_TOKENS
    return chars / tok_in, tok_out / n


def estimate(cfg: RunConfig) -> list[dict]:
    models = load_models()["models"]
    data = build_scenarios(cfg)
    rows = []
    for m in cfg.models:
        spec = models[m]
        todo = todo_for(cfg, m, data)
        cpt, out_per_call = calibration(m)
        tin = sum(len(prompts.render(s, cat, cfg.prompt)) / cpt for cat, s in todo)
        tout = out_per_call * len(todo)
        paid = (tin * spec["price_in"] + tout * spec["price_out"]) / 1e6
        rows.append({
            "model": m,
            "calls": len(todo),
            "tokens_in": int(tin),
            "tokens_out": int(tout),
            "paid_equivalent_usd": paid,
            "cost_usd": 0.0 if spec.get("free_tier") else paid,
            "free_tier": bool(spec.get("free_tier")),
            "minutes_at_rpm": len(todo) / spec.get("rpm", 10),
        })
    return rows


def print_estimate(cfg: RunConfig) -> float:
    rows = estimate(cfg)
    cap = load_models()["budget"]["cap_usd"]
    spent = spent_usd()
    total = sum(r["cost_usd"] for r in rows)
    per = ", ".join(f"{m}: {cfg.n_for(m)}" for m in cfg.models)
    print(f"\nRun '{cfg.name}' ({cfg.experiment}), scenarios per category -> {per}")
    print(f"{'model':24} {'calls':>6} {'tok_in':>9} {'tok_out':>8} {'cost $':>8} {'paid-equiv $':>12} {'~min':>5}")
    for r in rows:
        print(f"{r['model']:24} {r['calls']:6d} {r['tokens_in']:9d} {r['tokens_out']:8d} "
              f"{r['cost_usd']:8.4f} {r['paid_equivalent_usd']:12.4f} {r['minutes_at_rpm']:5.1f}"
              + ("  (free tier)" if r["free_tier"] else ""))
    print(f"Estimated cost: ${total:.4f} | spent so far: ${spent:.4f} | cap: ${cap:.2f} "
          f"| after run: ${spent + total:.4f}")
    return total


# --------------------------------------------------------------------------- execution
class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.next_t = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                await asyncio.sleep(self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval


async def _one(client, spec, model, cat, scen, cfg, limiter, out_f, lock, stats):
    if stats.get("quota_exhausted"):
        return  # not logged: picked up again on the next (resumed) run
    prompt = prompts.render(scen, cat, cfg.prompt)
    fn = PROVIDERS[spec["provider"]]
    attempts, err, res, t0 = 0, None, None, time.monotonic()
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        await limiter.wait()
        t0 = time.monotonic()
        try:
            res = await fn(client, spec["model_id"], prompt, spec["params"])
            err = None
            break
        except QuotaExhausted as e:
            err = str(e)
            if not stats.get("quota_exhausted"):
                print(f"  [{model}] daily quota exhausted: skipping its remaining calls (resume tomorrow)")
            stats["quota_exhausted"] = True
            break
        except RetryableError as e:
            err = str(e)
            await asyncio.sleep((e.retry_after or min(60, 2 ** attempts)) + random.random())
        except (httpx.TimeoutException, httpx.TransportError) as e:
            err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(min(60, 2 ** attempts) + random.random())
        except ProviderError as e:
            err = str(e)
            break
    latency = int((time.monotonic() - t0) * 1000)
    tin = (res or {}).get("tokens_in") or 0
    tout = ((res or {}).get("tokens_out") or 0) + ((res or {}).get("tokens_thinking") or 0)
    paid = (tin * spec["price_in"] + tout * spec["price_out"]) / 1e6
    call = RawCall(
        call_id=uuid.uuid4().hex[:12],
        scenario_id=scen.scenario_id,
        model=model,
        model_id=spec["model_id"],
        model_version=(res or {}).get("model_version"),
        provider=spec["provider"],
        params=spec["params"],
        prompt=prompt,
        prompt_hash=prompts.prompt_hash(prompt),
        response_text=(res or {}).get("text"),
        finish_reason=(res or {}).get("finish_reason"),
        tokens_in=(res or {}).get("tokens_in"),
        tokens_out=(res or {}).get("tokens_out"),
        tokens_thinking=(res or {}).get("tokens_thinking"),
        cost_usd=0.0 if spec.get("free_tier") else paid,
        cost_usd_paid_equivalent=paid,
        latency_ms=latency,
        attempts=attempts,
        error=err,
        ts=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    async with lock:
        out_f.write(call.model_dump_json() + "\n")
        out_f.flush()
        stats["n"] += 1
        stats["err"] += bool(err)
        stats["cost"] += call.cost_usd
        if stats["n"] % 10 == 0 or stats["n"] == stats["total"]:
            print(f"  [{model}] {stats['n']}/{stats['total']} done, errors {stats['err']}, cost ${stats['cost']:.4f}")


async def _run_model(cfg, model, spec, data, concurrency=4):
    path = log_path(cfg, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = todo_for(cfg, model, data)
    stats = {"n": 0, "err": 0, "cost": 0.0, "total": len(todo)}
    if not todo:
        print(f"  [{model}] nothing to do")
        return stats
    limiter, lock, sem = RateLimiter(spec.get("rpm", 10)), asyncio.Lock(), asyncio.Semaphore(concurrency)

    async def guarded(cat, s):
        async with sem:
            await _one(client, spec, model, cat, s, cfg, limiter, out_f, lock, stats)

    async with httpx.AsyncClient(timeout=90) as client:
        with open(path, "a") as out_f:
            await asyncio.gather(*(guarded(cat, s) for cat, s in todo))
    return stats


def run(cfg: RunConfig, yes: bool = False) -> None:
    load_dotenv(ROOT / ".env")
    models = load_models()
    for m in cfg.models:
        env = KEY_ENV[models["models"][m]["provider"]]
        if not os.environ.get(env):
            raise SystemExit(f"Missing {env} in .env for model {m}")

    total = print_estimate(cfg)
    cap = models["budget"]["cap_usd"]
    if spent_usd() + total > cap:
        raise SystemExit(f"REFUSED: this run would push spend over the ${cap:.2f} cap.")
    if not yes:
        raise SystemExit("Estimate only. Re-run with --yes to execute.")

    data = build_scenarios(cfg)

    async def all_models():
        return await asyncio.gather(*(_run_model(cfg, m, models["models"][m], data) for m in cfg.models))

    results = asyncio.run(all_models())
    actual = sum(r["cost"] for r in results)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "run", "models", "calls", "errors", "cost_usd"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), cfg.name, "|".join(cfg.models),
                    sum(r["n"] for r in results), sum(r["err"] for r in results), f"{actual:.6f}"])
    print(f"Done. Actual cost ${actual:.4f}; total spent ${spent_usd():.4f}")
