"""
Ablation runner for parameter-golf.
Runs the baseline (or modified) train_gpt.py with different configs and
collects val_bpb at specified step checkpoints for comparison.

Usage:
    python3 ablation.py                              # Run baseline to 2000 steps
    python3 ablation.py --steps 1000                 # Run baseline to 1000 steps
    python3 ablation.py --sweep lr                   # Sweep learning rates
    python3 ablation.py --compare                    # Compare past runs
    python3 ablation.py --script sota_train_gpt.py --name sota_2k
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "ablation_results"
TB_DIR = REPO_ROOT / "tb_logs"


def parse_log_line(line: str) -> dict | None:
    """Parse a single log line into a metric dict."""
    val_match = re.match(
        r"step:(\d+)/\d+\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:([\d.]+)ms",
        line,
    )
    if val_match:
        return {
            "step": int(val_match.group(1)),
            "val_loss": float(val_match.group(2)),
            "val_bpb": float(val_match.group(3)),
            "train_time_ms": float(val_match.group(4)),
            "type": "val",
        }
    train_match = re.match(
        r"step:(\d+)/\d+\s+train_loss:([\d.]+)\s+train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms",
        line,
    )
    if train_match:
        return {
            "step": int(train_match.group(1)),
            "train_loss": float(train_match.group(2)),
            "train_time_ms": float(train_match.group(3)),
            "step_avg_ms": float(train_match.group(4)),
            "type": "train",
        }
    return None


class LiveTBWriter:
    """Watches a log file and streams metrics to tensorboard in real-time."""

    def __init__(self, log_path: Path, name: str):
        self.log_path = log_path
        self.writer = SummaryWriter(log_dir=str(TB_DIR / name))
        self.seen_lines = 0
        self._stop = threading.Event()
        self._prev_train_loss = None

    def _poll(self):
        while not self._stop.is_set():
            if not self.log_path.exists():
                self._stop.wait(1)
                continue
            lines = self.log_path.read_text().splitlines()
            for line in lines[self.seen_lines:]:
                entry = parse_log_line(line)
                if not entry:
                    continue
                time_s = int(entry["train_time_ms"] / 1000)
                if entry["type"] == "val":
                    self.writer.add_scalar("val/bpb", entry["val_bpb"], entry["step"])
                    self.writer.add_scalar("val/loss", entry["val_loss"], entry["step"])
                    self.writer.add_scalar("time/val_bpb", entry["val_bpb"], time_s)
                    self.writer.add_scalar("time/val_loss", entry["val_loss"], time_s)
                elif entry["type"] == "train":
                    self.writer.add_scalar("train/loss", entry["train_loss"], entry["step"])
                    self.writer.add_scalar("time/train_loss", entry["train_loss"], time_s)
                    if self._prev_train_loss is not None:
                        delta = entry["train_loss"] - self._prev_train_loss
                        self.writer.add_scalar("train/loss_delta", delta, entry["step"])
                    self._prev_train_loss = entry["train_loss"]
                    if "step_avg_ms" in entry:
                        self.writer.add_scalar("perf/step_avg_ms", entry["step_avg_ms"], entry["step"])
                self.writer.flush()
            self.seen_lines = len(lines)
            self._stop.wait(2)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.writer.close()


def run_config(
    name: str,
    env_overrides: dict[str, str],
    steps: int,
    val_every: int,
    script: str = "train_gpt.py",
) -> dict:
    """Run a single training config and return parsed results."""
    # Create result subfolder, clean stale data
    run_dir = RESULTS_DIR / name
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_run_dir = TB_DIR / name
    if tb_run_dir.exists():
        import shutil
        shutil.rmtree(tb_run_dir)
    log_file = REPO_ROOT / "logs" / f"{name}.txt"
    if log_file.exists():
        log_file.unlink()

    env = os.environ.copy()
    env.update({
        "ITERATIONS": str(steps),
        "VAL_LOSS_EVERY": str(val_every),
        "TRAIN_LOG_EVERY": "10",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMDOWN_ITERS": "0",  # flat LR; warmdown is same for all archs so skip it
        "RUN_ID": name,
    })
    env.update(env_overrides)

    print(f"\n{'='*60}")
    print(f"  ABLATION: {name}")
    print(f"  steps={steps}, val_every={val_every}")
    print(f"  script: {script}")
    if env_overrides:
        print(f"  overrides: {env_overrides}")
    print(f"{'='*60}\n")

    # Start live TB writer
    log_path = REPO_ROOT / "logs" / f"{name}.txt"
    tb_writer = LiveTBWriter(log_path, name)
    tb_writer.start()

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    # Stop TB writer
    tb_writer.stop()

    # Parse full log
    log_text = proc.stdout + "\n" + proc.stderr
    entries = []
    for line in log_text.splitlines():
        entry = parse_log_line(line)
        if entry:
            entries.append(entry)

    val_entries = [e for e in entries if e["type"] == "val"]
    final_bpb = val_entries[-1]["val_bpb"] if val_entries else None
    final_loss = val_entries[-1]["val_loss"] if val_entries else None

    result = {
        "name": name,
        "script": script,
        "overrides": env_overrides,
        "steps": steps,
        "val_every": val_every,
        "elapsed_seconds": elapsed,
        "final_val_bpb": final_bpb,
        "final_val_loss": final_loss,
        "val_entries": val_entries,
        "returncode": proc.returncode,
    }

    if proc.returncode != 0:
        result["error"] = proc.stderr[-2000:] if proc.stderr else "unknown error"
        print(f"  ERROR (rc={proc.returncode})")
        print(proc.stderr[-1000:])
    else:
        print(f"  Final BPB: {final_bpb:.4f}" if final_bpb else "  No val results found")
        print(f"  Elapsed: {elapsed:.1f}s")

    # Save result JSON + copy log to run dir
    result_path = run_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    if log_path.exists():
        import shutil
        shutil.copy2(log_path, run_dir / "train.log")
    print(f"  Saved: {run_dir}/")

    return result


def compare_results(results_dir: Path) -> None:
    """Print a comparison table of all ablation results."""
    results = []
    # Support both old flat .json and new subfolder/result.json
    for f in sorted(results_dir.glob("*/result.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))

    if not results:
        print("No results found.")
        return

    # Deduplicate by name
    seen = set()
    unique = []
    for r in results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)

    print(f"\n{'Name':<40} {'Steps':>6} {'BPB':>8} {'Loss':>8} {'Time':>8}")
    print("-" * 74)
    for r in sorted(unique, key=lambda x: x.get("final_val_bpb") or 99):
        bpb = f"{r['final_val_bpb']:.4f}" if r.get("final_val_bpb") else "FAIL"
        loss = f"{r['final_val_loss']:.4f}" if r.get("final_val_loss") else "-"
        time_s = f"{r['elapsed_seconds']:.0f}s" if r.get("elapsed_seconds") else "-"
        print(f"{r['name']:<40} {r['steps']:>6} {bpb:>8} {loss:>8} {time_s:>8}")


def build_sweep(sweep_type: str, steps: int, val_every: int) -> list[tuple[str, dict]]:
    """Build a list of (name, env_overrides) for a sweep."""
    if sweep_type == "lr":
        return [
            (f"lr_matrix_{lr}", {"MATRIX_LR": str(lr)})
            for lr in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
        ]
    elif sweep_type == "dim":
        return [
            (f"dim_{d}", {"MODEL_DIM": str(d), "NUM_HEADS": str(max(4, d // 64))})
            for d in [384, 512, 640, 768]
        ]
    elif sweep_type == "layers":
        return [
            (f"layers_{n}", {"NUM_LAYERS": str(n)})
            for n in [6, 9, 12, 15, 18]
        ]
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}. Use: lr, dim, layers")


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf ablation runner")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps (default: 2000)")
    parser.add_argument("--val-every", type=int, default=None, help="Validate every N steps (default: steps//5 or 100)")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--compare", action="store_true", help="Compare existing results")
    parser.add_argument("--sweep", type=str, default=None, help="Run a preset sweep (lr, dim, layers)")
    parser.add_argument("--script", type=str, default="train_gpt.py", help="Training script to use")
    parser.add_argument("--env", type=str, nargs="*", default=[], help="Extra env vars as KEY=VALUE")
    args = parser.parse_args()

    if args.compare:
        compare_results(RESULTS_DIR)
        return

    val_every = args.val_every or max(args.steps // 5, 50)
    extra_env = {}
    for kv in args.env:
        k, v = kv.split("=", 1)
        extra_env[k] = v

    if args.sweep:
        runs = build_sweep(args.sweep, args.steps, val_every)
        for name, overrides in runs:
            merged = {**extra_env, **overrides}
            run_config(name, merged, args.steps, val_every, args.script)
        print("\n\nSWEEP SUMMARY:")
        compare_results(RESULTS_DIR)
    else:
        name = args.name or f"baseline_s{args.steps}"
        run_config(name, extra_env, args.steps, val_every, args.script)


if __name__ == "__main__":
    main()
