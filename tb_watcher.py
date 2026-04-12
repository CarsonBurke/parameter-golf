"""
Watch a training log file and stream metrics to tensorboard in real-time.
Usage: python3 tb_watcher.py logs/baseline_2k.txt --name baseline_2k
"""
import argparse
import re
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

TB_DIR = Path(__file__).resolve().parent / "tb_logs"


def parse_line(line):
    m = re.match(
        r"step:(\d+)/\d+\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:([\d.]+)ms",
        line,
    )
    if m:
        return {"step": int(m.group(1)), "val_loss": float(m.group(2)),
                "val_bpb": float(m.group(3)), "train_time_ms": float(m.group(4)), "type": "val"}
    m = re.match(
        r"step:(\d+)/\d+\s+train_loss:([\d.]+)\s+train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms",
        line,
    )
    if m:
        return {"step": int(m.group(1)), "train_loss": float(m.group(2)),
                "train_time_ms": float(m.group(3)), "step_avg_ms": float(m.group(4)), "type": "train"}
    return None


def tail_and_write(log_path: str, name: str):
    writer = SummaryWriter(log_dir=str(TB_DIR / name))
    seen_lines = 0
    print(f"Watching {log_path} -> tb_logs/{name}")

    try:
        while True:
            path = Path(log_path)
            if not path.exists():
                time.sleep(1)
                continue
            lines = path.read_text().splitlines()
            for line in lines[seen_lines:]:
                e = parse_line(line)
                if not e:
                    continue
                time_s = int(e["train_time_ms"] / 1000)
                if e["type"] == "val":
                    writer.add_scalar("val/bpb", e["val_bpb"], e["step"])
                    writer.add_scalar("val/loss", e["val_loss"], e["step"])
                    writer.add_scalar("time/val_bpb", e["val_bpb"], time_s)
                    writer.add_scalar("time/val_loss", e["val_loss"], time_s)
                    print(f"  val step={e['step']} bpb={e['val_bpb']:.4f} t={time_s}s")
                elif e["type"] == "train":
                    writer.add_scalar("train/loss", e["train_loss"], e["step"])
                    writer.add_scalar("time/train_loss", e["train_loss"], time_s)
                    if "step_avg_ms" in e:
                        writer.add_scalar("perf/step_avg_ms", e["step_avg_ms"], e["step"])
                writer.flush()
            seen_lines = len(lines)
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Path to training log file")
    parser.add_argument("--name", required=True, help="Tensorboard run name")
    args = parser.parse_args()
    tail_and_write(args.log_path, args.name)
