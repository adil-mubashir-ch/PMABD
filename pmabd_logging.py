"""
pmabd_logging.py
================
Run logging for the PMABD pipeline. Three sinks, all written under
<output_dir>/logs/:

  run_<timestamp>.log       every line of stdout/stderr, verbatim
  epochs.csv                one row per training epoch, every stage appended
  epochs.jsonl              the same rows as JSON, for programmatic reload
  stages.json               one entry per completed stage (final results table)
  run_meta.json             config snapshot, git commit, environment, splits

Nothing here changes training behaviour — it only records it.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EPOCH_FIELDS = [
    "wall_clock", "stage", "cycle", "cycle_epoch", "epoch", "max_epochs",
    "bitwidth_w", "bitwidth_a", "lr",
    "loss", "loss_ce", "loss_kd", "loss_feat", "loss_qat",
    "grad_norm", "kd_w", "ce_w", "T_eff",
    "train_top1", "val_top1", "val_top5", "test_top1", "test_top5",
    "best_val_so_far", "epoch_secs",
]


class Tee:
    """
    Duplicate a stream to a file. The pipeline logs training through plain
    `print`, so capturing stdout is the only way to get the per-batch and
    quantization-health output into a file without rewriting it.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            # Legacy Windows consoles are cp1252; the pipeline prints box-drawing
            # and arrow characters. Losing a glyph beats losing the run.
            enc = getattr(self.stream, "encoding", None) or "ascii"
            self.stream.write(data.encode(enc, errors="replace").decode(enc))
        try:
            self.file.write(data)
        except ValueError:      # file closed during interpreter shutdown
            pass
        return len(data)

    def flush(self):
        self.stream.flush()
        if not self.file.closed:
            self.file.flush()

    def isatty(self):
        return self.stream.isatty()

    def close(self):
        if not self.file.closed:
            self.file.close()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


class RunLogger:
    """
    Owns the log directory for one invocation of a runner script.

    Usage:
        rl = RunLogger(output_dir, run_name="tinyimagenet_nosat")
        rl.start_capture()                      # tee stdout/stderr to file
        rl.write_meta({...})
        ...
        rl.log_epoch({...})                     # pass to epoch_callback=
        rl.log_stage("stage_m4", {...})
        rl.close()
    """

    def __init__(self, output_dir: str, run_name: str = "run"):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.t0 = time.time()

        self.text_path   = self.log_dir / f"run_{self.run_name}_{self.stamp}.log"
        self.csv_path    = self.log_dir / "epochs.csv"
        self.jsonl_path  = self.log_dir / "epochs.jsonl"
        self.stages_path = self.log_dir / "stages.json"
        self.meta_path   = self.log_dir / f"run_meta_{self.stamp}.json"

        # CSV header is written once; later runs append to the same file so a
        # resumed ladder stays a single continuous history.
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=EPOCH_FIELDS).writeheader()

        self._stdout = None
        self._stderr = None

    # ── stdout/stderr capture ────────────────────────────────────────────
    def start_capture(self):
        # The pipeline prints non-ASCII glyphs; a cp1252 console would raise on
        # them and kill the run mid-stage. UTF-8 with replacement is harmless
        # everywhere and saves Windows terminals.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(self._stdout, self.text_path)
        sys.stderr = Tee(self._stderr, self.text_path)

    def stop_capture(self):
        for cur, orig in ((sys.stdout, self._stdout), (sys.stderr, self._stderr)):
            if isinstance(cur, Tee):
                cur.close()
        if self._stdout is not None:
            sys.stdout, sys.stderr = self._stdout, self._stderr
            self._stdout = self._stderr = None

    # ── structured records ───────────────────────────────────────────────
    def write_meta(self, meta: dict):
        meta = dict(meta)
        meta.update({
            "run_name":    self.run_name,
            "timestamp":   self.stamp,
            "argv":        sys.argv,
            "cwd":         os.getcwd(),
            "git_commit":  _git_commit(),
            "python":      sys.version,
            "platform":    platform.platform(),
        })
        try:
            import torch
            meta["torch"] = torch.__version__
            meta["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                meta["gpu"] = torch.cuda.get_device_name(0)
                meta["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[LOG] Run metadata -> {self.meta_path}")

    def log_epoch(self, rec: dict):
        row = {k: rec.get(k) for k in EPOCH_FIELDS}
        row["wall_clock"] = round(time.time() - self.t0, 1)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EPOCH_FIELDS).writerow(row)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def count_epochs(self, stage_name: str) -> int:
        """How many epoch rows this run recorded for a given stage name."""
        if not self.jsonl_path.exists():
            return 0
        n = 0
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("stage", "").startswith(stage_name):
                        n += 1
                except json.JSONDecodeError:
                    continue
        return n

    def log_stage(self, stage_id: str, rec: dict):
        entries = []
        if self.stages_path.exists():
            try:
                entries = json.loads(self.stages_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                entries = []
        entries = [e for e in entries if e.get("stage_id") != stage_id]
        entry = {"stage_id": stage_id, "logged_at": datetime.now().isoformat()}
        entry.update(rec)
        entries.append(entry)
        with open(self.stages_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
        print(f"[LOG] Stage {stage_id} recorded -> {self.stages_path}")

    def stage_table(self) -> str:
        """Render stages.json as a fixed-width results table."""
        if not self.stages_path.exists():
            return "(no stages recorded)"
        entries = json.loads(self.stages_path.read_text(encoding="utf-8"))
        order = ["m3_kd", "m4", "m5a", "m5b", "m6a", "m6b", "m7a", "m7b"]
        entries.sort(key=lambda e: order.index(e["stage_id"])
                     if e.get("stage_id") in order else 99)
        w = max([len(str(e.get("stage_id", ""))) for e in entries] + [5])
        head = (f"{'stage':{w}s} {'precision':10s} {'val top1':>9s} "
                f"{'test top1':>10s} {'test top5':>10s} {'epochs':>7s} {'hours':>7s}")
        lines = [head, "-" * len(head)]
        nan = float("nan")
        for e in entries:
            lines.append(
                f"{str(e.get('stage_id','')):{w}s} {str(e.get('precision','')):10s} "
                f"{e.get('best_val_top1') if e.get('best_val_top1') is not None else nan:9.2f} "
                f"{e.get('test_top1') if e.get('test_top1') is not None else nan:10.2f} "
                f"{e.get('test_top5') if e.get('test_top5') is not None else nan:10.2f} "
                f"{int(e.get('epochs_run') or 0):7d} {e.get('hours') or 0.0:7.2f}")
        return "\n".join(lines)

    def close(self):
        print(f"[LOG] Text log     : {self.text_path}")
        print(f"[LOG] Epoch CSV    : {self.csv_path}")
        print(f"[LOG] Epoch JSONL  : {self.jsonl_path}")
        print(f"[LOG] Stage results: {self.stages_path}")
        self.stop_capture()
