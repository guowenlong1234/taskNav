#!/usr/bin/env python3
"""
Batch evaluate checkpoints and summarize best result.

Features:
1) Evaluate all checkpoint files under a directory in one command.
2) Support both fine-tune checkpoints (ckpt.iter*.pth) and pretrain checkpoints (model_step_*.pt).
3) Save per-checkpoint metrics, best checkpoint summary, and total elapsed time.
4) Plot metric curves and major parameter-change curves.
"""

'''
python batch_eval_checkpoints.py \
  --ckpt-dir data/logs/checkpoints/release_r2r \
  --dataset r2r \
  --split val_unseen

'''
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


DATASET_DEFAULTS = {
    "r2r": {
        "exp_config": "run_r2r/iter_train.yaml",
        "split": "val_unseen",
        "best_metric": "spl",
    },
    "rxr": {
        "exp_config": "run_rxr/iter_train.yaml",
        "split": "val_unseen",
        "best_metric": "ndtw",
    },
}

MINIMIZE_METRICS = {
    "distance_to_goal",
    "collisions",
    "path_length",
    "steps_taken",
    "episode_time",
    "ghost_cnt",
}

CHECKPOINT_EXTS = {".pth", ".pt"}


@dataclass
class CkptItem:
    path: Path
    step: int
    kind: str  # finetune | pretrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click batch validation for DGNav checkpoints."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Directory containing checkpoint files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["r2r", "rxr"],
        required=True,
        help="Dataset preset for validation config.",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default=None,
        help="Override eval config yaml path (default by --dataset).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Eval split (default by --dataset).",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default=None,
        help="Metric used to pick best checkpoint (default by --dataset).",
    )
    parser.add_argument(
        "--best-mode",
        type=str,
        choices=["auto", "max", "min"],
        default="auto",
        help="How to compare best metric.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id used for evaluation.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="NUM_ENVIRONMENTS override for eval.",
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        default=-1,
        help="EVAL.EPISODE_COUNT override.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pth,*.pt",
        help="Glob patterns, comma-separated.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search checkpoints.",
    )
    parser.add_argument(
        "--ckpt-kind",
        type=str,
        choices=["auto", "finetune", "pretrain"],
        default="auto",
        help="Force checkpoint kind or auto detect.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="Max checkpoints to evaluate (0 means all).",
    )
    parser.add_argument(
        "--step-min",
        type=int,
        default=None,
        help="Only evaluate checkpoints with step >= this value.",
    )
    parser.add_argument(
        "--step-max",
        type=int,
        default=None,
        help="Only evaluate checkpoints with step <= this value.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: batch_eval_results/<timestamp>).",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run run.py.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-checkpoint timeout in seconds (0 means no timeout).",
    )
    parser.add_argument(
        "--skip-param-plot",
        action="store_true",
        help="Skip parameter-change extraction and plotting.",
    )
    parser.add_argument(
        "--max-plotted-params",
        type=int,
        default=12,
        help="Maximum number of parameter curves in the plot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions, do not run evaluation.",
    )
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def format_metrics(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "no metrics"

    preferred = [
        "sr",
        "success",
        "spl",
        "ndtw",
        "sdtw",
        "oracle_success",
        "distance_to_goal",
        "path_length",
        "collisions",
        "steps_taken",
        "ghost_cnt",
        "episode_time",
    ]
    ordered = [k for k in preferred if k in metrics] + [
        k for k in sorted(metrics.keys()) if k not in preferred
    ]
    parts = []
    for k in ordered:
        v = metrics[k]
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def parse_step_from_name(name: str) -> int:
    patterns = [
        r"ckpt\.iter(\d+)",
        r"model_step_(\d+)",
        r"train_state_(\d+)",
        r"iter(\d+)",
        r"(\d+)",
    ]
    for p in patterns:
        m = re.search(p, name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return -1


def detect_ckpt_kind(path: Path, forced_kind: str) -> str:
    if forced_kind != "auto":
        return forced_kind

    name = path.name.lower()
    if "model_step" in name:
        return "pretrain"
    if "ckpt.iter" in name:
        return "finetune"
    if path.suffix.lower() == ".pth":
        return "finetune"
    if path.suffix.lower() == ".pt":
        return "pretrain"

    # Fallback (rare)
    try:
        import torch  # local import to keep --help usable without torch

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            return "finetune"
        if isinstance(obj, dict):
            return "pretrain"
    except Exception:
        pass
    return "finetune"


def iter_checkpoint_files(
    ckpt_dir: Path,
    patterns: Sequence[str],
    recursive: bool,
) -> Iterable[Path]:
    seen = set()
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        iterator = ckpt_dir.rglob(pat) if recursive else ckpt_dir.glob(pat)
        for p in iterator:
            if (
                p.is_file()
                and p.suffix.lower() in CHECKPOINT_EXTS
                and "train_state_" not in p.name
            ):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield p


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def should_minimize(metric_name: str, mode: str) -> bool:
    if mode == "min":
        return True
    if mode == "max":
        return False
    return metric_name in MINIMIZE_METRICS


def find_eval_metrics_file(exp_dir: Path, split: str) -> Optional[Path]:
    files = sorted(exp_dir.glob(f"stats_ckpt_*_{split}.json"))
    if not files:
        return None
    # Use the newest file in case there are more than one.
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def ensure_symlink_or_copy(src: Path, dst: Path) -> Path:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
        return dst
    except Exception:
        shutil.copy2(src, dst)
        return dst


def build_eval_ckpt_path(
    item: CkptItem,
    temp_ckpt_dir: Path,
    seq_idx: int,
) -> Tuple[Path, Optional[Path]]:
    """
    Returns:
      eval_ckpt_path: file path passed to EVAL.CKPT_PATH_DIR
      pretrained_override: MODEL.pretrained_path override for pretrain ckpt
    """
    step = item.step if item.step >= 0 else seq_idx
    eval_name = f"ckpt.iter{step}.pth"
    eval_path = temp_ckpt_dir / eval_name

    if item.kind == "finetune":
        if re.search(r"ckpt\.iter\d+\.pth$", item.path.name):
            return item.path, None
        return ensure_symlink_or_copy(item.path, eval_path), None

    # pretrain checkpoint: create tiny adapter checkpoint for eval loader.
    import torch  # local import to keep --help usable without torch

    # Keep one dummy key to avoid empty-key edge case in trainer checkpoint loader.
    adapter = {
        "state_dict": {"__dummy__": torch.tensor(0.0)},
        "iteration": step,
    }
    torch.save(adapter, eval_path)
    return eval_path, item.path


def run_one_eval(
    project_root: Path,
    python_bin: str,
    exp_name: str,
    exp_config: str,
    split: str,
    eval_ckpt_path: Path,
    pretrained_override: Optional[Path],
    gpu_id: int,
    num_envs: int,
    episode_count: int,
    out_root: Path,
    timeout: int,
    log_path: Path,
) -> Tuple[int, float]:
    results_root = out_root / "eval_results"
    tb_root = out_root / "tensorboard"
    ckpt_root = out_root / "tmp_checkpoints"
    video_root = out_root / "video"
    runlog_root = out_root / "running_log"
    for p in [results_root, tb_root, ckpt_root, video_root, runlog_root]:
        p.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        "run.py",
        "--exp_name",
        exp_name,
        "--run-type",
        "eval",
        "--exp-config",
        exp_config,
        "SIMULATOR_GPU_IDS",
        f"[{gpu_id}]",
        "TORCH_GPU_IDS",
        f"[{gpu_id}]",
        "GPU_NUMBERS",
        "1",
        "NUM_ENVIRONMENTS",
        str(num_envs),
        "EVAL.SPLIT",
        split,
        "EVAL.EPISODE_COUNT",
        str(episode_count),
        "EVAL.CKPT_PATH_DIR",
        str(eval_ckpt_path),
        "RESULTS_DIR",
        f"{results_root}/",
        "TENSORBOARD_DIR",
        f"{tb_root}/",
        "CHECKPOINT_FOLDER",
        f"{ckpt_root}/",
        "VIDEO_DIR",
        f"{video_root}/",
        "IL.is_requeue",
        "False",
    ]

    if pretrained_override is not None:
        cmd += ["MODEL.pretrained_path", str(pretrained_override)]

    env = os.environ.copy()
    env.setdefault("GLOG_minloglevel", "2")
    env.setdefault("MAGNUM_LOG", "quiet")

    begin = time.time()
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("# Command\n")
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=None if timeout <= 0 else timeout,
            check=False,
            text=True,
        )
    elapsed = time.time() - begin
    return proc.returncode, elapsed


def normalize_param_name(name: str) -> str:
    name = re.sub(r"^module\.", "", name)
    return name


def extract_major_param_values(
    ckpt_path: Path,
    ckpt_kind: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    try:
        import torch  # local import to keep --help usable without torch

        raw = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return out

    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        return out

    scalar_focus = [".w1", ".w2", ".w3", "node_gating_mlp"]
    norm_focus = [
        "semantic_sim_mlp.0.weight",
        "instruction_rel_mlp.0.weight",
        "global_sap_head.net.0.weight",
        "global_sap_head.net.4.weight",
    ]

    fallback_scalars: List[Tuple[str, float]] = []

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if not tensor.dtype.is_floating_point:
            continue

        pname = normalize_param_name(name)
        if tensor.numel() == 1:
            val = float(tensor.detach().cpu().item())
            fallback_scalars.append((pname, val))
            if any(k in pname for k in scalar_focus):
                out[pname] = val
        else:
            if any(k in pname for k in norm_focus):
                norm_v = float(tensor.detach().float().norm().cpu().item())
                out[f"norm::{pname}"] = norm_v

    if not out:
        # Fallback: pick first few scalar values, at least provide a curve.
        for name, val in fallback_scalars[:8]:
            out[name] = val

    return out


def plot_metric_curves(
    rows: List[Dict],
    out_file: Path,
) -> bool:
    if not HAS_MATPLOTLIB:
        return False
    if not rows:
        return False

    metric_keys = set()
    for r in rows:
        metric_keys.update((r.get("metrics") or {}).keys())
    if not metric_keys:
        return False

    # Prefer common eval metrics.
    preferred = [
        "success",
        "spl",
        "ndtw",
        "sdtw",
        "distance_to_goal",
        "oracle_success",
        "path_length",
        "collisions",
        "steps_taken",
    ]
    ordered = [k for k in preferred if k in metric_keys] + [
        k for k in sorted(metric_keys) if k not in preferred
    ]
    ordered = ordered[:8]  # keep plot readable

    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(len(ordered), 1, figsize=(10, max(3.2, 2.6 * len(ordered))), sharex=True)
    if len(ordered) == 1:
        axes = [axes]
    for ax, metric in zip(axes, ordered):
        ys = [(r.get("metrics") or {}).get(metric, float("nan")) for r in rows]
        ax.plot(steps, ys, marker="o", linewidth=1.4)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("checkpoint step")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    return True


def plot_param_curves(
    rows: List[Dict],
    out_file: Path,
    max_plotted_params: int,
) -> bool:
    if not HAS_MATPLOTLIB:
        return False
    if not rows:
        return False

    all_keys = set()
    for r in rows:
        all_keys.update((r.get("param_values") or {}).keys())
    if not all_keys:
        return False

    steps = [r["step"] for r in rows]
    series = {}
    for k in all_keys:
        vals = []
        for r in rows:
            v = (r.get("param_values") or {}).get(k)
            vals.append(float("nan") if v is None else float(v))
        series[k] = vals

    # Select parameters with largest range.
    scored = []
    for k, vals in series.items():
        finite = [x for x in vals if math.isfinite(x)]
        if len(finite) < 2:
            continue
        scored.append((max(finite) - min(finite), k))
    scored.sort(reverse=True, key=lambda x: x[0])
    chosen = [k for _, k in scored[:max_plotted_params]]
    if not chosen:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    for k in chosen:
        ax.plot(steps, series[k], marker="o", linewidth=1.2, label=k)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("parameter / norm value")
    ax.set_title("Major Parameter Changes")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)
    return True


def write_results_csv(rows: List[Dict], csv_path: Path) -> None:
    metric_keys = set()
    param_keys = set()
    for r in rows:
        metric_keys.update((r.get("metrics") or {}).keys())
        param_keys.update((r.get("param_values") or {}).keys())
    metric_keys = sorted(metric_keys)
    param_keys = sorted(param_keys)

    fields = [
        "checkpoint",
        "step",
        "kind",
        "status",
        "return_code",
        "elapsed_sec",
    ]
    fields += [f"metric::{k}" for k in metric_keys]
    fields += [f"param::{k}" for k in param_keys]
    fields += ["error"]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {
                "checkpoint": r.get("checkpoint"),
                "step": r.get("step"),
                "kind": r.get("kind"),
                "status": r.get("status"),
                "return_code": r.get("return_code"),
                "elapsed_sec": f"{r.get('elapsed_sec', float('nan')):.4f}",
                "error": r.get("error", ""),
            }
            for k in metric_keys:
                v = (r.get("metrics") or {}).get(k)
                row[f"metric::{k}"] = "" if v is None else v
            for k in param_keys:
                v = (r.get("param_values") or {}).get(k)
                row[f"param::{k}"] = "" if v is None else v
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    defaults = DATASET_DEFAULTS[args.dataset]
    exp_config = args.exp_config or defaults["exp_config"]
    split = args.split or defaults["split"]
    best_metric = args.best_metric or defaults["best_metric"]
    minimize = should_minimize(best_metric, args.best_mode)

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = (project_root / "batch_eval_results" / f"{args.dataset}_{now_ts()}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    temp_ckpt_dir = out_dir / "_eval_ckpt_adapters"
    temp_ckpt_dir.mkdir(exist_ok=True)

    patterns = [x.strip() for x in args.pattern.split(",") if x.strip()]
    items: List[CkptItem] = []
    for p in iter_checkpoint_files(ckpt_dir, patterns, args.recursive):
        step = parse_step_from_name(p.name)
        if args.step_min is not None and step >= 0 and step < args.step_min:
            continue
        if args.step_max is not None and step >= 0 and step > args.step_max:
            continue
        kind = detect_ckpt_kind(p, args.ckpt_kind)
        items.append(CkptItem(path=p.resolve(), step=step, kind=kind))

    if not items:
        raise RuntimeError("No checkpoint file matched the conditions.")

    items.sort(key=lambda x: (x.step if x.step >= 0 else 10**18, x.path.name))
    if args.max_checkpoints and args.max_checkpoints > 0:
        items = items[: args.max_checkpoints]

    run_cfg = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "ckpt_dir": str(ckpt_dir),
        "dataset": args.dataset,
        "exp_config": exp_config,
        "split": split,
        "best_metric": best_metric,
        "best_mode": "min" if minimize else "max",
        "gpu_id": args.gpu_id,
        "num_envs": args.num_envs,
        "episode_count": args.episode_count,
        "checkpoint_count": len(items),
        "patterns": patterns,
        "skip_param_plot": args.skip_param_plot,
        "python_bin": args.python_bin,
        "timeout_sec": args.timeout,
    }
    save_json(out_dir / "run_config.json", run_cfg)

    log(f"[BatchEval] Output directory: {out_dir}")
    log(f"[BatchEval] Found {len(items)} checkpoints")
    for idx, it in enumerate(items, 1):
        log(f"  {idx:03d}. step={it.step:<8} kind={it.kind:<8} {it.path}")

    if args.dry_run:
        log("[BatchEval] Dry-run mode. Exiting without running eval.")
        return

    all_rows: List[Dict] = []
    start_all = time.time()

    for i, item in enumerate(items, 1):
        step = item.step if item.step >= 0 else i
        exp_name = f"batch_eval_{args.dataset}_{step}_{i}"
        log_file = out_dir / "logs" / f"{i:03d}_step{step}_{item.path.stem}.log"

        eval_ckpt_path, pretrained_override = build_eval_ckpt_path(
            item=item,
            temp_ckpt_dir=temp_ckpt_dir,
            seq_idx=i,
        )

        log(
            f"[{i}/{len(items)}] Evaluating step={step} kind={item.kind} file={item.path.name}"
        )
        rc = -1
        elapsed = 0.0
        error = ""
        metrics: Dict[str, float] = {}

        try:
            rc, elapsed = run_one_eval(
                project_root=project_root,
                python_bin=args.python_bin,
                exp_name=exp_name,
                exp_config=exp_config,
                split=split,
                eval_ckpt_path=eval_ckpt_path,
                pretrained_override=pretrained_override,
                gpu_id=args.gpu_id,
                num_envs=args.num_envs,
                episode_count=args.episode_count,
                out_root=out_dir,
                timeout=args.timeout,
                log_path=log_file,
            )
            if rc == 0:
                exp_dir = out_dir / "eval_results" / exp_name
                metrics_file = find_eval_metrics_file(exp_dir, split)
                if metrics_file is None:
                    rc = 99
                    error = f"Metrics file not found in {exp_dir}"
                else:
                    metrics = load_json(metrics_file)
                    # Alias success to sr for easier reading.
                    if "success" in metrics and "sr" not in metrics:
                        metrics["sr"] = metrics["success"]
        except subprocess.TimeoutExpired:
            rc = 124
            elapsed = args.timeout
            error = f"Timeout after {args.timeout}s"
        except Exception as e:
            rc = 1
            error = str(e)

        param_values = {}
        if rc == 0 and not args.skip_param_plot:
            try:
                # Extract from original checkpoint path.
                param_values = extract_major_param_values(item.path, item.kind)
            except Exception:
                param_values = {}

        row = {
            "checkpoint": str(item.path),
            "step": step,
            "kind": item.kind,
            "status": "ok" if rc == 0 else "failed",
            "return_code": rc,
            "elapsed_sec": elapsed,
            "metrics": metrics,
            "param_values": param_values,
            "log_file": str(log_file),
            "error": error,
        }
        all_rows.append(row)
        save_json(out_dir / "results.json", {"results": all_rows})

        if rc == 0:
            score = metrics.get(best_metric, None)
            score_msg = "N/A" if score is None else f"{score:.6f}"
            log(f"    -> done in {hms(elapsed)}; {best_metric}={score_msg}")
            log(f"    -> metrics: {format_metrics(metrics)}")
        else:
            log(f"    -> failed (rc={rc}) in {hms(elapsed)}; {error}")

    total_elapsed = time.time() - start_all

    ok_rows = [r for r in all_rows if r["status"] == "ok" and best_metric in (r.get("metrics") or {})]
    best_row = None
    if ok_rows:
        key_fn = lambda r: (r["metrics"][best_metric], -r["step"]) if minimize else (r["metrics"][best_metric], r["step"])
        best_row = min(ok_rows, key=key_fn) if minimize else max(ok_rows, key=key_fn)

    # Save csv and plots.
    write_results_csv(all_rows, out_dir / "results.csv")
    metric_plot_ok = plot_metric_curves(
        rows=[r for r in all_rows if r["status"] == "ok"],
        out_file=out_dir / "metric_curves.png",
    )
    param_plot_ok = False
    if not args.skip_param_plot:
        param_plot_ok = plot_param_curves(
            rows=[r for r in all_rows if r["status"] == "ok"],
            out_file=out_dir / "major_param_changes.png",
            max_plotted_params=args.max_plotted_params,
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "split": split,
        "exp_config": exp_config,
        "best_metric": best_metric,
        "best_mode": "min" if minimize else "max",
        "total_elapsed_sec": total_elapsed,
        "total_elapsed_hms": hms(total_elapsed),
        "count_total": len(all_rows),
        "count_success": sum(1 for r in all_rows if r["status"] == "ok"),
        "count_failed": sum(1 for r in all_rows if r["status"] != "ok"),
        "best_checkpoint": None
        if best_row is None
        else {
            "path": best_row["checkpoint"],
            "step": best_row["step"],
            "kind": best_row["kind"],
            "elapsed_sec": best_row["elapsed_sec"],
            "metric_value": best_row["metrics"].get(best_metric),
            "metrics": best_row["metrics"],
            "log_file": best_row["log_file"],
        },
        "plots": {
            "metric_curves": str((out_dir / "metric_curves.png")) if metric_plot_ok else None,
            "major_param_changes": str((out_dir / "major_param_changes.png")) if param_plot_ok else None,
        },
        "matplotlib_available": HAS_MATPLOTLIB,
        "results_file": str(out_dir / "results.json"),
        "results_csv": str(out_dir / "results.csv"),
    }
    save_json(out_dir / "summary.json", summary)

    # Markdown report
    report_lines = []
    report_lines.append("# Batch Eval Summary")
    report_lines.append("")
    report_lines.append(f"- Dataset: `{args.dataset}`")
    report_lines.append(f"- Split: `{split}`")
    report_lines.append(f"- Config: `{exp_config}`")
    report_lines.append(f"- Best metric: `{best_metric}` (`{'min' if minimize else 'max'}`)")
    report_lines.append(f"- Success/Total: `{summary['count_success']}/{summary['count_total']}`")
    report_lines.append(f"- Total elapsed: `{summary['total_elapsed_hms']}` ({summary['total_elapsed_sec']:.2f}s)")
    report_lines.append("")
    if best_row is not None:
        report_lines.append("## Best Checkpoint")
        report_lines.append("")
        report_lines.append(f"- Path: `{best_row['checkpoint']}`")
        report_lines.append(f"- Step: `{best_row['step']}`")
        report_lines.append(f"- Kind: `{best_row['kind']}`")
        report_lines.append(f"- {best_metric}: `{best_row['metrics'].get(best_metric)}`")
        report_lines.append("- Full metrics:")
        for k in sorted(best_row["metrics"].keys()):
            report_lines.append(f"  - `{k}`: `{best_row['metrics'][k]}`")
    else:
        report_lines.append("## Best Checkpoint")
        report_lines.append("")
        report_lines.append("- No successful evaluation found.")
    report_lines.append("")
    report_lines.append("## Files")
    report_lines.append("")
    report_lines.append(f"- Summary: `{out_dir / 'summary.json'}`")
    report_lines.append(f"- Results: `{out_dir / 'results.json'}`")
    report_lines.append(f"- CSV: `{out_dir / 'results.csv'}`")
    if metric_plot_ok:
        report_lines.append(f"- Metric plot: `{out_dir / 'metric_curves.png'}`")
    if param_plot_ok:
        report_lines.append(f"- Param plot: `{out_dir / 'major_param_changes.png'}`")
    if not HAS_MATPLOTLIB:
        report_lines.append("- Plot skipped: matplotlib not available.")
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    log("")
    log("[BatchEval] Finished.")
    log(f"[BatchEval] Total elapsed: {hms(total_elapsed)}")
    log(f"[BatchEval] Summary: {out_dir / 'summary.json'}")
    if best_row is not None:
        log(
            f"[BatchEval] Best: step={best_row['step']} {best_metric}={best_row['metrics'].get(best_metric)}"
        )
        log(f"[BatchEval] Best ckpt: {best_row['checkpoint']}")
        log(f"[BatchEval] Best metrics: {format_metrics(best_row['metrics'])}")
    else:
        log("[BatchEval] No valid best checkpoint (all eval failed or metric missing).")


if __name__ == "__main__":
    main()
