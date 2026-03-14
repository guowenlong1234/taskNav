#!/usr/bin/env python3
"""
Auto benchmark NUM_ENVIRONMENTS by running short training jobs and parsing
train rollout timing logs.

Example:
taskset -c 14-27 python batch_eval_checkpoints.py \
  --ckpt-dir /home/gwl/project/DGNav_new/habitat-lab/DGNav/data/logs/checkpoints/release_r2r_dino_best_gacc/ \
  --dataset r2r \
  --split val_unseen \
  --parallel-jobs 1 \
  --num-envs 6 \
  --run-name dino_eval_curve

"""
import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_env_candidates(text: str) -> List[int]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid NUM_ENVIRONMENTS candidate.")
    if vals[0] <= 0:
        raise ValueError("NUM_ENVIRONMENTS candidates must be > 0.")
    return vals


def detect_dist_launch_module(python_bin: str) -> str:
    cmd = [
        python_bin,
        "-c",
        "import importlib.util as u; raise SystemExit(0 if u.find_spec('torch.distributed.run') else 1)",
    ]
    rc = subprocess.run(cmd, check=False).returncode
    return "torch.distributed.run" if rc == 0 else "torch.distributed.launch"


def load_timing_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            item: Dict[str, float] = {}
            for k, v in r.items():
                if k == "timestamp":
                    continue
                try:
                    item[k] = float(v)
                except Exception:
                    pass
            if item:
                rows.append(item)
    return rows


def summarize_rows(rows: List[Dict[str, float]], warmup_rows: int) -> Dict[str, float]:
    if not rows:
        return {}
    used = rows[warmup_rows:] if len(rows) > warmup_rows else rows
    if not used:
        used = rows

    def mean_key(k: str) -> float:
        vals = [x[k] for x in used if k in x]
        return float(st.mean(vals)) if vals else float("nan")

    thru = [
        x["total_actions"] / x["rollout_total_s"]
        for x in used
        if x.get("rollout_total_s", 0.0) > 1e-9 and "total_actions" in x
    ]
    return {
        "rows_total": float(len(rows)),
        "rows_used": float(len(used)),
        "throughput_actions_per_s_mean": float(st.mean(thru)) if thru else float("nan"),
        "throughput_actions_per_s_median": float(st.median(thru)) if thru else float("nan"),
        "rollout_total_s_mean": mean_key("rollout_total_s"),
        "waypoint_pct_mean": mean_key("waypoint_pct"),
        "env_call_at_pct_mean": mean_key("env_call_at_pct"),
        "navigation_pct_mean": mean_key("navigation_pct"),
        "env_step_pct_mean": mean_key("env_step_pct"),
        "env_instances_avg_mean": mean_key("env_instances_avg"),
        "call_at_requests_mean": mean_key("call_at_requests"),
    }


def find_timing_log(perf_dir: Path, exp_name: str) -> Optional[Path]:
    if not perf_dir.is_dir():
        return None
    cands = sorted(
        perf_dir.glob(f"{exp_name}_train_rollout_timing_rank0_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    if cands:
        return cands[-1]
    fallback = sorted(perf_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    return fallback[-1] if fallback else None


def run_one_candidate(
    project_root: Path,
    python_bin: str,
    dist_launch_module: str,
    exp_config: str,
    exp_name: str,
    master_port: int,
    gpu_id: int,
    num_env: int,
    iters: int,
    log_every: int,
    out_root: Path,
    pretrained_path: str,
    timeout_sec: int,
    allow_sliding: bool,
) -> Dict:
    ckpt_root = (out_root / "checkpoints").resolve()
    tb_root = (out_root / "tensorboard").resolve()
    eval_root = (out_root / "eval_results").resolve()
    video_root = (out_root / "video").resolve()
    run_logs = (out_root / "run_logs").resolve()
    for p in [ckpt_root, tb_root, eval_root, video_root, run_logs]:
        p.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        "-m",
        dist_launch_module,
        "--nproc_per_node=1",
        "--master_port",
        str(master_port),
        "run.py",
        "--exp_name",
        exp_name,
        "--run-type",
        "train",
        "--exp-config",
        exp_config,
        "SIMULATOR_GPU_IDS",
        f"[{gpu_id}]",
        "TORCH_GPU_IDS",
        f"[{gpu_id}]",
        "GPU_NUMBERS",
        "1",
        "NUM_ENVIRONMENTS",
        str(num_env),
        "IL.iters",
        str(iters),
        "IL.log_every",
        str(log_every),
        "IL.load_from_ckpt",
        "False",
        "IL.is_requeue",
        "False",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING",
        "True" if allow_sliding else "False",
        "CHECKPOINT_FOLDER",
        str(ckpt_root) + "/",
        "TENSORBOARD_DIR",
        str(tb_root) + "/",
        "RESULTS_DIR",
        str(eval_root) + "/",
        "VIDEO_DIR",
        str(video_root) + "/",
    ]
    if pretrained_path:
        cmd += ["MODEL.pretrained_path", pretrained_path]

    env = os.environ.copy()
    env.update(
        {
            "GLOG_minloglevel": "2",
            "MAGNUM_LOG": "quiet",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )

    train_log = run_logs / f"{exp_name}.log"
    begin = time.time()
    with train_log.open("w", encoding="utf-8") as lf:
        lf.write("# Command\n")
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=None if timeout_sec <= 0 else timeout_sec,
            text=True,
        )
    elapsed = time.time() - begin

    perf_dir = ckpt_root / exp_name / "perf_timing"
    timing_log = find_timing_log(perf_dir, exp_name)
    return {
        "return_code": proc.returncode,
        "elapsed_sec": elapsed,
        "train_log": str(train_log),
        "perf_dir": str(perf_dir),
        "timing_log": str(timing_log) if timing_log else None,
    }


def format_float(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto benchmark NUM_ENVIRONMENTS using train rollout timing logs."
    )
    parser.add_argument("--exp-config", type=str, default="run_r2r/iter_train.yaml")
    parser.add_argument("--env-candidates", type=str, default="4,5,6,8")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--warmup-rows", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--master-port-start", type=int, default=29620)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--pretrained-path", type=str, default="")
    parser.add_argument("--base-exp-name", type=str, default="numenv_bench")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument(
        "--allow-sliding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="num_env_benchmark_results",
        help="Root directory to store benchmark artifacts.",
    )
    args = parser.parse_args()

    if args.iters <= 0 or args.log_every <= 0:
        raise ValueError("--iters and --log-every must be > 0")

    env_candidates = parse_env_candidates(args.env_candidates)
    project_root = Path(__file__).resolve().parent
    out_root = (project_root / args.output_dir / ts_now()).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dist_launch_module = detect_dist_launch_module(args.python_bin)
    log(f"Project root: {project_root}")
    log(f"Output dir: {out_root}")
    log(f"Using launch module: {dist_launch_module}")
    log(f"Candidates: {env_candidates}")

    results: List[Dict] = []
    for idx, num_env in enumerate(env_candidates, 1):
        exp_name = f"{args.base_exp_name}_env{num_env}_{ts_now()}"
        port = args.master_port_start + idx
        log(f"[{idx}/{len(env_candidates)}] Benchmark NUM_ENVIRONMENTS={num_env}")
        run_info = run_one_candidate(
            project_root=project_root,
            python_bin=args.python_bin,
            dist_launch_module=dist_launch_module,
            exp_config=args.exp_config,
            exp_name=exp_name,
            master_port=port,
            gpu_id=args.gpu_id,
            num_env=num_env,
            iters=args.iters,
            log_every=args.log_every,
            out_root=out_root,
            pretrained_path=args.pretrained_path,
            timeout_sec=args.timeout_sec,
            allow_sliding=args.allow_sliding,
        )

        rec: Dict = {
            "num_env": num_env,
            "exp_name": exp_name,
            **run_info,
            "summary": {},
        }

        if run_info["return_code"] == 0 and run_info["timing_log"]:
            timing_path = Path(run_info["timing_log"])
            rows = load_timing_rows(timing_path)
            rec["summary"] = summarize_rows(rows, args.warmup_rows)
            log(
                "    -> ok: "
                f"throughput_mean={format_float(rec['summary'].get('throughput_actions_per_s_mean', float('nan')))} "
                f"env_step_pct={format_float(rec['summary'].get('env_step_pct_mean', float('nan')))}"
            )
        else:
            log(
                f"    -> failed (rc={run_info['return_code']}); check log: {run_info['train_log']}"
            )
        results.append(rec)

    ok = [x for x in results if x.get("return_code") == 0 and x.get("summary")]
    best = None
    if ok:
        best = max(
            ok,
            key=lambda x: x["summary"].get("throughput_actions_per_s_mean", float("-inf")),
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "exp_config": args.exp_config,
        "env_candidates": env_candidates,
        "iters": args.iters,
        "log_every": args.log_every,
        "warmup_rows": args.warmup_rows,
        "gpu_id": args.gpu_id,
        "python_bin": args.python_bin,
        "dist_launch_module": dist_launch_module,
        "results": results,
        "best_num_env": None if best is None else best["num_env"],
        "best_result": best,
    }

    report_json = out_root / "benchmark_report.json"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# NUM_ENV Benchmark Report")
    lines.append("")
    lines.append(f"- Timestamp: `{report['timestamp']}`")
    lines.append(f"- Config: `{args.exp_config}`")
    lines.append(f"- Candidates: `{env_candidates}`")
    lines.append(f"- iters/log_every: `{args.iters}/{args.log_every}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for r in results:
        lines.append(f"- NUM_ENV={r['num_env']}, rc={r['return_code']}")
        if r.get("summary"):
            s = r["summary"]
            lines.append(
                "  throughput_mean="
                + format_float(s.get("throughput_actions_per_s_mean", float("nan")))
                + ", env_step_pct="
                + format_float(s.get("env_step_pct_mean", float("nan")))
                + ", rollout_mean_s="
                + format_float(s.get("rollout_total_s_mean", float("nan")))
            )
        lines.append(f"  timing_log={r.get('timing_log')}")
        lines.append(f"  train_log={r.get('train_log')}")
    lines.append("")
    if best is None:
        lines.append("## Recommendation")
        lines.append("")
        lines.append("- No successful run.")
    else:
        lines.append("## Recommendation")
        lines.append("")
        lines.append(
            f"- Best NUM_ENVIRONMENTS: `{best['num_env']}` "
            f"(throughput_mean={format_float(best['summary'].get('throughput_actions_per_s_mean', float('nan')))})"
        )
    report_md = out_root / "benchmark_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")

    log(f"Saved report: {report_json}")
    if best is None:
        log("No successful benchmark run.")
    else:
        log(
            "Recommended NUM_ENVIRONMENTS="
            f"{best['num_env']} "
            f"(throughput_mean={format_float(best['summary'].get('throughput_actions_per_s_mean', float('nan')))})"
        )


if __name__ == "__main__":
    main()
