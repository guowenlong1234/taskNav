#!/usr/bin/env bash
set -euo pipefail

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONUNBUFFERED=1

# Avoid CPU thread oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
echo "[run_oracle_eval.bash] CPU thread caps: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}"

# Resolve workspace paths from this script location.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dgnav_dir="$(cd "${script_dir}/.." && pwd)"
habitat_repo_root="$(cd "${dgnav_dir}/.." && pwd -P)"
export DGNAV_HABITAT_REPO_ROOT="${habitat_repo_root}"
export PYTHONPATH="${habitat_repo_root}/habitat-lab:${habitat_repo_root}/habitat-baselines:${PYTHONPATH:-}"
echo "[run_oracle_eval.bash] Using Habitat-Lab/Baselines from ${habitat_repo_root}"

dist_launch_module="torch.distributed.launch"
if python -c "import importlib.util as u; raise SystemExit(0 if u.find_spec('torch.distributed.run') else 1)"; then
      dist_launch_module="torch.distributed.run"
fi

# Common knobs. Edit here or override with environment variables.
conda_env="${CONDA_ENV:-py3-9}"
experiment_variant="${EXPERIMENT_VARIANT:-oracle}"

master_port="${MASTER_PORT:-4713}"

a3_scope="${A3_SCOPE:-${ORACLE_TARGET_GHOST_SCOPE:-all}}"
case "${a3_scope}" in
      all|new_only|local_frontier|top1_shadow)
            ;;
      *)
            echo "[run_oracle_eval.bash] Unsupported A3_SCOPE=${a3_scope}" >&2
            echo "[run_oracle_eval.bash] Supported values: all | new_only | local_frontier | top1_shadow" >&2
            exit 1
            ;;
esac

default_a3_config_path="run_r2r/eval_oracle_a3_base.yaml,run_r2r/eval_oracle_a3_${a3_scope}.yaml"
config_path="${CONFIG_PATH:-${default_a3_config_path}}"
ckpt_path="${CKPT_PATH:-/home/gwl/project/DGNav_new/habitat-lab/DGNav/data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth}"
episode_id_file="${EPISODE_ID_FILE:-run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt}"
num_environments="${NUM_ENVIRONMENTS:-1}"
gpu_numbers="${GPU_NUMBERS:-1}"
simulator_gpu_ids="${SIMULATOR_GPU_IDS:-[0]}"
torch_gpu_ids="${TORCH_GPU_IDS:-[0]}"
torch_gpu_id="${TORCH_GPU_ID:-0}"

episode_count="${EPISODE_COUNT:--1}"

fast_eval="${FAST_EVAL:-False}"
allow_sliding="${ALLOW_SLIDING:-True}"
results_dir="${RESULTS_DIR:-data/logs/eval_results/}"
tensorboard_dir="${TENSORBOARD_DIR:-data/logs/tensorboard_dirs/}"
checkpoint_folder="${CHECKPOINT_FOLDER:-data/logs/checkpoints/}"
video_dir="${VIDEO_DIR:-data/logs/video/}"
oracle_trace_enable="${ORACLE_TRACE_ENABLE:-False}"
oracle_apply_mode="${ORACLE_APPLY_MODE:-soft}"
oracle_soft_alpha="${ORACLE_SOFT_ALPHA:-0.50}"
oracle_target_ghost_scope="${ORACLE_TARGET_GHOST_SCOPE:-${a3_scope}}"
oracle_refresh_policy="${ORACLE_REFRESH_POLICY:-on_change}"
oracle_query_heading_strategy="${ORACLE_QUERY_HEADING_STRATEGY:-face_frontier}"
oracle_query_pipeline="${ORACLE_QUERY_PIPELINE:-future_node_avg_pano}"
oracle_strict_scope="${ORACLE_STRICT_SCOPE:-True}"
oracle_shadow_rerun_planner="${ORACLE_SHADOW_RERUN_PLANNER:-True}"
oracle_max_scope_ghosts="${ORACLE_MAX_SCOPE_GHOSTS:--1}"
oracle_scope_trace_enable="${ORACLE_SCOPE_TRACE_ENABLE:-False}"
oracle_scope_summary_enable="${ORACLE_SCOPE_SUMMARY_ENABLE:-False}"

cpu_set="${CPU_SET:-}"

case "${experiment_variant}" in
      oracle_cache)
            default_exp_name="fixed500_oracle_cache_${oracle_target_ghost_scope}"
            oracle_enable="True"
            oracle_cache_enable="True"
            ;;
      oracle)
            default_exp_name="fixed500_oracle_${oracle_target_ghost_scope}"
            oracle_enable="True"
            oracle_cache_enable="False"
            ;;
      baseline)
            default_exp_name="fixed500_baseline"
            oracle_enable="False"
            oracle_cache_enable="False"
            ;;
      *)
            echo "[run_oracle_eval.bash] Unsupported EXPERIMENT_VARIANT=${experiment_variant}" >&2
            echo "[run_oracle_eval.bash] Supported values: oracle_cache | oracle | baseline" >&2
            exit 1
            ;;
esac

exp_name="${EXP_NAME:-${default_exp_name}}"

echo "[run_oracle_eval.bash] EXP_NAME=${exp_name}"
echo "[run_oracle_eval.bash] EXPERIMENT_VARIANT=${experiment_variant}"
echo "[run_oracle_eval.bash] A3_SCOPE=${a3_scope}"
echo "[run_oracle_eval.bash] CONFIG_PATH=${config_path}"
echo "[run_oracle_eval.bash] CKPT_PATH=${ckpt_path}"
echo "[run_oracle_eval.bash] EPISODE_ID_FILE=${episode_id_file}"
echo "[run_oracle_eval.bash] NUM_ENVIRONMENTS=${num_environments}"
echo "[run_oracle_eval.bash] EPISODE_COUNT=${episode_count}"
echo "[run_oracle_eval.bash] MASTER_PORT=${master_port}"
echo "[run_oracle_eval.bash] ORACLE.enable=${oracle_enable}"
echo "[run_oracle_eval.bash] ORACLE.cache_enable=${oracle_cache_enable}"
echo "[run_oracle_eval.bash] ORACLE.apply_mode=${oracle_apply_mode}"
echo "[run_oracle_eval.bash] ORACLE.soft_alpha=${oracle_soft_alpha}"
echo "[run_oracle_eval.bash] ORACLE.target_ghost_scope=${oracle_target_ghost_scope}"
echo "[run_oracle_eval.bash] ORACLE.refresh_policy=${oracle_refresh_policy}"
echo "[run_oracle_eval.bash] ORACLE.query_heading_strategy=${oracle_query_heading_strategy}"
echo "[run_oracle_eval.bash] ORACLE.query_pipeline=${oracle_query_pipeline}"
echo "[run_oracle_eval.bash] ORACLE.strict_scope=${oracle_strict_scope}"
echo "[run_oracle_eval.bash] ORACLE.shadow_rerun_planner=${oracle_shadow_rerun_planner}"
echo "[run_oracle_eval.bash] ORACLE.max_scope_ghosts=${oracle_max_scope_ghosts}"
echo "[run_oracle_eval.bash] ORACLE.scope_trace_enable=${oracle_scope_trace_enable}"
echo "[run_oracle_eval.bash] ORACLE.scope_summary_enable=${oracle_scope_summary_enable}"
echo "[run_oracle_eval.bash] ORACLE.trace.enable=${oracle_trace_enable}"
echo "[run_oracle_eval.bash] CPU_SET=${cpu_set:-<disabled>}"

flag_eval="--exp_name ${exp_name}
      --run-type eval
      --exp-config ${config_path}
      SIMULATOR_GPU_IDS ${simulator_gpu_ids}
      TORCH_GPU_IDS ${torch_gpu_ids}
      TORCH_GPU_ID ${torch_gpu_id}
      GPU_NUMBERS ${gpu_numbers}
      NUM_ENVIRONMENTS ${num_environments}
      EVAL.CKPT_PATH_DIR ${ckpt_path}
      EVAL.EPISODE_COUNT ${episode_count}
      EVAL.EPISODE_ID_FILE ${episode_id_file}
      EVAL.fast_eval ${fast_eval}
      RESULTS_DIR ${results_dir}
      TENSORBOARD_DIR ${tensorboard_dir}
      CHECKPOINT_FOLDER ${checkpoint_folder}
      VIDEO_DIR ${video_dir}
      ORACLE.enable ${oracle_enable}
      ORACLE.cache_enable ${oracle_cache_enable}
      ORACLE.apply_mode ${oracle_apply_mode}
      ORACLE.soft_alpha ${oracle_soft_alpha}
      ORACLE.target_ghost_scope ${oracle_target_ghost_scope}
      ORACLE.refresh_policy ${oracle_refresh_policy}
      ORACLE.query_heading_strategy ${oracle_query_heading_strategy}
      ORACLE.query_pipeline ${oracle_query_pipeline}
      ORACLE.strict_scope ${oracle_strict_scope}
      ORACLE.shadow_rerun_planner ${oracle_shadow_rerun_planner}
      ORACLE.max_scope_ghosts ${oracle_max_scope_ghosts}
      ORACLE.scope_trace_enable ${oracle_scope_trace_enable}
      ORACLE.scope_summary_enable ${oracle_scope_summary_enable}
      ORACLE.trace.enable ${oracle_trace_enable}
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING ${allow_sliding}
      "

run_oracle_eval() {
      local port="$1"
      shift
      local -a run_prefix=()
      if [[ -n "${cpu_set}" ]]; then
            run_prefix=(taskset -c "${cpu_set}")
      fi

      if [[ "${CONDA_DEFAULT_ENV:-}" == "${conda_env}" ]]; then
            echo "[run_oracle_eval.bash] Using current conda env: ${CONDA_DEFAULT_ENV}"
            "${run_prefix[@]}" python -m "${dist_launch_module}" \
                  --nproc_per_node=1 \
                  --master_port "${port}" \
                  run.py "$@"
      else
            echo "[run_oracle_eval.bash] Using conda run -n ${conda_env} --no-capture-output"
            "${run_prefix[@]}" conda run --no-capture-output -n "${conda_env}" \
                  python -m "${dist_launch_module}" \
                  --nproc_per_node=1 \
                  --master_port "${port}" \
                  run.py "$@"
      fi
}

cd "${dgnav_dir}"

echo "###### oracle eval mode ######"
run_oracle_eval "${master_port}" ${flag_eval}
