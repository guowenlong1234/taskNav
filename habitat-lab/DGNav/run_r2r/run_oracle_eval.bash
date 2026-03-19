#!/usr/bin/env bash
set -euo pipefail

# 底层评测入口。
# 日常建议直接使用 run_r2r/run_oracle_experiment.bash 里的命名预设。

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
echo "[run_oracle_eval.bash] CPU thread caps: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}"

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

conda_env="${CONDA_ENV:-py3-9}"
experiment_variant="${EXPERIMENT_VARIANT:-custom}"
master_port="${MASTER_PORT:-4713}"

a3_scope="${A3_SCOPE:-all}"
case "${a3_scope}" in
      all|new_only|local_frontier|top1_shadow)
            ;;
      *)
            echo "[run_oracle_eval.bash] Unsupported A3_SCOPE=${a3_scope}" >&2
            echo "[run_oracle_eval.bash] Supported values: all | new_only | local_frontier | top1_shadow" >&2
            exit 1
            ;;
esac

case "${experiment_variant}" in
      oracle_cache)
            default_exp_name="oracle_cache_eval_${a3_scope}"
            ;;
      oracle)
            default_exp_name="oracle_eval_${a3_scope}"
            ;;
      baseline)
            default_exp_name="baseline_eval"
            ;;
      custom)
            default_exp_name="oracle_eval"
            ;;
      *)
            echo "[run_oracle_eval.bash] Unsupported EXPERIMENT_VARIANT=${experiment_variant}" >&2
            echo "[run_oracle_eval.bash] Supported values: custom | oracle_cache | oracle | baseline" >&2
            exit 1
            ;;
esac

exp_name="${EXP_NAME:-${default_exp_name}}"
config_path="${CONFIG_PATH:-run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_eval_on.yaml}"
ckpt_path="${CKPT_PATH:-${dgnav_dir}/data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth}"
episode_id_file="${EPISODE_ID_FILE-}"
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
log_dir="${LOG_DIR:-data/logs/running_log/}"

# 可选覆盖项。
# 变量未设置时，保持 YAML 配置栈中的原值，不额外覆盖。
oracle_enable="${ORACLE_ENABLE-}"
eval_env_refill_policy="${EVAL_ENV_REFILL_POLICY-}"
oracle_cache_enable="${ORACLE_CACHE_ENABLE-}"
oracle_trace_enable="${ORACLE_TRACE_ENABLE-}"
oracle_apply_mode="${ORACLE_APPLY_MODE-}"
oracle_soft_alpha="${ORACLE_SOFT_ALPHA-}"
oracle_target_ghost_scope="${ORACLE_TARGET_GHOST_SCOPE-}"
oracle_enable_in_train="${ORACLE_ENABLE_IN_TRAIN-}"
oracle_enable_in_eval="${ORACLE_ENABLE_IN_EVAL-}"
oracle_refresh_policy="${ORACLE_REFRESH_POLICY-}"
oracle_query_heading_strategy="${ORACLE_QUERY_HEADING_STRATEGY-}"
oracle_query_pipeline="${ORACLE_QUERY_PIPELINE-}"
oracle_strict_scope="${ORACLE_STRICT_SCOPE-}"
oracle_shadow_rerun_planner="${ORACLE_SHADOW_RERUN_PLANNER-}"
oracle_max_scope_ghosts="${ORACLE_MAX_SCOPE_GHOSTS-}"
oracle_scope_trace_enable="${ORACLE_SCOPE_TRACE_ENABLE-}"
oracle_scope_summary_enable="${ORACLE_SCOPE_SUMMARY_ENABLE-}"
oracle_ft_enable="${ORACLE_FT_ENABLE-}"
oracle_ft_gain_init="${ORACLE_FT_GAIN_INIT-}"
oracle_ft_mlp_lr="${ORACLE_FT_MLP_LR-}"
oracle_ft_graph_lr="${ORACLE_FT_GRAPH_LR-}"
oracle_ft_input_proj_lr="${ORACLE_FT_INPUT_PROJ_LR-}"
oracle_ft_train_scope="${ORACLE_FT_TRAIN_SCOPE-}"
oracle_ft_unfreeze_global_encoder="${ORACLE_FT_UNFREEZE_GLOBAL_ENCODER-}"
oracle_ft_unfreeze_input_proj="${ORACLE_FT_UNFREEZE_INPUT_PROJ-}"

cpu_set="${CPU_SET:-}"

echo "[run_oracle_eval.bash] EXP_NAME=${exp_name}"
echo "[run_oracle_eval.bash] EXPERIMENT_VARIANT=${experiment_variant}"
echo "[run_oracle_eval.bash] A3_SCOPE=${a3_scope}"
echo "[run_oracle_eval.bash] CONFIG_PATH=${config_path}"
echo "[run_oracle_eval.bash] CKPT_PATH=${ckpt_path}"
echo "[run_oracle_eval.bash] EPISODE_ID_FILE=${episode_id_file:-<yaml/default>}"
echo "[run_oracle_eval.bash] NUM_ENVIRONMENTS=${num_environments}"
echo "[run_oracle_eval.bash] EPISODE_COUNT=${episode_count}"
echo "[run_oracle_eval.bash] MASTER_PORT=${master_port}"
echo "[run_oracle_eval.bash] CPU_SET=${cpu_set:-<disabled>}"
echo "[run_oracle_eval.bash] EVAL_ENV_REFILL_POLICY=${eval_env_refill_policy:-<yaml/default>}"

append_opt() {
      local key="$1"
      local value="${2-}"
      if [[ -n "${value}" ]]; then
            flag_eval+=("${key}" "${value}")
      fi
}

flag_eval=(
      --exp_name "${exp_name}"
      --run-type eval
      --exp-config "${config_path}"
      SIMULATOR_GPU_IDS "${simulator_gpu_ids}"
      TORCH_GPU_IDS "${torch_gpu_ids}"
      TORCH_GPU_ID "${torch_gpu_id}"
      GPU_NUMBERS "${gpu_numbers}"
      NUM_ENVIRONMENTS "${num_environments}"
      EVAL.CKPT_PATH_DIR "${ckpt_path}"
      EVAL.EPISODE_COUNT "${episode_count}"
      EVAL.fast_eval "${fast_eval}"
      RESULTS_DIR "${results_dir}"
      TENSORBOARD_DIR "${tensorboard_dir}"
      CHECKPOINT_FOLDER "${checkpoint_folder}"
      VIDEO_DIR "${video_dir}"
      LOG_DIR "${log_dir}"
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING "${allow_sliding}"
)

append_opt "EVAL.EPISODE_ID_FILE" "${episode_id_file}"
append_opt "ORACLE.enable" "${oracle_enable}"
append_opt "EVAL.ENV_REFILL_POLICY" "${eval_env_refill_policy}"
append_opt "ORACLE.cache_enable" "${oracle_cache_enable}"
append_opt "ORACLE.enable_in_train" "${oracle_enable_in_train}"
append_opt "ORACLE.enable_in_eval" "${oracle_enable_in_eval}"
append_opt "ORACLE.apply_mode" "${oracle_apply_mode}"
append_opt "ORACLE.soft_alpha" "${oracle_soft_alpha}"
append_opt "ORACLE.target_ghost_scope" "${oracle_target_ghost_scope}"
append_opt "ORACLE.refresh_policy" "${oracle_refresh_policy}"
append_opt "ORACLE.query_heading_strategy" "${oracle_query_heading_strategy}"
append_opt "ORACLE.query_pipeline" "${oracle_query_pipeline}"
append_opt "ORACLE.strict_scope" "${oracle_strict_scope}"
append_opt "ORACLE.shadow_rerun_planner" "${oracle_shadow_rerun_planner}"
append_opt "ORACLE.max_scope_ghosts" "${oracle_max_scope_ghosts}"
append_opt "ORACLE.scope_trace_enable" "${oracle_scope_trace_enable}"
append_opt "ORACLE.scope_summary_enable" "${oracle_scope_summary_enable}"
append_opt "ORACLE.trace.enable" "${oracle_trace_enable}"
append_opt "MODEL.ORACLE_FT.enable" "${oracle_ft_enable}"
append_opt "MODEL.ORACLE_FT.gain_init" "${oracle_ft_gain_init}"
append_opt "MODEL.ORACLE_FT.oracle_mlp_lr" "${oracle_ft_mlp_lr}"
append_opt "MODEL.ORACLE_FT.graph_lr" "${oracle_ft_graph_lr}"
append_opt "MODEL.ORACLE_FT.input_proj_lr" "${oracle_ft_input_proj_lr}"
append_opt "MODEL.ORACLE_FT.train_scope" "${oracle_ft_train_scope}"
append_opt "MODEL.ORACLE_FT.unfreeze_global_encoder" "${oracle_ft_unfreeze_global_encoder}"
append_opt "MODEL.ORACLE_FT.unfreeze_input_proj" "${oracle_ft_unfreeze_input_proj}"

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
run_oracle_eval "${master_port}" "${flag_eval[@]}"
