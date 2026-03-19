#!/usr/bin/env bash
set -euo pipefail

# 底层训练入口。
# 日常建议直接使用 run_r2r/run_oracle_experiment.bash 里的命名预设。

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
echo "[run_oracle_ft_train.bash] CPU thread caps: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dgnav_dir="$(cd "${script_dir}/.." && pwd)"
habitat_repo_root="$(cd "${dgnav_dir}/.." && pwd -P)"
export DGNAV_HABITAT_REPO_ROOT="${habitat_repo_root}"
export PYTHONPATH="${habitat_repo_root}/habitat-lab:${habitat_repo_root}/habitat-baselines:${PYTHONPATH:-}"
echo "[run_oracle_ft_train.bash] Using Habitat-Lab/Baselines from ${habitat_repo_root}"

dist_launch_module="torch.distributed.launch"
if python -c "import importlib.util as u; raise SystemExit(0 if u.find_spec('torch.distributed.run') else 1)"; then
      dist_launch_module="torch.distributed.run"
fi

conda_env="${CONDA_ENV:-py3-9}"
master_port="${MASTER_PORT:-4561}"
exp_name="${EXP_NAME:-B1_oracle_ft_adapter_only}"
config_path="${CONFIG_PATH:-run_r2r/iter_train.yaml,run_r2r/train_oracle_ft_base.yaml,run_r2r/oracle_ft_adapter_only.yaml}"
ckpt_path="${CKPT_PATH:-${dgnav_dir}/data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter12000.pth}"
pretrained_path="${PRETRAINED_PATH-}"

num_environments="${NUM_ENVIRONMENTS:-1}"
gpu_numbers="${GPU_NUMBERS:-1}"
simulator_gpu_ids="${SIMULATOR_GPU_IDS:-[0]}"
torch_gpu_ids="${TORCH_GPU_IDS:-[0]}"
torch_gpu_id="${TORCH_GPU_ID:-0}"

results_dir="${RESULTS_DIR:-data/logs/eval_results/}"
tensorboard_dir="${TENSORBOARD_DIR:-data/logs/tensorboard_dirs/}"
checkpoint_folder="${CHECKPOINT_FOLDER:-data/logs/checkpoints/}"
video_dir="${VIDEO_DIR:-data/logs/video/}"
log_dir="${LOG_DIR:-data/logs/running_log/}"
allow_sliding="${ALLOW_SLIDING:-True}"

# 可选覆盖项。
# 变量未设置时，保持 YAML 配置栈中的原值，不额外覆盖。
oracle_enable="${ORACLE_ENABLE-}"
oracle_enable_in_train="${ORACLE_ENABLE_IN_TRAIN-}"
oracle_enable_in_eval="${ORACLE_ENABLE_IN_EVAL-}"
oracle_apply_mode="${ORACLE_APPLY_MODE-}"
oracle_soft_alpha="${ORACLE_SOFT_ALPHA-}"
oracle_target_ghost_scope="${ORACLE_TARGET_GHOST_SCOPE-}"
oracle_refresh_policy="${ORACLE_REFRESH_POLICY-}"
oracle_cache_enable="${ORACLE_CACHE_ENABLE-}"
oracle_trace_enable="${ORACLE_TRACE_ENABLE-}"
oracle_scope_trace_enable="${ORACLE_SCOPE_TRACE_ENABLE-}"
oracle_scope_summary_enable="${ORACLE_SCOPE_SUMMARY_ENABLE-}"
oracle_strict_scope="${ORACLE_STRICT_SCOPE-}"

oracle_ft_enable="${ORACLE_FT_ENABLE-}"
oracle_ft_gain_init="${ORACLE_FT_GAIN_INIT-}"
oracle_ft_mlp_lr="${ORACLE_FT_MLP_LR-}"
oracle_ft_graph_lr="${ORACLE_FT_GRAPH_LR-}"
oracle_ft_input_proj_lr="${ORACLE_FT_INPUT_PROJ_LR-}"
oracle_ft_train_scope="${ORACLE_FT_TRAIN_SCOPE-}"
oracle_ft_unfreeze_global_encoder="${ORACLE_FT_UNFREEZE_GLOBAL_ENCODER-}"
oracle_ft_unfreeze_input_proj="${ORACLE_FT_UNFREEZE_INPUT_PROJ-}"
il_load_from_ckpt="${IL_LOAD_FROM_CKPT:-True}"
il_iters="${IL_ITERS-}"
il_lr="${IL_LR-}"
il_waypoint_aug="${IL_WAYPOINT_AUG-}"
il_back_algo="${IL_BACK_ALGO-}"
train_env_refill_policy="${TRAIN_ENV_REFILL_POLICY-}"

echo "[run_oracle_ft_train.bash] EXP_NAME=${exp_name}"
echo "[run_oracle_ft_train.bash] CONFIG_PATH=${config_path}"
echo "[run_oracle_ft_train.bash] CKPT_PATH=${ckpt_path}"
echo "[run_oracle_ft_train.bash] PRETRAINED_PATH=${pretrained_path:-<yaml/default>}"
echo "[run_oracle_ft_train.bash] NUM_ENVIRONMENTS=${num_environments}"
echo "[run_oracle_ft_train.bash] MASTER_PORT=${master_port}"
echo "[run_oracle_ft_train.bash] TRAIN_ENV_REFILL_POLICY=${train_env_refill_policy:-<yaml/default>}"

append_opt() {
      local key="$1"
      local value="${2-}"
      if [[ -n "${value}" ]]; then
            flag_train+=("${key}" "${value}")
      fi
}

flag_train=(
      --exp_name "${exp_name}"
      --run-type train
      --exp-config "${config_path}"
      SIMULATOR_GPU_IDS "${simulator_gpu_ids}"
      TORCH_GPU_IDS "${torch_gpu_ids}"
      TORCH_GPU_ID "${torch_gpu_id}"
      GPU_NUMBERS "${gpu_numbers}"
      NUM_ENVIRONMENTS "${num_environments}"
      RESULTS_DIR "${results_dir}"
      TENSORBOARD_DIR "${tensorboard_dir}"
      CHECKPOINT_FOLDER "${checkpoint_folder}"
      VIDEO_DIR "${video_dir}"
      LOG_DIR "${log_dir}"
      IL.is_requeue False
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING "${allow_sliding}"
)

append_opt "IL.load_from_ckpt" "${il_load_from_ckpt}"
append_opt "IL.ckpt_to_load" "${ckpt_path}"
append_opt "MODEL.pretrained_path" "${pretrained_path}"
append_opt "ORACLE.enable" "${oracle_enable}"
append_opt "ORACLE.enable_in_train" "${oracle_enable_in_train}"
append_opt "ORACLE.enable_in_eval" "${oracle_enable_in_eval}"
append_opt "ORACLE.apply_mode" "${oracle_apply_mode}"
append_opt "ORACLE.soft_alpha" "${oracle_soft_alpha}"
append_opt "ORACLE.target_ghost_scope" "${oracle_target_ghost_scope}"
append_opt "ORACLE.refresh_policy" "${oracle_refresh_policy}"
append_opt "ORACLE.cache_enable" "${oracle_cache_enable}"
append_opt "ORACLE.trace.enable" "${oracle_trace_enable}"
append_opt "ORACLE.scope_trace_enable" "${oracle_scope_trace_enable}"
append_opt "ORACLE.scope_summary_enable" "${oracle_scope_summary_enable}"
append_opt "ORACLE.strict_scope" "${oracle_strict_scope}"
append_opt "MODEL.ORACLE_FT.enable" "${oracle_ft_enable}"
append_opt "MODEL.ORACLE_FT.gain_init" "${oracle_ft_gain_init}"
append_opt "MODEL.ORACLE_FT.oracle_mlp_lr" "${oracle_ft_mlp_lr}"
append_opt "MODEL.ORACLE_FT.graph_lr" "${oracle_ft_graph_lr}"
append_opt "MODEL.ORACLE_FT.input_proj_lr" "${oracle_ft_input_proj_lr}"
append_opt "MODEL.ORACLE_FT.train_scope" "${oracle_ft_train_scope}"
append_opt "MODEL.ORACLE_FT.unfreeze_global_encoder" "${oracle_ft_unfreeze_global_encoder}"
append_opt "MODEL.ORACLE_FT.unfreeze_input_proj" "${oracle_ft_unfreeze_input_proj}"
append_opt "IL.iters" "${il_iters}"
append_opt "IL.lr" "${il_lr}"
append_opt "IL.waypoint_aug" "${il_waypoint_aug}"
append_opt "IL.back_algo" "${il_back_algo}"
append_opt "IL.TRAIN_ENV_REFILL_POLICY" "${train_env_refill_policy}"

run_oracle_ft_train() {
      local port="$1"
      shift
      if [[ "${CONDA_DEFAULT_ENV:-}" == "${conda_env}" ]]; then
            echo "[run_oracle_ft_train.bash] Using current conda env: ${CONDA_DEFAULT_ENV}"
            python -m "${dist_launch_module}" --nproc_per_node=1 --master_port "${port}" run.py "$@"
      else
            echo "[run_oracle_ft_train.bash] Using conda run -n ${conda_env} --no-capture-output"
            conda run --no-capture-output -n "${conda_env}" \
                  python -m "${dist_launch_module}" --nproc_per_node=1 --master_port "${port}" run.py "$@"
      fi
}

cd "${dgnav_dir}"

echo "###### oracle ft train mode ######"
run_oracle_ft_train "${master_port}" "${flag_train[@]}"

#MPLCONFIGDIR=/tmp/mpl CONDA_ENV=py3-9 EXP_NAME=release_r2r_dino_best_nav_streaming_full CONFIG_PATH=run_r2r/iter_train.yaml,run_r2r/train_streaming_refill_smoke.yaml PRETRAINED_PATH=/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/pretrained/r2r_ce/mlm.sap_habitat_depth_dinov2_clean/ckpts/model_step_97500.pt NUM_ENVIRONMENTS=4 GPU_NUMBERS=1 SIMULATOR_GPU_IDS='[0]' TORCH_GPU_IDS='[0]' TORCH_GPU_ID=0 MASTER_PORT=4793 IL_ITERS=30000 IL_LR=1e-5 IL_LOAD_FROM_CKPT=False IL_WAYPOINT_AUG=True IL_BACK_ALGO=teleport TRAIN_ENV_REFILL_POLICY=streaming_refill ALLOW_SLIDING=True ORACLE_ENABLE=False ORACLE_ENABLE_IN_TRAIN=False ORACLE_ENABLE_IN_EVAL=False bash habitat-lab/DGNav/run_r2r/run_oracle_ft_train.bash