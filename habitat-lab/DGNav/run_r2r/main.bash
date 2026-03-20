#!/usr/bin/env bash
set -euo pipefail

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
echo "[main.bash] CPU thread caps: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dgnav_dir="$(cd "${script_dir}/.." && pwd)"
preferred_habitat_repo_root="$(cd "${dgnav_dir}/.." && pwd -P)"

if [[ -n "${DGNAV_HABITAT_REPO_ROOT:-}" ]]; then
      if ! requested_habitat_repo_root="$(cd "${DGNAV_HABITAT_REPO_ROOT}" && pwd -P 2>/dev/null)"; then
            echo "[main.bash] Invalid DGNAV_HABITAT_REPO_ROOT=${DGNAV_HABITAT_REPO_ROOT}" >&2
            exit 1
      fi
      if [[ "${requested_habitat_repo_root}" != "${preferred_habitat_repo_root}" ]]; then
            echo "[main.bash] Refusing Habitat repo root ${requested_habitat_repo_root}" >&2
            echo "[main.bash] Expected clean worktree root ${preferred_habitat_repo_root}" >&2
            exit 1
      fi
fi

habitat_repo_root="${preferred_habitat_repo_root}"
if [[ ! -f "${habitat_repo_root}/habitat-baselines/habitat_baselines/common/environments.py" ]]; then
      echo "[main.bash] Missing required file: ${habitat_repo_root}/habitat-baselines/habitat_baselines/common/environments.py" >&2
      exit 1
fi

export DGNAV_HABITAT_REPO_ROOT="${habitat_repo_root}"
export PYTHONPATH="${habitat_repo_root}/habitat-lab:${habitat_repo_root}/habitat-baselines:${PYTHONPATH:-}"
echo "[main.bash] Using Habitat-Lab/Baselines from ${habitat_repo_root}"

usage() {
      cat <<'EOF'
用法:
  bash habitat-lab/DGNav/run_r2r/main.bash <preset> [CONFIG_OVERRIDE...]

唯一必填参数:
  <preset> 选择实验配置块。

可用 preset:
  batch_oracle_eval_stream_cache
  serial_oracle_eval_stream_cache

默认对齐口径:
  - ckpt.iter18600
  - Oracle 打开
  - soft 注入 alpha=0.25
  - cache 打开
  - streaming_refill 打开
  - NUM_ENVIRONMENTS=4
  - fixed500 val_unseen

命令行覆盖:
  直接在 preset 后面追加 run.py 支持的 KEY VALUE 覆盖，例如:
    bash habitat-lab/DGNav/run_r2r/main.bash batch_oracle_eval_stream_cache \
      EVAL.EPISODE_COUNT 10 ORACLE.trace.enable False
EOF
}

if [[ $# -lt 1 ]]; then
      usage >&2
      exit 1
fi

preset="$1"
shift
extra_opts=("$@")

conda_env="${CONDA_ENV:-py3-9}"
dist_launch_module="${DIST_LAUNCH_MODULE:-torch.distributed.run}"
gpu_numbers="${GPU_NUMBERS:-1}"
simulator_gpu_ids="${SIMULATOR_GPU_IDS:-[0]}"
torch_gpu_ids="${TORCH_GPU_IDS:-[0]}"
torch_gpu_id="${TORCH_GPU_ID:-0}"
allow_sliding="${ALLOW_SLIDING:-True}"
results_dir="${RESULTS_DIR:-data/logs/eval_results/}"
tensorboard_dir="${TENSORBOARD_DIR:-data/logs/tensorboard_dirs/}"
checkpoint_folder="${CHECKPOINT_FOLDER:-data/logs/checkpoints/}"
video_dir="${VIDEO_DIR:-data/logs/video/}"
log_dir="${LOG_DIR:-data/logs/running_log/}"

release_ckpt_iter18600="${dgnav_dir}/data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth"
fixed500_file="run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt"
oracle_stack="run_r2r/r2r_oracle.yaml"

run_type="eval"
exp_name=""
exp_config="${oracle_stack}"
preset_master_port=""
preset_num_environments="4"
preset_opts=()

case "${preset}" in
      A1)
            exp_name="batch_oracle_eval_stream_cache"
            preset_master_port="4851"
            preset_opts=(
                  "EVAL.CKPT_PATH_DIR" "${release_ckpt_iter18600}"
                  "EVAL.EPISODE_ID_FILE" "${fixed500_file}"
                  "EVAL.ENV_REFILL_POLICY" "streaming_refill"
                  "IL.back_algo" "control"
                  "ORACLE.enable" "True"
                  "ORACLE.enable_in_eval" "True"
                  "ORACLE.apply_mode" "soft"
                  "ORACLE.soft_alpha" "0.25"
                  "ORACLE.cache_enable" "True"
                  "ORACLE.batch_query_enable" "True"
                  "MODEL.ORACLE_FT.enable" "False"
            )
            ;;
      A2)
            exp_name="serial_oracle_eval_stream_cache"
            preset_master_port="4852"
            preset_opts=(
                  "EVAL.CKPT_PATH_DIR" "${release_ckpt_iter18600}"
                  "EVAL.EPISODE_ID_FILE" "${fixed500_file}"
                  "EVAL.ENV_REFILL_POLICY" "streaming_refill"
                  "IL.back_algo" "control"
                  "ORACLE.enable" "True"
                  "ORACLE.enable_in_eval" "True"
                  "ORACLE.apply_mode" "soft"
                  "ORACLE.soft_alpha" "0.25"
                  "ORACLE.cache_enable" "True"
                  "ORACLE.batch_query_enable" "False"
                  "MODEL.ORACLE_FT.enable" "False"
            )
            ;;
      *)
            echo "[main.bash] Unknown preset: ${preset}" >&2
            usage >&2
            exit 1
            ;;
esac

master_port="${MASTER_PORT:-${preset_master_port}}"
num_environments="${NUM_ENVIRONMENTS:-${preset_num_environments}}"

common_opts=(
      "SIMULATOR_GPU_IDS" "${simulator_gpu_ids}"
      "TORCH_GPU_IDS" "${torch_gpu_ids}"
      "TORCH_GPU_ID" "${torch_gpu_id}"
      "GPU_NUMBERS" "${gpu_numbers}"
      "NUM_ENVIRONMENTS" "${num_environments}"
      "RESULTS_DIR" "${results_dir}"
      "TENSORBOARD_DIR" "${tensorboard_dir}"
      "CHECKPOINT_FOLDER" "${checkpoint_folder}"
      "VIDEO_DIR" "${video_dir}"
      "LOG_DIR" "${log_dir}"
      "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING" "${allow_sliding}"
)

if [[ ! -f "${release_ckpt_iter18600}" ]]; then
      echo "[main.bash] Missing checkpoint: ${release_ckpt_iter18600}" >&2
      exit 1
fi

run_dgnav() {
      local -a cmd=(
            python -m "${dist_launch_module}"
            --nproc_per_node=1
            --master_port "${master_port}"
            run.py
            --exp_name "${exp_name}"
            --run-type "${run_type}"
            --exp-config "${exp_config}"
      )
      cmd+=("${common_opts[@]}")
      cmd+=("${preset_opts[@]}")
      cmd+=("${extra_opts[@]}")

      echo "[main.bash] preset=${preset}"
      echo "[main.bash] run_type=${run_type}"
      echo "[main.bash] exp_name=${exp_name}"
      echo "[main.bash] exp_config=${exp_config}"
      echo "[main.bash] master_port=${master_port}"
      echo "[main.bash] num_environments=${num_environments}"

      if [[ "${CONDA_DEFAULT_ENV:-}" == "${conda_env}" ]]; then
            echo "[main.bash] Using current conda env: ${CONDA_DEFAULT_ENV}"
            "${cmd[@]}"
      else
            echo "[main.bash] Using conda run -n ${conda_env} --no-capture-output"
            conda run --no-capture-output -n "${conda_env}" "${cmd[@]}"
      fi
}

cd "${dgnav_dir}"
run_dgnav
