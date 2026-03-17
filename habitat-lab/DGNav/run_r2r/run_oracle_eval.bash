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
experiment_variant="${EXPERIMENT_VARIANT:-oracle_cache}"

master_port="${MASTER_PORT:-4565}"

config_path="${CONFIG_PATH:-run_r2r/eval_oracle_o1.yaml}"
ckpt_path="${CKPT_PATH:-/home/gwl/project/DGNav_new/habitat-lab/DGNav/data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter21800.pth}"
episode_id_file="${EPISODE_ID_FILE:-run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt}"
num_environments="${NUM_ENVIRONMENTS:-4}"
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

cpu_set="${CPU_SET:-}"

case "${experiment_variant}" in
      oracle_cache)
            default_exp_name="fixed500_oracle_cache"
            oracle_enable="True"
            oracle_cache_enable="True"
            ;;
      oracle)
            default_exp_name="fixed500_oracle"
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
echo "[run_oracle_eval.bash] CONFIG_PATH=${config_path}"
echo "[run_oracle_eval.bash] CKPT_PATH=${ckpt_path}"
echo "[run_oracle_eval.bash] EPISODE_ID_FILE=${episode_id_file}"
echo "[run_oracle_eval.bash] NUM_ENVIRONMENTS=${num_environments}"
echo "[run_oracle_eval.bash] EPISODE_COUNT=${episode_count}"
echo "[run_oracle_eval.bash] MASTER_PORT=${master_port}"
echo "[run_oracle_eval.bash] ORACLE.enable=${oracle_enable}"
echo "[run_oracle_eval.bash] ORACLE.cache_enable=${oracle_cache_enable}"
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
