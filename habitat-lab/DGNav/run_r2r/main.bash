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
  A1  batch Oracle eval, streaming_refill + cache + batch_query, full val_unseen
  A2  serial Oracle eval, streaming_refill + cache
  A3  batch Oracle eval, streaming_refill + cache + batch_query, full val_unseen
  T1  Oracle train, streaming_refill + cache + batch_query + Oracle-FT

默认对齐口径:
  - A1:
    预填 top-15 full-val ckpt shortlist
    Oracle 打开
    soft 注入 alpha=0.25
    cache 打开
    streaming_refill 打开
    NUM_ENVIRONMENTS=4
    full val_unseen
  - A2:
    与 A1 相同
    fixed500 val_unseen
  - A3:
    与 A1 相同
    full val_unseen
  - T1:
    best_nav 同口径训练超参
    从 best_nav 预训练底座开始
    Oracle train/eval 打开
    cache + batch_query 打开
    TRAIN_ENV_REFILL_POLICY=streaming_refill
    NUM_ENVIRONMENTS=6
    IL.log_every=100（每 100 iter 保存一次 ckpt）
命令行覆盖:
  直接在 preset 后面追加 run.py 支持的 KEY VALUE 覆盖，例如:
    bash habitat-lab/DGNav/run_r2r/main.bash A1 \
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

ensure_trailing_slash() {
      local path="$1"
      if [[ -z "${path}" ]]; then
            printf "%s" "${path}"
            return
      fi
      if [[ "${path}" == */ ]]; then
            printf "%s" "${path}"
      else
            printf "%s/" "${path}"
      fi
}

to_python_list_literal() {
      local out="["
      local first=1
      local item=""
      for item in "$@"; do
            if (( first == 0 )); then
                  out+=", "
            fi
            out+="'${item}'"
            first=0
      done
      out+="]"
      printf "%s" "${out}"
}

results_dir="$(ensure_trailing_slash "${results_dir}")"
tensorboard_dir="$(ensure_trailing_slash "${tensorboard_dir}")"
checkpoint_folder="$(ensure_trailing_slash "${checkpoint_folder}")"
video_dir="$(ensure_trailing_slash "${video_dir}")"
log_dir="$(ensure_trailing_slash "${log_dir}")"

release_ckpt_iter18600="${dgnav_dir}/data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth"
best_nav_pretrain_base="/home/gwl/project/DGNav_new/habitat-lab/DGNav/pretrained/r2r_ce/mlm.sap_habitat_depth_dinov2s/ckpts/model_step_97500.pt"
oracle_train_resume_ckpt="${dgnav_dir}/data/logs/checkpoints/oracle_train_stream_batch_cache_ft/ckpt.iter4900.pth"
fixed500_file="run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt"
fixed500_file_path="${dgnav_dir}/${fixed500_file}"
oracle_stack="run_r2r/r2r_oracle.yaml"

run_type="eval"
exp_name=""
exp_config="${oracle_stack}"
preset_master_port=""
preset_num_environments="8"
preset_opts=()
required_paths=()

case "${preset}" in
      A1)
            a1_ckpt_dir="/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/oracle_all_pool2"
            a1_ckpt_list=(
                  "ckpt.iter15400.pth"
                  "ckpt.iter15500.pth"
                  "ckpt.iter16100.pth"
                  "ckpt.iter16200.pth"
                  "ckpt.iter16300.pth"
                  "ckpt.iter18100.pth"
                  "ckpt.iter18200.pth"
                  "ckpt.iter18300.pth"
                  "ckpt.iter18900.pth"
                  "ckpt.iter19000.pth"
                  "ckpt.iter19100.pth"
            )
            a1_ckpt_list_literal="$(to_python_list_literal "${a1_ckpt_list[@]}")"
            exp_name="oracle_all_pool2_full_selected"
            preset_master_port="4851"
            preset_opts=(
                  "EVAL.CKPT_PATH_DIR" "${a1_ckpt_dir}"
                  "EVAL.CKPT_PATH_LIST" "${a1_ckpt_list_literal}"
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
            for a1_ckpt_name in "${a1_ckpt_list[@]}"; do
                  required_paths+=("${a1_ckpt_dir}/${a1_ckpt_name}")
            done
            ;;
      A3)
            exp_name="oracle_train_stream_batch_cache_ft_full_unseen"
            preset_master_port="4853"
            preset_opts=(
                  "EVAL.CKPT_PATH_DIR" "/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/oracle_all_pool"
                  # "EVAL.CKPT_PATH_DIR" "${release_ckpt_iter18600}"
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
            required_paths+=("${release_ckpt_iter18600}")
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
            required_paths+=("${release_ckpt_iter18600}" "${fixed500_file_path}")
            ;;
      T1)
            run_type="train"
            exp_name="oracle_all_pool2"
            preset_master_port="4861"
            preset_num_environments="6"
            preset_opts=(
                  "IL.iters" "20000"
                  "IL.log_every" "100"
                  "IL.lr" "1e-5"
                  "IL.sample_ratio" "0.75"
                  "IL.decay_interval" "3000"
                  "IL.waypoint_aug" "True"
                  "IL.load_from_ckpt" "False"
                  "IL.is_requeue" "False"
                  "IL.ckpt_to_load" "/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/oracle_all_pool"
                  "IL.TRAIN_ENV_REFILL_POLICY" "streaming_refill"
                  "IL.TRAIN_STATIC_SCENE_POOLS_ENABLE" "True"
                  "IL.TRAIN_SLOW_SCENES" "['gTV8FGcVJC9','VzqfbhrpDEA']"
                  "IL.TRAIN_FAST_POOL_NUM_ENVS" "4"
                  "IL.TRAIN_SLOW_POOL_NUM_ENVS" "2"
                  "IL.TRAIN_POOL_FAST_ITERS" "24"
                  "IL.TRAIN_POOL_SLOW_ITERS" "2"
                  "MODEL.pretrained_path" "${best_nav_pretrain_base}"
                  "ORACLE.enable" "True"
                  "ORACLE.enable_in_train" "True"
                  "ORACLE.enable_in_eval" "True"
                  "ORACLE.apply_mode" "soft"
                  "ORACLE.soft_alpha" "0.25"
                  "ORACLE.cache_enable" "True"
                  "ORACLE.batch_query_enable" "True"
                  "ORACLE.trace.enable" "False"
                  "ORACLE.scope_trace_enable" "False"
                  "ORACLE.scope_summary_enable" "False"
                  "MODEL.ORACLE_FT.enable" "True"
                  "MODEL.ORACLE_FT.train_scope" "baseline_plus_oracle_adapter"
            )
            required_paths+=("${best_nav_pretrain_base}" "${oracle_train_resume_ckpt}")
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

for required_path in "${required_paths[@]}"; do
      if [[ ! -f "${required_path}" ]]; then
            echo "[main.bash] Missing required file: ${required_path}" >&2
            exit 1
      fi
done

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
