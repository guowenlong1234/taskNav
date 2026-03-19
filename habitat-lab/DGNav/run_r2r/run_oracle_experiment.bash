#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dgnav_dir="$(cd "${script_dir}/.." && pwd)"
train_entry="${script_dir}/run_oracle_ft_train.bash"
eval_entry="${script_dir}/run_oracle_eval.bash"

usage() {
      cat <<'EOF'
用法:
  bash habitat-lab/DGNav/run_r2r/run_oracle_experiment.bash <preset>

对外只暴露一个参数:
  <preset> 用来选择本文件里定义好的完整实验预设。

可用预设:
  b0_train
  b1_train
  b1_eval_on_fixed500
  b1_eval_off_fixed500
  b1_eval_on_full
  b1_eval_off_full
  b2_train
  b2_eval_on_fixed500
  b2_eval_off_fixed500
  b2_eval_on_full
  b2_eval_off_full
EOF
}

if [[ $# -ne 1 ]]; then
      usage >&2
      exit 1
fi

preset="$1"

# 共享运行参数。
# CONDA_ENV: 运行底层脚本时使用的 conda 环境。
CONDA_ENV="py3-9"
# NUM_ENVIRONMENTS: Habitat VectorEnv 数量。Oracle-FT 调试期建议保持 1。
NUM_ENVIRONMENTS="1"
# GPU_NUMBERS: torch.distributed.run 启动的进程数。
GPU_NUMBERS="1"
# SIMULATOR_GPU_IDS: Habitat simulator 使用的 GPU。
SIMULATOR_GPU_IDS="[0]"
# TORCH_GPU_IDS / TORCH_GPU_ID: 当前进程的 torch 设备编号。
TORCH_GPU_IDS="[0]"
TORCH_GPU_ID="0"
# ALLOW_SLIDING: 保持与当前 control/eval 口径一致。
ALLOW_SLIDING="True"
# 输出目录根路径。run.py 会在合适位置追加 EXP_NAME。
RESULTS_DIR="data/logs/eval_results/"
TENSORBOARD_DIR="data/logs/tensorboard_dirs/"
CHECKPOINT_FOLDER="data/logs/checkpoints/"
VIDEO_DIR="data/logs/video/"
LOG_DIR="data/logs/running_log/"

# 共享 checkpoint 目录。
# 具体使用哪个 checkpoint，在各自 preset 配置块里显式写出。
release_ckpt_dir="${dgnav_dir}/data/logs/checkpoints/release_r2r_dino_best_nav"

require_existing_file() {
      local path="$1"
      local label="$2"
      if [[ -z "${path}" ]]; then
            echo "[run_oracle_experiment.bash] ${label} is empty. Edit this file first." >&2
            exit 1
      fi
      if [[ ! -f "${path}" ]]; then
            echo "[run_oracle_experiment.bash] ${label} not found: ${path}" >&2
            exit 1
      fi
}

runner=""
master_port=""
exp_name=""
config_path=""
ckpt_path=""
train_iters=""
num_environments=""
oracle_enable=""
oracle_enable_in_train=""
oracle_enable_in_eval=""
oracle_apply_mode=""
oracle_soft_alpha=""
oracle_target_ghost_scope=""
oracle_refresh_policy=""
oracle_cache_enable=""
oracle_trace_enable=""
oracle_scope_trace_enable=""
oracle_scope_summary_enable=""
oracle_strict_scope=""
oracle_ft_enable=""
oracle_ft_gain_init=""
oracle_ft_mlp_lr=""
oracle_ft_graph_lr=""
oracle_ft_input_proj_lr=""
oracle_ft_unfreeze_global_encoder=""
oracle_ft_unfreeze_input_proj=""

case "${preset}" in
      b0_train)
            # B0 冒烟实验:
            # - 训练态打开 Oracle query/writeback
            # - 关闭 Oracle-FT adapter
            # - 只验证训练链路是否打通
            runner="${train_entry}"
            master_port="4760"
            exp_name="B0_oracle_train_smoke"
            config_path="run_r2r/iter_train.yaml,run_r2r/train_oracle_ft_base.yaml,run_r2r/oracle_ft_disabled.yaml"
            ckpt_path="${release_ckpt_dir}/ckpt.iter18600.pth"
            train_iters="200"
            num_environments="1"
            oracle_enable="True"
            oracle_enable_in_train="True"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_ft_enable="False"
            oracle_ft_unfreeze_global_encoder="False"
            oracle_ft_unfreeze_input_proj="False"
            ;;
      b1_train)
            # B1:
            # - Oracle 打开
            # - soft alpha = 0.25
            # - cache 打开
            # - 只训练 oracle_adapter
            runner="${train_entry}"
            master_port="4761"
            exp_name="B1_oracle_ft_adapter_only"
            config_path="run_r2r/iter_train.yaml,run_r2r/train_oracle_ft_base.yaml,run_r2r/oracle_ft_adapter_only.yaml"
            ckpt_path="${release_ckpt_dir}/ckpt.iter18600.pth"
            train_iters="20000"
            num_environments="1"
            oracle_enable="True"
            oracle_enable_in_train="True"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_trace_enable="True"
            oracle_scope_trace_enable="True"
            oracle_scope_summary_enable="True"
            oracle_ft_enable="True"
            oracle_ft_gain_init="1.0"
            oracle_ft_mlp_lr="5e-5"
            oracle_ft_graph_lr="5e-6"
            oracle_ft_input_proj_lr="1e-5"
            oracle_ft_unfreeze_global_encoder="False"
            oracle_ft_unfreeze_input_proj="False"
            ;;
      b1_eval_on_fixed500)
            # B1 fixed500 评测:
            # - 与 B1 train 保持同一组关键 Oracle / Oracle-FT 开关
            # - 使用 B1 微调后的 checkpoint
            # - Oracle 保持开启
            runner="${eval_entry}"
            master_port="4762"
            exp_name="B1_eval_oracle_on_fixed500"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_adapter_only.yaml,run_r2r/eval_fixed500.yaml"
            # 显式填写 B1 训练产出的待评测 checkpoint。
            ckpt_path="/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/B1_oracle_ft_adapter_only/ckpt.iter20000.pth"
            num_environments="1"
            oracle_enable="True"
            oracle_enable_in_train="True"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_trace_enable="True"
            oracle_scope_trace_enable="True"
            oracle_scope_summary_enable="True"
            oracle_ft_enable="True"
            oracle_ft_gain_init="1.0"
            oracle_ft_mlp_lr="5e-5"
            oracle_ft_graph_lr="5e-6"
            oracle_ft_input_proj_lr="1e-5"
            oracle_ft_unfreeze_global_encoder="False"
            oracle_ft_unfreeze_input_proj="False"
            ;;
      b1_eval_off_fixed500)
            # B1 fixed500 评测, Oracle 关闭:
            # - 禁止 Oracle query
            # - 保留 Oracle-FT 模块，使微调权重仍然加载
            runner="${eval_entry}"
            master_port="4763"
            exp_name="B1_eval_oracle_off_fixed500"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_eval_off.yaml,run_r2r/eval_fixed500.yaml"
            # 显式填写 B1 训练产出的待评测 checkpoint。
            ckpt_path=""
            num_environments="1"
            ;;
      b1_eval_on_full)
            # B1 full val_unseen 评测:
            # - 与 B1 train 保持同一组关键 Oracle / Oracle-FT 开关
            # - Oracle 保持开启
            runner="${eval_entry}"
            master_port="4764"
            exp_name="B1_eval_oracle_on_full"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_adapter_only.yaml"
            # 显式填写 B1 训练产出的待评测 checkpoint。
            ckpt_path=""
            num_environments="1"
            oracle_enable="True"
            oracle_enable_in_train="True"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_trace_enable="True"
            oracle_scope_trace_enable="True"
            oracle_scope_summary_enable="True"
            oracle_ft_enable="True"
            oracle_ft_gain_init="1.0"
            oracle_ft_mlp_lr="5e-5"
            oracle_ft_graph_lr="5e-6"
            oracle_ft_input_proj_lr="1e-5"
            oracle_ft_unfreeze_global_encoder="False"
            oracle_ft_unfreeze_input_proj="False"
            ;;
      b1_eval_off_full)
            # B1 full val_unseen 评测, Oracle 关闭。
            runner="${eval_entry}"
            master_port="4765"
            exp_name="B1_eval_oracle_off_full"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_eval_off.yaml"
            # 显式填写 B1 训练产出的待评测 checkpoint。
            ckpt_path=""
            num_environments="1"
            ;;
      b2_train)
            # B2:
            # - Oracle 打开
            # - soft alpha = 0.25
            # - cache 打开
            # - 训练 oracle_adapter + global_encoder x_layers
            runner="${train_entry}"
            master_port="4771"
            exp_name="B2_oracle_ft_xlayers"
            config_path="run_r2r/iter_train.yaml,run_r2r/train_oracle_ft_base.yaml,run_r2r/oracle_ft_xlayers.yaml"
            ckpt_path="/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/B2_oracle_ft_xlayers/ckpt.iter20000.pth"
            train_iters="20000"
            num_environments="1"
            oracle_enable="True"
            oracle_enable_in_train="True"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_trace_enable="True"
            oracle_scope_trace_enable="True"
            oracle_scope_summary_enable="True"
            oracle_ft_enable="True"
            oracle_ft_gain_init="1.0"
            oracle_ft_mlp_lr="5e-5"
            oracle_ft_graph_lr="5e-6"
            oracle_ft_input_proj_lr="1e-5"
            oracle_ft_unfreeze_global_encoder="True"
            oracle_ft_unfreeze_input_proj="False"
            ;;
      b2_eval_on_fixed500)
            # B2 fixed500 评测:
            # - 与 B2 train 保持同一组关键 Oracle / Oracle-FT 开关
            # - Oracle 保持开启
            runner="${eval_entry}"
            master_port="4772"
            exp_name="B2_eval_oracle_on_fixed500"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_xlayers.yaml,run_r2r/eval_fixed500.yaml"
            # 显式填写 B2 训练产出的待评测 checkpoint。
            ckpt_path="/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/B2_oracle_ft_xlayers/ckpt.iter20000.pth"
            num_environments="1"
            oracle_enable="True"
            oracle_enable_in_train="True"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_trace_enable="True"
            oracle_scope_trace_enable="True"
            oracle_scope_summary_enable="True"
            oracle_ft_enable="True"
            oracle_ft_gain_init="1.0"
            oracle_ft_mlp_lr="5e-5"
            oracle_ft_graph_lr="5e-6"
            oracle_ft_input_proj_lr="1e-5"
            oracle_ft_unfreeze_global_encoder="True"
            oracle_ft_unfreeze_input_proj="False"
            ;;
      b2_eval_off_fixed500)
            # B2 fixed500 评测, Oracle 关闭。
            runner="${eval_entry}"
            master_port="4773"
            exp_name="B2_eval_oracle_off_fixed500"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_eval_off.yaml,run_r2r/eval_fixed500.yaml"
            # 显式填写 B2 训练产出的待评测 checkpoint。
            ckpt_path=""
            num_environments="1"
            ;;
      b2_eval_on_full)
            # B2 full val_unseen 评测:
            # - 与 B2 train 保持同一组关键 Oracle / Oracle-FT 开关
            # - Oracle 保持开启
            runner="${eval_entry}"
            master_port="4774"
            exp_name="B2_eval_oracle_on_full"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_xlayers.yaml"
            # 显式填写 B2 训练产出的待评测 checkpoint。
            ckpt_path=""
            num_environments="1"
            oracle_enable="True"
            oracle_enable_in_train="True"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_trace_enable="True"
            oracle_scope_trace_enable="True"
            oracle_scope_summary_enable="True"
            oracle_ft_enable="True"
            oracle_ft_gain_init="1.0"
            oracle_ft_mlp_lr="5e-5"
            oracle_ft_graph_lr="5e-6"
            oracle_ft_input_proj_lr="1e-5"
            oracle_ft_unfreeze_global_encoder="True"
            oracle_ft_unfreeze_input_proj="False"
            ;;
      b2_eval_off_full)
            # B2 full val_unseen 评测, Oracle 关闭。
            runner="${eval_entry}"
            master_port="4775"
            exp_name="B2_eval_oracle_off_full"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_eval_off.yaml"
            # 显式填写 B2 训练产出的待评测 checkpoint。
            ckpt_path=""
            num_environments="1"
            ;;
      *)
            echo "[run_oracle_experiment.bash] 未知预设: ${preset}" >&2
            usage >&2
            exit 1
            ;;
esac

require_existing_file "${ckpt_path}" "CKPT_PATH"

echo "[run_oracle_experiment.bash] preset=${preset}"
echo "[run_oracle_experiment.bash] runner=${runner##*/}"
echo "[run_oracle_experiment.bash] exp_name=${exp_name}"
echo "[run_oracle_experiment.bash] config_path=${config_path}"
echo "[run_oracle_experiment.bash] ckpt_path=${ckpt_path}"
echo "[run_oracle_experiment.bash] train_iters=${train_iters:-<not-used>}"
echo "[run_oracle_experiment.bash] num_environments=${num_environments}"

env \
      CONDA_ENV="${CONDA_ENV}" \
      MASTER_PORT="${master_port}" \
      EXP_NAME="${exp_name}" \
      CONFIG_PATH="${config_path}" \
      CKPT_PATH="${ckpt_path}" \
      NUM_ENVIRONMENTS="${num_environments}" \
      GPU_NUMBERS="${GPU_NUMBERS}" \
      SIMULATOR_GPU_IDS="${SIMULATOR_GPU_IDS}" \
      TORCH_GPU_IDS="${TORCH_GPU_IDS}" \
      TORCH_GPU_ID="${TORCH_GPU_ID}" \
      ALLOW_SLIDING="${ALLOW_SLIDING}" \
      RESULTS_DIR="${RESULTS_DIR}" \
      TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
      CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER}" \
      VIDEO_DIR="${VIDEO_DIR}" \
      LOG_DIR="${LOG_DIR}" \
      IL_ITERS="${train_iters}" \
      ORACLE_ENABLE="${oracle_enable}" \
      ORACLE_ENABLE_IN_TRAIN="${oracle_enable_in_train}" \
      ORACLE_ENABLE_IN_EVAL="${oracle_enable_in_eval}" \
      ORACLE_APPLY_MODE="${oracle_apply_mode}" \
      ORACLE_SOFT_ALPHA="${oracle_soft_alpha}" \
      ORACLE_TARGET_GHOST_SCOPE="${oracle_target_ghost_scope}" \
      ORACLE_REFRESH_POLICY="${oracle_refresh_policy}" \
      ORACLE_CACHE_ENABLE="${oracle_cache_enable}" \
      ORACLE_TRACE_ENABLE="${oracle_trace_enable}" \
      ORACLE_SCOPE_TRACE_ENABLE="${oracle_scope_trace_enable}" \
      ORACLE_SCOPE_SUMMARY_ENABLE="${oracle_scope_summary_enable}" \
      ORACLE_STRICT_SCOPE="${oracle_strict_scope}" \
      ORACLE_FT_ENABLE="${oracle_ft_enable}" \
      ORACLE_FT_GAIN_INIT="${oracle_ft_gain_init}" \
      ORACLE_FT_MLP_LR="${oracle_ft_mlp_lr}" \
      ORACLE_FT_GRAPH_LR="${oracle_ft_graph_lr}" \
      ORACLE_FT_INPUT_PROJ_LR="${oracle_ft_input_proj_lr}" \
      ORACLE_FT_UNFREEZE_GLOBAL_ENCODER="${oracle_ft_unfreeze_global_encoder}" \
      ORACLE_FT_UNFREEZE_INPUT_PROJ="${oracle_ft_unfreeze_input_proj}" \
      bash "${runner}"
