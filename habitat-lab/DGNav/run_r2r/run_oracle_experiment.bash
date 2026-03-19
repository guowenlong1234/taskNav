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
  full_oracle_train
  full_oracle_eval_on_fixed500
  streaming_train_oracle_smoke
  streaming_eval_oracle_smoke
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
train_env_refill_policy="${TRAIN_ENV_REFILL_POLICY-}"
eval_env_refill_policy="${EVAL_ENV_REFILL_POLICY-}"

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
pretrained_path=""
train_iters=""
num_environments=""
il_load_from_ckpt=""
il_lr=""
il_waypoint_aug=""
il_back_algo=""
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
oracle_ft_train_scope=""
oracle_ft_unfreeze_global_encoder=""
oracle_ft_unfreeze_input_proj=""

case "${preset}" in
      streaming_train_oracle_smoke)
            # streaming + Oracle on 最小训练冒烟:
            # - 使用 stable-slot 语义
            # - 关闭 Oracle-FT adapter，聚焦 slot 绑定正确性
            runner="${train_entry}"
            master_port="4783"
            exp_name="streaming_train_oracle_smoke"
            config_path="run_r2r/iter_train.yaml,run_r2r/train_oracle_ft_base.yaml,run_r2r/oracle_ft_disabled.yaml,run_r2r/train_streaming_refill_oracle_smoke.yaml"
            ckpt_path="${release_ckpt_dir}/ckpt.iter18600.pth"
            train_iters="20"
            num_environments="2"
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
            oracle_ft_enable="False"
            ;;
      streaming_eval_oracle_smoke)
            # streaming + Oracle on 最小评测冒烟:
            # - 使用 stable-slot 语义
            # - 关闭 Oracle-FT adapter，聚焦 slot / trace 正确性
            runner="${eval_entry}"
            master_port="4784"
            exp_name="streaming_eval_oracle_smoke"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/oracle_ft_disabled.yaml,run_r2r/eval_fixed500.yaml,run_r2r/eval_streaming_refill_oracle_smoke.yaml"
            ckpt_path="${release_ckpt_dir}/ckpt.iter18600.pth"
            num_environments="2"
            oracle_enable="True"
            oracle_enable_in_train="False"
            oracle_enable_in_eval="True"
            oracle_apply_mode="soft"
            oracle_soft_alpha="0.25"
            oracle_target_ghost_scope="all"
            oracle_refresh_policy="on_change"
            oracle_cache_enable="True"
            oracle_trace_enable="True"
            oracle_scope_trace_enable="True"
            oracle_scope_summary_enable="True"
            oracle_ft_enable="False"
            ;;
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
      full_oracle_train)
            # 全量 Oracle 训练:
            # - 从 release_r2r_dino_best_nav 当年使用的同一预训练基座起跑
            # - 训练/评测 Oracle 打开, cache 打开
            # - baseline 风格 Oracle-FT：保留 baseline 可训练集合，再额外训练 oracle_adapter
            # - DINO backbone 继续冻结；让视觉 MLP、Oracle MLP 和下游网络一起适应 Oracle 特征
            # - 学习率与 baseline/B2 主 LR 保持一致
            # - full val_unseen Oracle 上限参考：oracle_success ~= 0.6585 (release_r2r_dino_best_nav/ckpt.iter21800)
            runner="${train_entry}"
            master_port="4781"
            exp_name="full_oracle_train"
            config_path="run_r2r/iter_train.yaml"
            ckpt_path=""
            pretrained_path="/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/pretrained/r2r_ce/mlm.sap_habitat_depth_dinov2_clean/ckpts/model_step_97500.pt"
            train_iters="25000"
            num_environments="6"
            il_load_from_ckpt="False"
            il_lr="1e-5"
            il_waypoint_aug="True"
            il_back_algo="teleport"
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
            oracle_ft_train_scope="baseline_plus_oracle_adapter"
            ;;
      full_oracle_eval_on_fixed500)
            # 全量 Oracle 训练产物验证:
            # - 评测口径对齐 B2_eval_on_fixed500
            # - fixed500 val_unseen
            # - Oracle 打开, cache 打开
            # - 与 full_oracle_train 一致，评测时保持 Oracle-FT 打开
            runner="${eval_entry}"
            master_port="4782"
            exp_name="full_oracle_eval_on_fixed500"
            config_path="run_r2r/iter_train.yaml,run_r2r/eval_oracle_ft_base.yaml,run_r2r/eval_fixed500.yaml"
            # 显式填写按 baseline_plus_oracle_adapter 新逻辑重新训练后产出的 checkpoint。
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
            oracle_ft_train_scope="baseline_plus_oracle_adapter"
            ;;
      *)
            echo "[run_oracle_experiment.bash] 未知预设: ${preset}" >&2
            usage >&2
            exit 1
            ;;
esac

if [[ "${il_load_from_ckpt:-True}" == "False" ]]; then
      require_existing_file "${pretrained_path}" "PRETRAINED_PATH"
else
      require_existing_file "${ckpt_path}" "CKPT_PATH"
fi

echo "[run_oracle_experiment.bash] preset=${preset}"
echo "[run_oracle_experiment.bash] runner=${runner##*/}"
echo "[run_oracle_experiment.bash] exp_name=${exp_name}"
echo "[run_oracle_experiment.bash] config_path=${config_path}"
echo "[run_oracle_experiment.bash] ckpt_path=${ckpt_path}"
echo "[run_oracle_experiment.bash] pretrained_path=${pretrained_path:-<not-used>}"
echo "[run_oracle_experiment.bash] train_iters=${train_iters:-<not-used>}"
echo "[run_oracle_experiment.bash] num_environments=${num_environments}"
echo "[run_oracle_experiment.bash] TRAIN_ENV_REFILL_POLICY=${train_env_refill_policy:-<yaml/default>}"
echo "[run_oracle_experiment.bash] EVAL_ENV_REFILL_POLICY=${eval_env_refill_policy:-<yaml/default>}"

env \
      CONDA_ENV="${CONDA_ENV}" \
      MASTER_PORT="${master_port}" \
      EXP_NAME="${exp_name}" \
      CONFIG_PATH="${config_path}" \
      CKPT_PATH="${ckpt_path}" \
      PRETRAINED_PATH="${pretrained_path}" \
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
      TRAIN_ENV_REFILL_POLICY="${train_env_refill_policy}" \
      EVAL_ENV_REFILL_POLICY="${eval_env_refill_policy}" \
      IL_LOAD_FROM_CKPT="${il_load_from_ckpt}" \
      IL_ITERS="${train_iters}" \
      IL_LR="${il_lr}" \
      IL_WAYPOINT_AUG="${il_waypoint_aug}" \
      IL_BACK_ALGO="${il_back_algo}" \
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
      ORACLE_FT_TRAIN_SCOPE="${oracle_ft_train_scope}" \
      ORACLE_FT_UNFREEZE_GLOBAL_ENCODER="${oracle_ft_unfreeze_global_encoder}" \
      ORACLE_FT_UNFREEZE_INPUT_PROJ="${oracle_ft_unfreeze_input_proj}" \
      bash "${runner}"
