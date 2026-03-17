# DGNav 3.3 Clean Train Runbook

本工作树固定在 `2a3ba5b`，用于干净的 Habitat 3.3 预训练与微调主线。

## 1. Worktree

- 路径：`/home/gwl/project/DGNav_new_clean33_train_main`
- 分支：`clean33_train_main`
- Python：`/home/gwl/miniconda3/envs/py3-9/bin/python`

## 2. 本地资产挂载

先在 `habitat-lab/DGNav` 下执行：

```bash
/home/gwl/miniconda3/envs/py3-9/bin/python tools/setup_clean33_local_assets.py
```

该脚本会在本工作树中挂载以下只读资产：

- `data`
- `pretrained`
- `pretrain_src/datasets`
- `pretrain_src/img_features`
- `vlnce_baselines/models/train`
- `../habitat-baselines/habitat_baselines/il/data`

## 3. 环境校验

```bash
cd /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav
/home/gwl/miniconda3/envs/py3-9/bin/python tools/check_clean33_env.py
```

要求：

- `habitat=0.3.3`
- `habitat_baselines=0.3.3`
- `habitat_sim=0.3.3`
- `habitat` 与 `habitat_baselines` 均来自 clean worktree

## 4. Pretrain Smoke

```bash
cd /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav
/home/gwl/miniconda3/envs/py3-9/bin/python pretrain_src/pretrain_src/train_r2r.py \
  --world_size 1 \
  --vlnbert cmt \
  --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
  --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
  --output_dir /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/pretrained/r2r_ce/clean33_smoke \
  --num_train_steps 1000 \
  --valid_steps 1000 \
  --log_steps 200
```

Smoke 判据：

- dataloader 正常启动
- seen/unseen validation 正常执行
- 输出目录下生成 `ckpts/model_step_1000.pt`

## 5. Full Pretrain

```bash
cd /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav
/home/gwl/miniconda3/envs/py3-9/bin/python pretrain_src/pretrain_src/train_r2r.py \
  --world_size 1 \
  --vlnbert cmt \
  --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
  --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
  --output_dir /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/pretrained/r2r_ce/clean33_full
```

## 6. Zero-Shot Pretrain Eval

主排序键 `success`，次排序键 `spl`：

```bash
cd /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav
/home/gwl/miniconda3/envs/py3-9/bin/python batch_eval_checkpoints.py \
  --ckpt-dir /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/pretrained/r2r_ce/clean33_full/ckpts \
  --dataset r2r \
  --split val_unseen \
  --best-metric success \
  --best-mode max \
  --parallel-jobs 1 \
  --num-envs 6 \
  --run-name clean33_pretrain_zeroshot
```

然后执行：

```bash
/home/gwl/miniconda3/envs/py3-9/bin/python tools/select_batch_eval_checkpoint.py \
  --results-json /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/batch_eval_results/r2r_clean33_pretrain_zeroshot/results.json \
  --primary-metric success \
  --secondary-metric spl \
  --mode max
```

输出中的 `selected.checkpoint` 就是唯一 finetune 起点。

## 7. Finetune Smoke

将 `<SELECTED_PRETRAIN_CKPT>` 替换为上一步选中的 `model_step_*.pt`：

```bash
cd /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav
/home/gwl/miniconda3/envs/py3-9/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port 24601 \
  run.py \
  --exp_name clean33_smoke_ft \
  --run-type train \
  --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS [0] \
  TORCH_GPU_IDS [0] \
  GPU_NUMBERS 1 \
  NUM_ENVIRONMENTS 6 \
  IL.iters 200 \
  IL.log_every 200 \
  IL.load_from_ckpt False \
  IL.is_requeue False \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  MODEL.pretrained_path <SELECTED_PRETRAIN_CKPT> \
  RESULTS_DIR /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/eval_results/clean33_smoke_ft/ \
  CHECKPOINT_FOLDER /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/clean33_smoke_ft/ \
  TENSORBOARD_DIR /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/tensorboard_dirs/clean33_smoke_ft/
```

Smoke 判据：

- loss 正常打印
- 生成 `ckpt.iter200.pth`

## 8. Full Finetune

```bash
cd /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav
/home/gwl/miniconda3/envs/py3-9/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port 24611 \
  run.py \
  --exp_name clean33_full_ft \
  --run-type train \
  --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS [0] \
  TORCH_GPU_IDS [0] \
  GPU_NUMBERS 1 \
  NUM_ENVIRONMENTS 6 \
  IL.load_from_ckpt False \
  IL.is_requeue False \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  MODEL.pretrained_path <SELECTED_PRETRAIN_CKPT> \
  RESULTS_DIR /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/eval_results/clean33_full_ft/ \
  CHECKPOINT_FOLDER /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/clean33_full_ft/ \
  TENSORBOARD_DIR /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/tensorboard_dirs/clean33_full_ft/
```

## 9. Full Batch Eval

主结果按 `spl` 选，`success` 同时记录：

```bash
cd /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav
/home/gwl/miniconda3/envs/py3-9/bin/python batch_eval_checkpoints.py \
  --ckpt-dir /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/data/logs/checkpoints/clean33_full_ft \
  --dataset r2r \
  --split val_unseen \
  --best-metric spl \
  --best-mode max \
  --parallel-jobs 1 \
  --num-envs 6 \
  --run-name clean33_full_ft
```

然后执行：

```bash
/home/gwl/miniconda3/envs/py3-9/bin/python tools/select_batch_eval_checkpoint.py \
  --results-json /home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/batch_eval_results/r2r_clean33_full_ft/results.json \
  --primary-metric spl \
  --secondary-metric success \
  --reference-metric success \
  --mode max
```

验收目标：

- `SR >= 0.50`
- `SPL >= 0.42`
