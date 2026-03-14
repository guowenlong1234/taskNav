export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# Default to Habitat-Lab/Baselines from current DGNav_new workspace.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dgnav_dir="$(cd "${script_dir}/.." && pwd)"
habitat_repo_root="$(cd "${dgnav_dir}/.." && pwd)"
export PYTHONPATH="${habitat_repo_root}/habitat-lab:${habitat_repo_root}/habitat-baselines:${PYTHONPATH}"
echo "[main.bash] Using Habitat-Lab/Baselines from ${habitat_repo_root}"

dist_launch_module="torch.distributed.launch"
if python -c "import importlib.util as u; raise SystemExit(0 if u.find_spec('torch.distributed.run') else 1)"; then
      dist_launch_module="torch.distributed.run"
fi

flag1="--exp_name release_r2r_dino_best_gacc
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 10
      IL.iters 20000
      IL.lr 1e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 3000
      IL.load_from_ckpt False
      IL.is_requeue False
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path /home/gwl/project/DGNav_new/habitat-lab/DGNav/pretrained/r2r_ce/mlm.sap_habitat_depth_dinov2s/ckpts/model_step_92500.pt
      "

flag2=" --exp_name release_r2r_dino
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR /home/gwl/project/DGNav_new/habitat-lab/DGNav/data/logs/checkpoints/release_r2r_dino
      IL.is_requeue False
      IL.back_algo control
      "

flag3="--exp_name release_r2r
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/DGNav/ckpt.iter15200.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m ${dist_launch_module} --nproc_per_node=1 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      python -m ${dist_launch_module} --nproc_per_node=1 --master_port $2 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python -m ${dist_launch_module} --nproc_per_node=1 --master_port $2 run.py $flag3
      ;;
esac
