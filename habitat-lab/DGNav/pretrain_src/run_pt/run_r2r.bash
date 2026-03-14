
NODE_RANK=0
NUM_GPUS=1  
outdir=pretrained/r2r_ce/mlm.sap_habitat_depth_dinov2s

python pretrain_src/pretrain_src/train_r2r.py \
    --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
    --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
    --output_dir $outdir



# NODE_RANK=0
# NUM_GPUS=2
# outdir=pretrained/r2r_ce/mlm.sap_habitat_depth

# # train
# # 分布式启动训练
# python -m torch.distributed.launch \

#     #每台机器起 2 个进程（对应 2 张 GPU）--node_rank $NODE_RANK：本机 rank，--master_port=$1：主进程端口，来自脚本的第一个参数（你调用脚本时传入的 2333 就是这里）。
#     --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
#     pretrain_src/pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
#     --vlnbert cmt \
#     --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
#     --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
#     --output_dir $outdir