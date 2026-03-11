from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.etp.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, angle_feature_torch, length2mask)
import math

@baseline_registry.register_policy
class PolicyViewSelectionETP(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            ETP(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class ETP(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the ETP model ...')
        self.vln_bert = get_vlnbert_models(config=model_config)
        # if model_config.task_type == 'r2r':
        #     self.rgb_projection = nn.Linear(2048, 768)
        # elif model_config.task_type == 'rxr':
        #     self.rgb_projection = nn.Linear(2048, 512)
        # self.rgb_projection = nn.Linear(2048, 768) # for vit 768 compability
        # if model_config.task_type == 'r2r':
        #     self.rgb_projection = nn.Linear(512, 768)
        # else:
        #     self.rgb_projection = None
        self.drop_env = nn.Dropout(p=0.4)

        # self.pos_encoder = nn.Sequential(
        #     nn.Linear(6, 768),
        #     nn.LayerNorm(768, eps=1e-12)
        # )
        # self.hist_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 768)
        # )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))

        # Init the RGB encoder
        # assert model_config.RGB_ENCODER.cnn_type in [
        #     "TorchVisionResNet152", "TorchVisionResNet50"
        # ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        # if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
        #     self.rgb_encoder = TorchVisionResNet50(
        #         observation_space,
        #         model_config.RGB_ENCODER.output_size,
        #         device,
        #         spatial_output=model_config.spatial_output,
        #     )

        #实例化一个CLIP视觉编码器
        self.rgb_encoder = CLIPEncoder(self.device)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # Counter-clockwise
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # Corresponding to counter-clockwise
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))   #预计算12个角度的方向角特征

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, mode=None, 
                txt_ids=None, txt_masks=None, txt_embeds=None, 
                waypoint_predictor=None, observations=None, in_train=True,
                rgb_fts=None, dep_fts=None, loc_fts=None, 
                nav_types=None, view_lens=None,
                gmap_vp_ids=None, gmap_step_ids=None,
                gmap_img_fts=None, gmap_pos_fts=None, 
                gmap_masks=None, gmap_visited_masks=None, gmap_pair_dists=None):

        if mode == 'language':
            #语言编码模块，不涉及到跨模态
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids, txt_masks,
            )
            return encoded_sentence

        elif mode == 'waypoint':
            #路点预测模块
            # batch_size = observations['instruction'].size(0)
            batch_size = observations['rgb'].shape[0]
            ''' encoding rgb/depth at all directions ----------------------------- '''

            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_IMGS = 12
            NUM_CLASSES = 12    # 12 distances at each sector

            #pytorch自带函数，按照某个张量的形状，创建一个相同形状的值全为0的新张量
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            # reverse the order of input images to clockwise
            a_count = 0 #是原始感测的顺序
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' in k:  # You might need to double check the keys order
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS  #是重新排列后的观测数据
                        depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1
            obs_view12 = {}

            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch

            #调用视觉特征提取器
            depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
            rgb_embedding = self.rgb_encoder(obs_view12)      # torch.Size([bs, 2048, 7, 7]) [12B, 512]

            ''' waypoint prediction ----------------------------- '''

            #经过transforme和线性分类头，给每个与选路点进行打分
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding, depth_embedding)

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 512, 1, 1)
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4)
            
            #这两段是在把 12 视角特征的顺序从“顺时针顺序”改回“逆时针顺序”，同时保留第 0 个视角不动。
            rgb_feats = torch.cat((
                rgb_embed_reshape[:,0:1,:], 
                torch.flip(rgb_embed_reshape[:,1:,:], [1]),
            ), dim=1)

            depth_feats = torch.cat((
                depth_embed_reshape[:,0:1,:], 
                torch.flip(depth_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            # way_feats = torch.cat((
            #     way_feats[:,0:1,:], 
            #     torch.flip(way_feats[:,1:,:], [1]),
            # ), dim=1)

            # from heatmap to points

            #[120, 12]->[1440]->softmax ->每个潜在 waypoint 位置的概率
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            # [1440]->[B, 120, 12],存储的是概率
            batch_x_norm = batch_x_norm.reshape(
                batch_size, NUM_ANGLES, NUM_CLASSES,
            )

            #在角度维前后各补一圈边界把最后一个角度 bin 复制到最前面，把第一个角度 bin 复制到最后面
            batch_x_norm_wrap = torch.cat((
                batch_x_norm[:,-1:,:], 
                batch_x_norm, 
                batch_x_norm[:,:1,:]), 
                dim=1)
            
            #在每张二维热图上找局部最大值，并抑制周围重复高响应点，从 dense heatmap 中筛出稀疏的候选 waypoint 峰值
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), 
                max_predictions=5,  #最多保留5个路点
                sigma=(7.0,5.0))    #抑制范围 7.0 对应角度维方向的抑制范围。5.0 对应距离维方向的抑制范围


            # predicted waypoints before sampling
            #数据维度不变，但是变成稀疏的，里面只有0和1,1就是候选路点。
            batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

            # candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
            # if isinstance(candidate_lengths, int):
            #     candidate_lengths = [candidate_lengths]
            # max_candidate = max(candidate_lengths)  # including stop
            # cand_mask = length2mask(candidate_lengths, device=self.device)

            #如果当前是训练模式并且打开了采样增强
            if in_train:
                # Waypoint augmentation
                # parts of heatmap for sampling (fix offset first)
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                # batch_way_log_prob = []
                for j in range(batch_size):

                    #针对每一个环境的数据进行操作
                    # angle indexes with candidates
                    #先找出 NMS 保留下来的候选峰值角度
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # clockwise image indexes (same as batch_x_norm)

                    #把 angle bin 映射到所属的 12 视角图像编号
                    img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                    img_idxes[img_idxes==12] = 0
                    # # candidate waypoint states
                    # way_feats_regional = way_feats[j][img_idxes]
                    # heatmap regions for sampling

                    #这里不是直接用 NMS 的峰值点，而是把对应视角区域里的更细粒度 heatmap 取出来。展开后就是一个局部候选集合
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)

                    #对局部区域做 softmax，变成采样分布
                    way_heats_probs = F.softmax(way_heats_regional, 1)

                    #在局部区域里随机采样一个具体位置
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()

                    #后面准备把采样结果还原成 angle_idx 和 distance_idx
                    sample_angle_idxes = []
                    sample_distance_idxes = []

                    #这一段是在把“局部区域里采样出来的编号 way_act”还原成真正的全局候选路点索引：
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(way_act//12+angle_pointer)
                        sample_distance_idxes.append(way_act%12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)
                    # batch_way_log_prob.append(
                    #     probs_c.log_prob(way_heats_act))
            else:
                # batch_way_log_prob = None
                None
            
            #视觉特征的池化，但是本身在CLIP的编码器架构下没有什么意义
            rgb_feats = self.space_pool_rgb(rgb_feats)

            #深度特征值的池化，4*4 -> 1
            depth_feats = self.space_pool_depth(depth_feats)

            # for cand
            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []

            #对于每一个环境
            for j in range(batch_size):
                if in_train:    #如果训练模式并且开启了数据增强
                    #取出对应的角度和距离
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                # for angle & distance

                #将离散的角度索引转化为弧度角，顺时针角度表示 angle_rad_c
                angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # Clockwise

                #逆时针角度表示 angle_rad_cc
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # Counter-clockwise

                #把顺时针弧度角 angle_rad_c 编成角度特征向量
                cand_angle_fts.append( angle_feature_torch(angle_rad_c) )

                #把逆时针弧度角保存下来，作为候选 waypoint 的真实几何
                cand_angles.append(angle_rad_cc.tolist())

                #把离散距离 bin 转成真实距离值
                cand_distances.append( ((distance_idxes + 1)*0.25).tolist() )
                # for img idxes
                #先把 angle bin 映射成视角编号
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # Counter-clockwise
                img_idxes[img_idxes==12] = 0
                cand_img_idxes.append(img_idxes)
                # for rgb & depth
                #再按这些视角编号抽取候选视觉特征
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])
            
            # for pano
            pano_rgb = rgb_feats                            # B x 12 x 2048，其实是512维度向量
            pano_depth = depth_feats                        # B x 12 x 128
            pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
            pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

            # cand_angle_fts clockwise
            # cand_angles counter-clockwise
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]，对应路点的视觉特征向量
                'cand_depth': cand_depth,           # [K x 128]，对应路点的深度特征向量
                'cand_angle_fts': cand_angle_fts,   # [K x 4]，对应路点的角度特征向量
                'cand_img_idxes': cand_img_idxes,   # [K]，对应路点的视觉图片索引
                'cand_angles': cand_angles,         # [K]，对应路点的逆时针角度（弧度值）
                'cand_distances': cand_distances,   # [K]，对应路点的真实距离（m）

                'pano_rgb': pano_rgb,               # B x 12 x 512，全景照片的特征向量
                'pano_depth': pano_depth,           # B x 12 x 128，全景照片的维度向量
                'pano_angle_fts': pano_angle_fts,   # 12 x 4，全景照片每个角度特征
                'pano_img_idxes': pano_img_idxes,   # 12 ，0-11的标号，照片索引数组。
            }
            
            return outputs

        elif mode == 'panorama':
            #传进来的都是全景图组织好的batch tensor
            #             return {
            #     'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            #     'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            # }
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts, dep_fts, loc_fts, nav_types, view_lens,
            )
            #最终返回的是经过上下文融合之后的全景编码，包括角度、位置、深度、rgb等信息，形状为[B, L, 768]。还有一个mask
            return outs

        elif mode == 'navigation':
                #             return {
        #     'gmap_vp_ids': batch_gmap_vp_ids, #图里有哪些点
        #     'gmap_step_ids': batch_gmap_step_ids,   #这些点什么时候来的
        #     'gmap_img_fts': batch_gmap_img_fts,     #这些点长什么样
        #     'gmap_pos_fts': batch_gmap_pos_fts,     #这些点相对我在哪
        #     'gmap_masks': batch_gmap_masks,         #哪些点有效
        #     'gmap_visited_masks': batch_gmap_visited_masks,     #哪些点已访问
        #     'gmap_pair_dists': gmap_pair_dists,     #点和点之间有多远
        #     'no_vp_left': batch_no_vp_left,         #还有没有可探索 ghost
        # }
            outs = self.vln_bert.forward_navigation(
                txt_embeds, txt_masks, 
                gmap_vp_ids, gmap_step_ids,
                gmap_img_fts, gmap_pos_fts, 
                gmap_masks, gmap_visited_masks, gmap_pair_dists,
            )
        #             outs = {
        #     'gmap_embeds': gmap_embeds, #经过全局图导航编码器更新后的图节点表示[B, L, H]
        #     'global_logits': global_logits, # 对图中每个可选节点的打分[B, L]
        # }
            return outs

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
