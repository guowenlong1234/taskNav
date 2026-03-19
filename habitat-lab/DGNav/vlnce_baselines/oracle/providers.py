import numpy as np
from abc import abstractmethod,ABC
import torch
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
import time
from vlnce_baselines.common.ops import pad_tensors_wgrad
from torch.nn.utils.rnn import pad_sequence
from vlnce_baselines.common.utils import extract_instruction_tokens
from .types import OracleQuerySpec,OracleFeatureResult,TrajectoryObservationBufferItem
class OracleProvider(ABC):
    @abstractmethod
    def query(self,spec:OracleQuerySpec) ->OracleFeatureResult:
        '''
        输入OracleQuerySpec
        输出OracleFeatureResult
        异常：仅在 strict 模式下抛 RuntimeError外层 manager 负责捕获并写 trace
        '''

        raise NotImplementedError

class SimulatorPeekOracleProvider(OracleProvider):
    def __init__(
            self,
            envs,
            policy,
            waypoint_predictor,
            obs_transforms,
            device:torch.device,
            INSTRUCTION_SENSOR_UUID,
            task_type,
            instr_max_len,
                 ):
        self.envs = envs
        self.obs_transforms = obs_transforms
        self.device = device
        self.policy = policy
        self.INSTRUCTION_SENSOR_UUID = INSTRUCTION_SENSOR_UUID  #self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        self.task_type = task_type  
        self.instr_max_len = instr_max_len  #self.config.IL.max_text_len
        self.waypoint_predictor = waypoint_predictor
        
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)
        self.waypoint_predictor.to(self.device)

    def query(self,spec):
        t0 = time.perf_counter()
        policy_training = self.policy.training
        net_training = self.policy.net.training if hasattr(self.policy, "net") else None
        try:
            if spec.pipeline != "future_node_avg_pano":
                raise NotImplementedError(
                    f"Unsupported oracle pipeline: {spec.pipeline}"
                )
            env_index = spec.active_env_index
            query_pos = spec.query_pos
            query_heading_rad = spec.query_heading_rad

            obs = self.envs.call_at(
                env_index,
                "get_oracle_pano_obs_at",
                {
                    "position": query_pos,
                    "heading_rad": query_heading_rad,
                    "strict": True,
                },
            )
            #把观测中的指令字段进行处理，过长的截断，不足的补足pad，是指令长度与维度统一。
                    #设置最长步长
            # instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200

            # #设置不同的pad_id
            instr_pad_id = 1 if self.task_type == 'rxr' else 0

            obs = extract_instruction_tokens([obs], self.INSTRUCTION_SENSOR_UUID,
                                                    max_length=self.instr_max_len, pad_id=instr_pad_id)[0]
            
            batch = batch_obs([obs], self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            self.waypoint_predictor.eval()
            self.policy.eval()
            if hasattr(self.policy, "net"):
                self.policy.net.eval()
            with torch.no_grad():
                wp_outputs = self.policy.net(
                        mode = "waypoint",
                        waypoint_predictor = self.waypoint_predictor,
                        observations = batch,
                        #config.IL.waypoint_aug是否进行采样增强，训练的时候按照概率再nms周围选出一定的点
                        in_train = False,
                    )
                
                vp_inputs = self._vp_feature_variable(wp_outputs)
                #将这里面的都pad到相同长度，组织成batch，转换成tensor
                #             return {
                #     'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
                #     'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
                # }
                #向字典里新增或者覆盖一个键值对
                vp_inputs.update({
                    'mode': 'panorama',
                })

                #最终返回的是经过上下文融合之后的全景编码，包括角度、位置、深度、rgb等信息，形状为[B, L, 768]。还有一个mask
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)

                #把一整圈全景视角 token，压缩成“当前节点的单个全景摘要表示”。[B, L, H] -> [B, H],将12个视角特征进行融合
                avg_pano = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                            torch.sum(pano_masks, 1, keepdim=True)

                embed = avg_pano[0].detach().clone()

        
            return OracleFeatureResult(
                ghost_vp_id = spec.ghost_vp_id,
                ok = True,
                embed=embed,
                reason = None,
                embed_dtype = {
                                    torch.float16: "fp16",
                                    torch.float32: "fp32",
                                    torch.float64: "fp64",
                                }.get(embed.dtype, str(embed.dtype)),
                embed_norm = float(embed.norm().item()),
                used_pos = tuple(spec.query_pos),
                used_heading_rad = float(spec.query_heading_rad),
                cache_hit = False,
                cache_key = None,
                latency_ms = (time.perf_counter() - t0) * 1000.0
            )
        except Exception as e:
            raise RuntimeError(
                f"query过程报错,episode_id = {spec.episode_id}/ "
                f"ghost_vp_id = {spec.ghost_vp_id}/ "
                f"active_env_index = {spec.active_env_index}/ "
                f"original_env_index = {spec.original_env_index}"
            ) from e
        finally:
            if policy_training:
                self.policy.train()
            else:
                self.policy.eval()
            if hasattr(self.policy, "net") and net_training is not None:
                if net_training:
                    self.policy.net.train()
                else:
                    self.policy.net.eval()
            self.waypoint_predictor.eval()


    def _vp_feature_variable(self, obs):
        # obs = {
        #     'cand_rgb': cand_rgb,               # [2048]，对应路点的视觉特征向量
        #     'cand_depth': cand_depth,           # [128]，对应路点的深度特征向量
        #     'cand_angle_fts': cand_angle_fts,   # [4]，对应路点的角度特征向量
        #     'cand_img_idxes': cand_img_idxes,   # [1]，对应路点的视觉图片索引
        #     'cand_angles': cand_angles,         # [1]，对应路点的逆时针角度（弧度值）
        #     'cand_distances': cand_distances,   # [1]，对应路点的真实距离（m）

        #     'pano_rgb': pano_rgb,               # B x 12 x 512，全景照片的特征向量
        #     'pano_depth': pano_depth,           # B x 12 x 128，全景照片的维度向量
        #     'pano_angle_fts': pano_angle_fts,   # 12 x 4，全景照片每个角度特征
        #     'pano_img_idxes': pano_img_idxes,   # 12 ，0-11的标号，照片索引数组。
        # }
        # 输出一组相对位置，极坐标表示形式
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        batch_size = len(obs["cand_img_idxes"])
        for i in range(batch_size): #对于每个环境循环
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes]) #对布尔数组取反
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])

            #nav_types 1 表示 candidate view，0 表示 non-candidate view
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            #合成一个完整的视角张量，前K个是有候选路点的方向，后面的是非候选的
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))

            batch_nav_types.append(torch.LongTensor(nav_types))     #把当前环境的 nav_types 从 Python list 变成 LongTensor
            batch_view_lens.append(len(nav_types))                  #记录当前的视角数量
        # collate
        #把一个由不同长度 tensor 组成的 list，padding 到同样长度，再 stack 成一个 batch tensor

        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).to(self.device)
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).to(self.device)
        batch_view_lens = torch.LongTensor(batch_view_lens).to(self.device)

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
