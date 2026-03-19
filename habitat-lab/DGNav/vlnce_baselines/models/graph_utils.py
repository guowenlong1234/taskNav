from collections import defaultdict
import numpy as np
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
import torch

MAX_DIST = 30
MAX_STEP = 10
# NOISE = 0.5

def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0, to_clock=False):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    # xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    # heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    heading = np.arcsin(-dx / xz_dist)  # [-pi/2, pi/2]
    # if b[1] < a[1]:
    #     heading = np.pi - heading
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    if to_clock:
        heading = 2 * np.pi - heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

def heading_from_quaternion(quat: np.array):
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    quat = quaternion_from_coeff(quat)
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)

def estimate_cand_pos(pos, ori, ang, dis):
    cand_num = len(ang)
    cand_pos = np.zeros([cand_num, 3])

    ang = np.array(ang)
    dis = np.array(dis)
    ang = (heading_from_quaternion(ori) + ang) % (2 * np.pi)
    cand_pos[:, 0] = pos[0] - dis * np.sin(ang)    # x
    cand_pos[:, 1] = pos[1]                        # y
    cand_pos[:, 2] = pos[2] - dis * np.cos(ang)    # z
    return cand_pos


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y and x !=k and y != k:
                    t_dis = self._dis[x][y] + self._dis[y][k]
                    if t_dis < self._dis[x][k]:
                        self._dis[x][k] = t_dis
                        self._dis[k][x] = t_dis
                        self._point[x][k] = y
                        self._point[k][x] = y

        for x in self._dis:
            for y in self._dis:
                if x != y:
                    t_dis = self._dis[x][k] + self._dis[k][y]
                    if t_dis < self._dis[x][y]:
                        self._dis[x][y] = t_dis
                        self._dis[y][x] = t_dis
                        self._point[x][y] = k
                        self._point[y][x] = k

        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self, has_real_pos, loc_noise, merge_ghost, ghost_aug, oracle_cfg = None):

        self.oracle_cfg = oracle_cfg
        self.graph_nx = nx.Graph()

        self.node_pos = {}          # 视点的位置
        self.node_embeds = {}       # 视点的特征值
        self.node_stepId = {}

        self.ghost_cnt = 0          # 节点计数变量
        self.ghost_pos = {}         #ghost节点位置
        self.ghost_mean_pos = {}    #ghost节点平均位置
        self.ghost_embeds = {}      #ghost特征编码
        self.ghost_fronts = {}      #从哪个节点能看到这个ghost节点
        self.ghost_real_pos = {}    #ghost节点的真实位置
        self.has_real_pos = has_real_pos        #是否拥有真实的位置
        self.merge_ghost = merge_ghost          #新预测出来的节点是否要和原本节点进行合并
        self.ghost_aug = ghost_aug  # 0 ~ 1, noise level，干扰水平
        self.loc_noise = loc_noise              #是否开启干扰

        self.shortest_path = None   #最短路径
        self.shortest_dist = None   #最短距离
        
        self.node_stop_scores = {}  # 当前节点的停止得分

        #新增：oracle ghost embedding
        self.ghost_oracle_embeds = {}
        #oracle 元信息,可用于刷新判定与trace
        self.ghost_oracle_meta = {}
        self.graph_step = 0
        self.last_added_ghost_ids = []
        self.step_added_ghost_ids = {}
        self.ghost_parent_real_node = {}
        self.oracle_last_scope_ids = []
        self.oracle_last_written_ids = []
        self.oracle_last_skipped_ids = []
        self.oracle_write_step = {}


    def has_oracle_embed(self, vp_id:str)->bool:
        '''
        返回:vp_id 是否存在 oracle embedding。
        异常：无
        '''
        return vp_id in self.ghost_oracle_embeds
        

    def get_oracle_embed(self, vp_id:str):
        '''
        返回 oracle embedding 或 None。
        异常KeyError 不抛出统一返回None。
        '''
        return self.ghost_oracle_embeds.get(vp_id, None)

    def get_base_ghost_embed(self, vp_id: str):
        return self.ghost_embeds[vp_id][0] / self.ghost_embeds[vp_id][1]

    def get_effective_ghost_embed(
        self,
        vp_id: str,
        use_oracle: bool = True,
        allowed_oracle_ghost_ids=None,
    ):
        e_base = self.get_base_ghost_embed(vp_id)
        if (
            use_oracle
            and
            self.oracle_cfg is not None
            and self.oracle_cfg.enable
            and vp_id in self.ghost_oracle_embeds
        ):
            if (
                allowed_oracle_ghost_ids is not None
                and vp_id not in allowed_oracle_ghost_ids
            ):
                return e_base
            apply_mode = str(
                getattr(self.oracle_cfg, "apply_mode", "hard")
            ).lower()
            if apply_mode == "soft":
                alpha = float(getattr(self.oracle_cfg, "soft_alpha", 1.0))
                alpha = max(0.0, min(1.0, alpha))
                e_oracle = self.get_oracle_embed(vp_id)
                return (1.0 - alpha) * e_base + alpha * e_oracle
            return self.get_oracle_embed(vp_id)
        return e_base


    def set_oracle_embed(self, vp_id: str, embed, meta=None, overwrite: bool = True):
        if vp_id not in self.ghost_mean_pos:    #验证节点是不是ghost节点
            raise ValueError(f"{vp_id} is not a valid ghost node")

        if not isinstance(embed, torch.Tensor): #验证embed数据类型
            raise ValueError("embed must be a torch.Tensor")

        if embed.ndim != 1: #验证embed的维度
            raise ValueError(f"embed must be 1D, got shape={tuple(embed.shape)}")

        expected_shape = None
        if self.node_embeds:    #取出自身node节点的维度
            expected_shape = next(iter(self.node_embeds.values())).shape
        elif self.ghost_embeds: #取出自身的ghost节点的维度
            expected_shape = next(iter(self.ghost_embeds.values()))[0].shape

        if expected_shape is not None and tuple(embed.shape) != tuple(expected_shape):
            raise ValueError(
                f"embed shape mismatch, got {tuple(embed.shape)}, expected {tuple(expected_shape)}"
            )

        if (not overwrite) and self.has_oracle_embed(vp_id):
            #如果你要求“不要覆盖旧值”，并且这个 ghost 已经有 oracle embedding，那就直接退出，不再写新值。
            return

        self.ghost_oracle_embeds[vp_id] = embed.detach().clone()
        self.ghost_oracle_meta[vp_id] = {} if meta is None else dict(meta)


    def pop_oracle_embed(self,vp_id:str):
        '''
        行为：若存在则删除 oracle embed 与 meta。
        异常：无
        '''
        self.ghost_oracle_embeds.pop(vp_id, None)
        self.ghost_oracle_meta.pop(vp_id, None)

    def get_last_added_ghost_ids(self):
        return list(self.last_added_ghost_ids)

    def get_local_frontier_ghost_ids(self, current_real_vp: str):
        local_frontier_ids = []
        for ghost_id, front_vps in self.ghost_fronts.items():
            if ghost_id not in self.ghost_pos or current_real_vp not in front_vps:
                continue
            _, nearest_front_vp = self.front_to_ghost_dist(ghost_id)
            if nearest_front_vp == current_real_vp:
                local_frontier_ids.append(ghost_id)
        return local_frontier_ids

    def get_all_alive_ghost_ids(self):
        return list(self.ghost_pos.keys())

    def get_node_embed_components(self, vp_id: str):
        if not vp_id.startswith('g'):
            base = self.node_embeds[vp_id]
            return {
                "base": base,
                "oracle_raw": None,
                "has_oracle": False,
                "is_ghost": False,
            }
        base = self.get_base_ghost_embed(vp_id)
        oracle_raw = self.get_oracle_embed(vp_id)
        return {
            "base": base,
            "oracle_raw": oracle_raw,
            "has_oracle": oracle_raw is not None,
            "is_ghost": True,
        }

    def apply_oracle_embeds(
        self,
        ghost_embeds: dict,
        allowed_ghost_ids: list,
        step_id: int,
        strict_scope: bool = True,
    ):
        allow = set(allowed_ghost_ids)
        written, skipped = [], []
        for ghost_id, payload in ghost_embeds.items():
            if strict_scope and ghost_id not in allow:
                skipped.append(ghost_id)
                continue
            if ghost_id not in self.ghost_pos:
                skipped.append(ghost_id)
                continue

            embed = payload
            meta = None
            if isinstance(payload, dict):
                embed = payload.get("embed")
                meta = payload.get("meta")
            elif hasattr(payload, "embed"):
                embed = getattr(payload, "embed")
                meta = getattr(payload, "meta", None)

            if embed is None:
                skipped.append(ghost_id)
                continue

            effective_meta = {} if meta is None else dict(meta)
            effective_meta.setdefault("stepk", step_id)
            self.set_oracle_embed(ghost_id, embed, meta=effective_meta)
            self.oracle_write_step[ghost_id] = step_id
            written.append(ghost_id)

        self.oracle_last_scope_ids = list(allowed_ghost_ids)
        self.oracle_last_written_ids = written
        self.oracle_last_skipped_ids = skipped
        return written, skipped


    def _localize(self, qpos, kpos_dict, ignore_height=False, loc_noise=None):
        """
        Args:
            loc_noise: If provided, use this value; otherwise use self.loc_noise
        """
        if loc_noise is None:
            loc_noise = self.loc_noise
        min_dis = 10000
        min_vp = None
        for kvp, kpos in kpos_dict.items():
            if ignore_height:
                dis = ((qpos[[0,2]] - kpos[[0,2]])**2).sum()**0.5
            else:
                dis = ((qpos - kpos)**2).sum()**0.5
            if dis < min_dis:
                min_dis = dis
                min_vp = kvp
        min_vp = None if min_dis > loc_noise else min_vp
        return min_vp
    
    def identify_node(self, cur_pos, cur_ori, cand_ang, cand_dis):
        # assume no repeated node
        # since action is restricted to ghosts
        #“当前节点”和“当前时刻预测出的候选节点”生成图里的临时 ID，并把候选点从相对极坐标转换成估计的三维位置。
        #给当前真实所在位置生成一个新的 node id
        cur_vp = str(len(self.node_pos))
        #给当前节点发散出来的每个候选 waypoint 生成一个候选 id。
        cand_vp = [f'{cur_vp}_{str(i)}' for i in range(len(cand_ang))]
        #把候选 waypoint 的相对几何信息变成估计的绝对位置
        cand_pos = [p for p in estimate_cand_pos(cur_pos, cur_ori, cand_ang, cand_dis)]

        #cur_vp当前节点的 id
        # cand_vp当前时刻所有候选点的 id 列表
        # cand_pos当前时刻所有候选点的估计位置列表

        return cur_vp, cand_vp, cand_pos

    def delete_ghost(self, vp):
        self.ghost_pos.pop(vp)  
        self.ghost_mean_pos.pop(vp)
        self.ghost_embeds.pop(vp)
        self.ghost_fronts.pop(vp)

        self.pop_oracle_embed(vp)   #调用pop_oracle_embed函数，更安全的删掉ghost_oracle_embed和ghost_oracle_meta

        if self.has_real_pos:
            self.ghost_real_pos.pop(vp)
        self.ghost_parent_real_node.pop(vp, None)
        self.oracle_write_step.pop(vp, None)
        self.last_added_ghost_ids = [
            ghost_id for ghost_id in self.last_added_ghost_ids if ghost_id != vp
        ]
        for step_id, ghost_ids in list(self.step_added_ghost_ids.items()):
            self.step_added_ghost_ids[step_id] = [
                ghost_id for ghost_id in ghost_ids if ghost_id != vp
            ]

    def get_ghost_members(self, ghost_vp_id):
        if ghost_vp_id not in self.ghost_mean_pos:
            raise ValueError(f"{ghost_vp_id} is not a valid ghost node")
        if not self.has_real_pos:
            raise ValueError("ghost member binding requires has_real_pos=True")
        if ghost_vp_id not in self.ghost_real_pos:
            raise ValueError(f"{ghost_vp_id} has no ghost_real_pos")
        if ghost_vp_id not in self.ghost_fronts:
            raise ValueError(f"{ghost_vp_id} has no ghost_fronts")
        if ghost_vp_id not in self.ghost_pos:
            raise ValueError(f"{ghost_vp_id} has no ghost_pos")

        real_pos_list = self.ghost_real_pos[ghost_vp_id]
        front_vp_list = self.ghost_fronts[ghost_vp_id]
        cand_pos_list = self.ghost_pos[ghost_vp_id]

        if not (
            len(real_pos_list) == len(front_vp_list) == len(cand_pos_list)
        ):
            raise ValueError(
                f"ghost member binding mismatch for {ghost_vp_id}: "
                f"len(real_pos)={len(real_pos_list)} "
                f"len(fronts)={len(front_vp_list)} "
                f"len(cand_pos)={len(cand_pos_list)}"
            )

        members = []
        for idx, (real_pos, front_vp_id, cand_pos) in enumerate(
            zip(real_pos_list, front_vp_list, cand_pos_list)
        ):
            members.append(
                {
                    "index": idx,
                    "real_pos": tuple(np.asarray(real_pos).tolist()),
                    "front_vp_id": front_vp_id,
                    "cand_pos": tuple(np.asarray(cand_pos).tolist()),
                }
            )
        return members

    def update_graph(self, prev_vp, step_id,
                           cur_vp, cur_pos, cur_embeds,
                           cand_vp, cand_pos, cand_embeds, 
                           cand_real_pos, loc_noise=None):
        """
        Args:
            loc_noise: Dynamic loc_noise value, if provided use this value; otherwise use self.loc_noise
        """
        # prev_vp[i]上一个真实节点 id
        # stepk + 1当前是第几步
        # cur_vp[i]当前真实节点 id
        # cur_pos[i]当前真实位置
        # cur_embeds当前节点 embedding
        # cand_vp[i]当前所有候选点 id
        # cand_pos[i]当前所有候选点估计位置
        # cand_embeds当前所有候选点对应的视觉特征
        # cand_real_pos[i]候选点真实位置，训练/可视化时可能用来监督
        # loc_noise=loc_noise_to_use当前步用于节点/ghost 合并判断的容差阈值
        
        self.graph_step = step_id
        new_ghost_ids = []

        # 1. connect prev_vp
        self.graph_nx.add_node(cur_vp)

        #如果上一步有节点
        if prev_vp is not None:
            prev_pos = self.node_pos[prev_vp]   #找到上一步的真实位置
            dis = calc_position_distance(prev_pos, cur_pos) #计算到上一步的真实距离
            self.graph_nx.add_edge(prev_vp, cur_vp, weight=dis) #添加一条边，从上一个节点到下一个节点，边的值是距离

        # 2. update node & ghost info
        self.node_pos[cur_vp] = cur_pos #记录节点的空间位置
        self.node_embeds[cur_vp] = cur_embeds# 记录节点的视觉全景特征表示
        self.node_stepId[cur_vp] = step_id  #记录节点的步数

        # If dynamic loc_noise is provided, update self.loc_noise (for subsequent _localize calls)
        if loc_noise is not None:
            self.loc_noise = loc_noise

        for i, (cvp, cpos, cembeds) in enumerate(zip(cand_vp, cand_pos, cand_embeds)):
            #这里是在遍历当前步所有候选点，并把三类信息同步取出来，当前候选点的候选 id，当前候选点估计出来的空间位置，当前候选点对应的视觉特征

            #判断是不是一个新节点，标准与阈值就是loc_noise
            localized_nvp = self._localize(cpos, self.node_pos, loc_noise=loc_noise)    #候选点 cpos 在当前图里最近且足够近的那个 node 的 id，如果没有任何已有 node 足够近，返回：None

            # cand overlap with node, connect cur_vp with localized_nvp
            if localized_nvp is not None :
                dis = calc_position_distance(cur_pos, self.node_pos[localized_nvp])
                self.graph_nx.add_edge(cur_vp, localized_nvp, weight=dis)
            # cand not overlap with node, create/update ghost
            else:
                if self.merge_ghost:    #允许把相近的 ghost 合并
                    localized_gvp = self._localize(cpos, self.ghost_mean_pos, loc_noise=loc_noise)  #cpos 去和已有所有 ghost 的平均位置 self.ghost_mean_pos 比较。
                    # create ghost
                    if localized_gvp is None:   #如果没有匹配到已有 ghost，就新建 ghost
                        gvp = f'g{str(self.ghost_cnt)}' #生成一个新的 ghost id，比如 g0, g1, g2
                        self.ghost_cnt += 1     #
                        self.ghost_pos[gvp] = [cpos]    #记录它的观测位置列表
                        self.ghost_mean_pos[gvp] = cpos #初始化它的平均位置 ghost_mean_pos
                        self.ghost_embeds[gvp] = [cembeds, 1]   #记录它的 embedding 和计数 ghost_embeds = [特征和, 数量]，当前候选方向对应的上下文化 panorama token 特征
                        self.ghost_fronts[gvp] = [cur_vp]       #记录这个 ghost 是从哪个当前节点 cur_vp 看到的 ghost_fronts
                        self.ghost_parent_real_node[gvp] = cur_vp
                        new_ghost_ids.append(gvp)
                        if self.has_real_pos:   #如果保存真实位置，还把真实位置记下来
                            self.ghost_real_pos[gvp] = [cand_real_pos[i]]
                    # update ghost
                    else:   #如果不是一个新的ghost，就合并进去，并且更新合并之后的各种信息
                        gvp = localized_gvp
                        self.ghost_pos[gvp].append(cpos)
                        self.ghost_mean_pos[gvp] = np.mean(self.ghost_pos[gvp], axis=0)
                        self.ghost_embeds[gvp][0] = self.ghost_embeds[gvp][0] + cembeds
                        self.ghost_embeds[gvp][1] += 1
                        self.ghost_fronts[gvp].append(cur_vp)
                        if self.has_real_pos:
                            self.ghost_real_pos[gvp].append(cand_real_pos[i])
                else:
                    gvp = f'g{str(self.ghost_cnt)}'
                    self.ghost_cnt += 1
                    self.ghost_pos[gvp] = [cpos]
                    self.ghost_mean_pos[gvp] = cpos
                    self.ghost_embeds[gvp] = [cembeds, 1]
                    self.ghost_fronts[gvp] = [cur_vp]
                    self.ghost_parent_real_node[gvp] = cur_vp
                    new_ghost_ids.append(gvp)
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp] = [cand_real_pos[i]]

        self.last_added_ghost_ids = list(new_ghost_ids)
        self.step_added_ghost_ids[step_id] = list(new_ghost_ids)
        
        self.ghost_aug_pos = deepcopy(self.ghost_mean_pos)  #ghost_mean_pos存的是每个 ghost 的“平均位置”#self.ghost_aug_pos前实际拿来参与后续图计算的 ghost 位置
        if self.ghost_aug != 0: #如过有扰动，就在平均位置上加上一些扰动，如果没有扰动，就直接使用当前的平均位置    
            for gvp, gpos in self.ghost_aug_pos.items():
                gpos_noise = np.random.normal(loc=(0,0,0), scale=(self.ghost_aug,0,self.ghost_aug), size=(3,))
                gpos_noise[gpos_noise < -self.ghost_aug] = -self.ghost_aug
                gpos_noise[gpos_noise >  self.ghost_aug] =  self.ghost_aug
                self.ghost_aug_pos[gvp] = gpos + gpos_noise

        self.shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph_nx))    #任意两个节点之间“最短路径经过哪些节点”
        self.shortest_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph_nx)) #任意两个节点之间“最短路径总长度”

    def front_to_ghost_dist(self, ghost_vp):
        # assume the nearest front
        min_dis = 10000
        min_front = None
        for front_vp in self.ghost_fronts[ghost_vp]:
            dis = calc_position_distance(
                self.node_pos[front_vp], self.ghost_aug_pos[ghost_vp]
            )
            if dis < min_dis:
                min_dis = dis
                min_front = front_vp
        return min_dis, min_front

    def get_node_embeds(self, vp, use_oracle: bool = True, allowed_oracle_ghost_ids=None):
        if not vp.startswith('g'):  #如果p是以g开头的，说明是普通节点
            return self.node_embeds[vp] #直接返回节点保存的特征
        else:   #如果是以g开头的，说明是ghost节点，返回最终送入planner的ghost特征。
            return self.get_effective_ghost_embed(
                vp,
                use_oracle=use_oracle,
                allowed_oracle_ghost_ids=allowed_oracle_ghost_ids,
            )

    def get_pos_fts(self, cur_vp, cur_pos, cur_ori, gmap_vp_ids):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vp_ids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            # for ghost
            elif vp.startswith('g'):
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.ghost_aug_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True,
                )
                rel_angles.append([rel_heading, rel_elevation])
                front_dis, front_vp = self.front_to_ghost_dist(vp)
                shortest_dist = self.shortest_dist[cur_vp][front_vp] + front_dis
                shortest_step = len(self.shortest_path[cur_vp][front_vp]) + 1
                rel_dists.append(
                    [rel_dist / MAX_DIST, 
                    shortest_dist / MAX_DIST, 
                    shortest_step / MAX_STEP]
                )
            # for node
            else:
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.node_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True,
                )
                rel_angles.append([rel_heading, rel_elevation])
                shortest_dist = self.shortest_dist[cur_vp][vp]
                shortest_step = len(self.shortest_path[cur_vp][vp])
                rel_dists.append(
                    [rel_dist / MAX_DIST, 
                    shortest_dist / MAX_DIST, 
                    shortest_step / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size=4)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
