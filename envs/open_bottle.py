from base_env import BaseEnv
import torch
import numpy as np
import os
import math
import json
import yaml
import random
from isaacgym.torch_utils import *
from pointnet2_ops import pointnet2_utils
from pytorch3d.ops import sample_farthest_points
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Pointclouds
from random import shuffle
import pytorch3d.transforms as tf
from isaacgym import gymutil, gymtorch, gymapi
# from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import Episode_Buffer
import ipdb


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):

    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u
# franka panda tensor / hand rigid body tensor
def relative_pose(src, dst) :

    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    ip = dst.view(-1, shape[-1])[:, 3:]
    ret = torch.cat((p, ip), dim=1)
    return ret.view(*shape)

class OpenBottle(BaseEnv):

    def __init__(self,cfg, sim_params, physics_engine, device_type, device_id, headless, log_dir=None):
        # env config
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.log_dir = log_dir
        self.up_axis = 'z'
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.device = "cpu"
        self.gripper = False
        self.dataset_path = self.cfg["env"]["asset"]["datasetPath"] #datset/door
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.env_num = cfg["env"]["numEnvs"]
        self.asset_root = cfg["env"]["asset"]["assetRoot"]
        self.asset_num = cfg["env"]["asset"]["AssetNum"]  # 1
        self.load_block = cfg["env"]["asset"]["load_block"]
        assets_list_len = len(cfg["env"]["asset"]["Assets"][self.load_block])  # 1
        self.task = cfg["task"]["task_name"]
        self.unlock = False
        self.seed = cfg["seed"]

        print("Simulator: number of assets", self.asset_num)
        print("Simulator: number of environments", self.env_num)

        if self.asset_num:
            assert (self.env_num % self.asset_num == 0)

        assert (self.asset_num <= assets_list_len)  # the number of used length must less than real length

        self.env_per_asset = self.env_num // self.asset_num

        self.dof_lower_limits_tensor = torch.zeros((self.asset_num, 2), device=self.device)
        self.dof_upper_limits_tensor = torch.zeros((self.asset_num, 2), device=self.device)
        self.mechanism_flag = torch.zeros((self.env_num,),device = self.device)
        self.clock_wise = np.zeros((self.env_num,))
        self.goal_pos_offset_tensor = torch.zeros((self.asset_num, 3), device=self.device)

        self.env_ptr_list = []
        self.obj_loaded = False
        self.franka_loaded = False
        self.table_loaded = False
        # data collection
        self.collectData = self.cfg["env"]["collectData"]
        self.fixed_camera_handle_list = [[] for i in range(self.env_num)]
        self.hand_camera_handle_list = []
        self.fixed_camera_vinv_list = [[] for i in range(self.env_num)]
        self.fixed_camera_proj_list = [[] for i in range(self.env_num)]
        self.fixed_env_origin_list = []

        self.PointDownSampleNum = self.cfg["env"]["PointDownSampleNum"]
        self.action_chosen = np.zeros((self.env_num,self.cfg["env"]["horizon"]),dtype=str)

        super().__init__(cfg=self.cfg, enable_camera_sensors=cfg["env"]["enableCameraSensors"])

        # acquire tensors
        # actor state [num_actors,13]
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # dofs state [num_dofs,2]
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        # rigid body state [num_rigid_bodies,13]
        self.rigid_body_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.cfg["env"]["driveMode"] == "osc":
            self.kp = 100
            self.kv = 2 * math.sqrt(self.kp)
            self.kp_null = 10.
            self.kd_null = 2.0 * np.sqrt(self.kp_null)
            self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))
            self.mm_tensor = gymtorch.wrap_tensor(self.gym.acquire_mass_matrix_tensor(self.sim, "franka"))
            hand_index = self.gym.get_asset_rigid_body_dict(self.franka_asset)["panda_hand"]
            self.j_eef = self.jacobian_tensor[:, hand_index - 1, :, :7]
            self.mm = self.mm_tensor[:, :7, :7]
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.render_all_camera_sensors(self.sim)

        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        self.dof_state_tensor = self.dof_state_tensor.view(self.num_envs, -1, 2)
        self.rigid_body_tensor = self.rigid_body_tensor.view(self.num_envs, -1, 13)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.initial_dof_states = self.dof_state_tensor.clone()
        self.initial_root_states = self.root_tensor.clone()
        self.initial_rigid_body_states = self.rigid_body_tensor.clone()

        # precise slices of tensors
        env_ptr = self.env_ptr_list[0]
        franka1_actor = self.franka_actor_list[0]
        obj1_actor = self.actor_list[0]

        self.hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            franka1_actor,
            "panda_hand",
            gymapi.DOMAIN_ENV
        )
        self.hand_lfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            franka1_actor,
            "panda_leftfinger",
            gymapi.DOMAIN_ENV
        )
        self.hand_rfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            franka1_actor,
            "panda_rightfinger",
            gymapi.DOMAIN_ENV
        )
        # self.basex_rigid_body_index = self.gym.find_actor_rigid_body_index(
        #     env_ptr,
        #     franka1_actor,
        #     "basex",
        #     gymapi.DOMAIN_ENV
        # )
        # print("basex", self.basex_rigid_body_index)
        self.part_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            obj1_actor,
            self.part_rig_name,
            gymapi.DOMAIN_ENV
        )
        self.rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            obj1_actor,
            self.rig_name,
            gymapi.DOMAIN_ENV
        )
        self.base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            obj1_actor,
            self.base_rig_name,
            gymapi.DOMAIN_ENV
        )
        self.two_dof_index = self.gym.find_actor_dof_index(
            env_ptr,
            obj1_actor,
            self.two_dof_name,
            gymapi.DOMAIN_ENV
        )
        self.one_dof_index = self.gym.find_actor_dof_index(
            env_ptr,
            obj1_actor,
            self.one_dof_name,
            gymapi.DOMAIN_ENV
        )
        self.hand_rigid_body_tensor = self.rigid_body_tensor[:, self.hand_rigid_body_index, :]
        self.franka_dof_tensor = self.dof_state_tensor[:, :self.franka_num_dofs, :]

        self.two_dof_tensor = self.dof_state_tensor[:, self.two_dof_index, :]

        self.one_dof_tensor = self.dof_state_tensor[:, self.one_dof_index, :]

        self.part_rigid_body_tensor = self.rigid_body_tensor[:, self.part_rigid_body_index, :]

        self.rigid_body_tensor = self.rigid_body_tensor[:, self.rigid_body_index, :]
        self.franka_root_tensor = self.root_tensor[:, 0, :]  # [num_envs,13]
        self.obj_root_tensor = self.root_tensor[:, 1, :]

        self.obj_actor_dof_max_torque_tensor = torch.zeros((self.num_envs, 2), device=self.device)
        self.obj_actor_dof_upper_limits_tensor = torch.zeros((self.num_envs, 2), device=self.device)
        self.obj_actor_dof_lower_limits_tensor = torch.zeros((self.num_envs, 2), device=self.device)
        self.get_obj_dof_property_tensor()

        self.dof_dim = self.franka_num_dofs + 2  # two dof mechanism
        self.pos_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.stage = torch.zeros((self.num_envs, 2), device=self.device)
        self.open_door_stage = torch.zeros((self.num_envs), device=self.device)

        self.action_speed_scale = cfg["env"]["actionSpeedScale"]

        # params for success rate
        self.success = torch.zeros((self.env_num,), device=self.device)
        self.success_rate = torch.zeros((self.env_num,), device=self.device)
        self.success_buf = torch.zeros((self.env_num,), device=self.device).long()

        self.try_range = 0.99875 # min random_range * open_stage_scale --> 2.35*0.5*0.85
        # flags for switching between training and evaluation mode
        self.train_mode = True
        # self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.hand_pose = self.get_adjust_hand_pose()
        self.adjust_hand_pose = self.hand_pose.clone()

        if cfg["env"]["visualizePointcloud"] == True :
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else :
            self.pointCloudVisualizer = None

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._place_agents(self.num_envs, self.cfg["env"]["envSpacing"])

    def _create_ground_plane(self) :
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _place_agents(self, env_num, spacing):

        print("Simulator: creating agents")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        num_per_row = int(np.sqrt(env_num))
        
        for env_id in range(env_num):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_ptr_list.append(env_ptr)
            self._load_franka(env_ptr, env_id)
            self._load_obj(env_ptr, env_id)
            # self._load_table(env_ptr,env_id)
            self.create_camera(env_ptr, env_id)

    def get_obj_dof_property_tensor(self):
        env_id = 0
        for env, actor in zip(self.env_ptr_list, self.actor_list):
            dof_props = self.gym.get_actor_dof_properties(env, actor)
            # print(dof_props["upper"][0])
            dof_num = self.gym.get_actor_dof_count(env, actor)
            for i in range(dof_num):
                self.obj_actor_dof_max_torque_tensor[env_id, i] = torch.tensor(dof_props['effort'][i],
                                                                                device=self.device)
                self.obj_actor_dof_upper_limits_tensor[env_id, i] = torch.tensor(dof_props['upper'][i],
                                                                                  device=self.device)
                self.obj_actor_dof_lower_limits_tensor[env_id, i] = torch.tensor(dof_props['lower'][i],
                                                                                  device=self.device)
            env_id += 1
    
    def refresh_rotate_lock(self, actions):
        # this function is to make sure the revolute joint cannot rotate if the mechanism is locked and the hand is still lifting up
        cur_z = self.hand_rigid_body_tensor[:, :3]
        lifting = (actions[:,2] - cur_z[:, 2]) > 0.003
        for i in range(self.num_envs):
            if lifting[i] != 0 and self.mechanism_flag[i] == 0:
                # lifting but the mechanism is locked
                # set the damping to be 100
                dof_props = self.gym.get_actor_dof_properties(self.env_ptr_list[i], self.actor_list[i])
                dof_props['damping'][1] = 100.0
                self.gym.set_actor_dof_properties(self.env_ptr_list[i], self.actor_list[i], dof_props)
                # print(f'env {i} damping set to 100')
            else:
                dof_props = self.gym.get_actor_dof_properties(self.env_ptr_list[i], self.actor_list[i])
                dof_props['damping'][1] = 0.0
                self.gym.set_actor_dof_properties(self.env_ptr_list[i], self.actor_list[i], dof_props)
                # print(f'env {i} damping set to 0')

    def refresh_mechanism(self):
        update_index = torch.nonzero((self.open_bottle_stage == 1) & (self.mechanism_flag == 0))
        # print(update_index)
        for i in update_index:
            env = self.env_ptr_list[i]
            actor = self.actor_list[i]
            dof_props = self.gym.get_actor_dof_properties(env, actor)
            dof_props['damping'][0] = 1.0
            # dof_props['effort'][0] = 0.1
            dof_props['upper'][0] = 0.05
            self.gym.set_actor_dof_properties(env, actor, dof_props)
            self.obj_actor_dof_max_torque_tensor[i, 0] = torch.tensor(dof_props['effort'][0],device=self.device)
            self.obj_actor_dof_upper_limits_tensor[i, 0] = torch.tensor(dof_props['upper'][0],device=self.device)
        self.mechanism_flag = self.open_bottle_stage.clone()

    def _load_franka(self, env_ptr, env_id):

        if self.franka_loaded == False:

            self.franka_actor_list = []

            asset_root = self.asset_root
            asset_file = "franka_description/robots/franka_panda.urdf"
            self.gripper_length = 0.11
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
            asset_options.flip_visual_attachments = True
            asset_options.armature = 0.01
            self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            
            self.franka_loaded = True

        franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits = self._get_dof_property(
            self.franka_asset)
        self.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=self.device)
        
        self.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=self.device)

        dof_props = self.gym.get_asset_dof_properties(self.franka_asset)

        if self.cfg["env"]["driveMode"] in ["pos", "ik"]:
            dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][:-3].fill(400.0)
            dof_props["damping"][:-3].fill(40.0)
            dof_props["stiffness"][-3].fill(400.0)
            dof_props["damping"][-3].fill(10.0)
        else:  # osc
            dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:-2].fill(0.0)
            dof_props["damping"][:-2].fill(0.0)
        # grippers
        dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][-2:].fill(200.0)
        dof_props["damping"][-2:].fill(20.0)
        # dof_props["friction"][-2:].fill(200.0)

        # root pose
        initial_franka_pose = self._franka_init_pose()

        # set start dof
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.4
        # grippers open
        default_dof_pos[-2:] = franka_dof_upper_limits[-2:]
        franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype)
        franka_dof_state["pos"] = default_dof_pos

        franka_actor = self.gym.create_actor(
            env_ptr,
            self.franka_asset,
            initial_franka_pose,
            "franka",
            env_id,
            0,
            0)

        # rigid props
        franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_actor)
        for shape in franka_shape_props[-2:]:
            shape.friction = 20
        self.gym.set_actor_rigid_shape_properties(env_ptr, franka_actor, franka_shape_props)
        self.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
        self.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
        self.franka_actor_list.append(franka_actor)

    def _load_table(self, env_ptr, env_id):
        if self.table_loaded == False:

            self.table_actor_list = []

            table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
            asset_options.armature = 0.01
            self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
            
            self.table_loaded = True
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
        table_handle = self.gym.create_actor(env_ptr, self.table_asset, table_pose, "table", env_id, 2, 0)
        self.table_actor_list.append(table_handle)

    def _get_dof_property(self, asset):
        dof_props = self.gym.get_asset_dof_properties(asset)
        #print(dof_props["upper"])
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        for i in range(dof_num) :
            dof_max_torque.append(dof_props['effort'][i])
            dof_lower_limits.append(dof_props['lower'][i])
            dof_upper_limits.append(dof_props['upper'][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits

    def _franka_init_pose(self):

        initial_franka_pose = gymapi.Transform()

        initial_franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        initial_franka_pose.p = gymapi.Vec3(0.6, 0.0, 0.1)

        return initial_franka_pose

    def get_adjust_hand_pose(self):
        target_pos = self.part_rigid_body_tensor[:, :3].clone()
        target_rot = self.part_rigid_body_tensor[:, 3:7].clone()

        down_q = torch.stack(self.num_envs * [torch.tensor([0, -0.7071068, 0.7071068, 0])]).to(self.device).view((self.num_envs, 4))
        # 0.7071068, 0, 0, 0.7071068 0, -0.7071068, 0.7071068, 0
        goal_pos_tensor = quat_apply(target_rot, self.goal_pos_offset_tensor) + target_pos
        self.global_goal_pos = goal_pos_tensor.clone()

        goal_rot_tensor = quat_mul(target_rot, down_q)

        goal_pose = torch.cat([goal_pos_tensor, goal_rot_tensor], dim=-1)
        return goal_pose

    def init_obj_dof_state(self, env_id):
        door_type = env_id // self.env_per_asset
        # door_type = self.asset_indices_for_envs[env_id].item()
        dof_props = self.gym.get_asset_dof_properties(self.asset_list[door_type])
        # print(dof_props['lower'][1], dof_props['upper'][1])
        if self.task == "open_bottle":
            dof_props['damping'][0] = 100.0
            dof_props['upper'][0] = 0.0
            limit_random = self.cfg['env']['asset']['limit_random']
            if np.random.rand() < self.cfg["env"]["clockwise"]:
                # clock wise
                self.clock_wise[env_id] = 1
                random_lower = -(limit_random*np.random.rand()+1-limit_random)* dof_props['upper'][1]
                # random_lower = - dof_props['upper'][1]
                dof_props['lower'][1] = random_lower
                dof_props['upper'][1] = 0.0
                # print(dof_props['lower'][1], dof_props['upper'][1])
            else:
                # counter clock wise
                self.clock_wise[env_id] = 0
                random_upper = (limit_random*np.random.rand()+1-limit_random)* dof_props['upper'][1]
                dof_props['upper'][1] = random_upper
                dof_props['lower'][1] = 0.0
            dof_props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
        else:
            print("Unrecognized task!\nTask should be one of: [leverdoor, rounddoor, opendoor]")
        
        self.gym.set_actor_dof_properties(self.env_ptr_list[env_id], self.actor_list[env_id], dof_props)


    def _load_obj(self, env_ptr, env_id):

        if self.obj_loaded == False :

            self._load_obj_asset()

            self.obj_lower_limits_tensor = self.dof_lower_limits_tensor.repeat_interleave(self.env_per_asset,dim=0)
            self.obj_upper_limits_tensor = self.dof_upper_limits_tensor.repeat_interleave(self.env_per_asset,dim=0)
            self.goal_pos_offset_tensor = self.goal_pos_offset_tensor.repeat_interleave(self.env_per_asset, dim=0)
            
            # # self.dof_lower_limits_tensor 的 shape 是 [1000, 2]
            # # self.asset_indices_for_envs 的 shape 是 [256]
            # # 我們用 gather/indexing 來挑出 256 個 asset 的屬性
            # self.obj_lower_limits_tensor = self.dof_lower_limits_tensor[self.asset_indices_for_envs]
            # self.obj_upper_limits_tensor = self.dof_upper_limits_tensor[self.asset_indices_for_envs]
            
            # # 建立一個「新」的 tensor 來存放 env 的 goal_pos，才不會覆蓋舊的
            # self.env_goal_pos_offset_tensor = self.goal_pos_offset_tensor[self.asset_indices_for_envs]

            self.obj_loaded = True
        
        door_type = env_id // self.env_per_asset
        subenv_id = env_id % self.env_per_asset
        obj_actor = self.gym.create_actor(
            env_ptr,
            self.asset_list[door_type],
            self.pose_list[door_type],
            "bottle-{}-{}".format(door_type, subenv_id),
            env_id,
            1,
            0)
        
        # asset_index = self.asset_indices_for_envs[env_id].item()
        # obj_actor = self.gym.create_actor(
        #     env_ptr,
        #     self.asset_list[asset_index],    # <-- 使用 asset_index
        #     self.pose_list[asset_index],    # <-- 使用 asset_index
        #     f"bottle-{asset_index}-{env_id}", # 命名 (asset_index, env_id)
        #     env_id,
        #     1,
        #     0)
                
        self.actor_list.append(obj_actor)

        self.init_obj_dof_state(env_id)

    def _load_obj_asset(self):

        self.asset_name_list = []
        self.asset_list = []
        self.pose_list = []
        self.actor_list = []

        asset_len = len(self.cfg["env"]["asset"]["Assets"][self.load_block].items())
        asset_len = min(asset_len, self.asset_num)

        random_asset = self.cfg["env"]["asset"]["randomAsset"]
        select_asset = [i for i in range(asset_len)]
        if random_asset:  # if we need random asset list from given dataset, we shuffle the list to be read
            shuffle(select_asset)

        cur = 0

        asset_config_list = []

        # prepare the assets to be used
        for id, (name, val) in enumerate(self.cfg["env"]["asset"]["Assets"][self.load_block].items()):

            if id in select_asset:
                asset_config_list.append((id, (name, val)))

        for id, (name, val) in asset_config_list:

            self.asset_name_list.append(val["name"])
            
            print(f"loding asset ID: {id}, name: {val['name']}")

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.collapse_fixed_joints = True  
            asset_options.use_mesh_materials = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 1024

            asset = self.gym.load_asset(self.sim, self.asset_root, os.path.join(self.dataset_path,val["path"]), asset_options)
            self.asset_list.append(asset)

            with open(os.path.join(self.asset_root, os.path.join(self.dataset_path,val["bounding_box"])), "r") as f:
                bounding_box = json.load(f)
                min_dict = bounding_box["min"]
                max_dict = bounding_box["max"]

            dof_dict = self.gym.get_asset_dof_dict(asset)
            self.one_dof_name = list(dof_dict.keys())[0]
            self.two_dof_name = list(dof_dict.keys())[1] 

            rig_dict = self.gym.get_asset_rigid_body_dict(asset)
            assert (len(rig_dict) == 3) 
            self.part_rig_name = list(rig_dict.keys())[2]
            self.rig_name = list(rig_dict.keys())[1]
            self.base_rig_name = list(rig_dict.keys())[0]
            assert (self.rig_name != "base")
            assert (self.part_rig_name != "base")

            self.pose_list.append(self._obj_init_pose(min_dict, max_dict, val["name"]))

            max_torque, lower_limits, upper_limits = self._get_dof_property(asset)
            self.dof_lower_limits_tensor[cur, :] = torch.tensor(lower_limits, device=self.device)
            self.dof_upper_limits_tensor[cur, :] = torch.tensor(upper_limits, device=self.device)

            
            goal_pos = torch.load(os.path.join(self.asset_root, self.dataset_path, str(val["name"]), "goal_pos.npy"))
            
            self.goal_pos_offset_tensor[cur][0] = goal_pos[0]
            self.goal_pos_offset_tensor[cur][1] = goal_pos[1]
            self.goal_pos_offset_tensor[cur][2] = goal_pos[2]
            
            cur += 1

    def _obj_init_pose(self, min_dict, max_dict, name):
        # {"min": [-0.687565, -0.723071, -0.373959], "max": [0.698835, 0.605562, 0.410705]}
        height = 0.3
        if name == "b6" or name =="b7":
            height = 0.2
        # if name == "xx":
        #     height == 0.25
        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(0.0, 0.0, -min_dict[2]+height)
        cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        return cabinet_start_pose
    

    def reset(self, to_reset="all"):
        if to_reset == "all":
            to_reset = np.ones((self.env_num,))
        self._partial_reset(to_reset)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless:
            self.render()
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.step_graphics(self.sim)
        self._refresh_observation()

        success = self.success.clone()
        self.extras["successes"] = success
        self.extras["success_rate"] = self.success_rate
        return self.obs_buf, self.rew_buf, self.reset_buf, None

    def _partial_reset(self, to_reset="all"):

        """
        reset those need to be reseted
        """

        # if to_reset == "all":
        #     to_reset = np.ones((self.env_num,))
        reseted = False
        for env_id, reset in enumerate(to_reset):
            # is reset:
            if reset.item():
                reset_dof_states = self.initial_dof_states[env_id].clone()
                reset_root_states = self.initial_root_states[env_id].clone()

                self.dof_state_tensor[env_id].copy_(reset_dof_states)
                self.root_tensor[env_id].copy_(reset_root_states)
                reseted = True
                self.progress_buf[env_id] = 0
                self.reset_buf[env_id] = 0
                self.success_buf[env_id] = 0
                self.action_chosen[env_id] = np.zeros(self.cfg["env"]["horizon"],dtype=str)
                self.init_obj_dof_state(env_id)
                
                dof_props = self.gym.get_actor_dof_properties(self.env_ptr_list[env_id], self.actor_list[env_id])
                dof_num = self.gym.get_actor_dof_count(self.env_ptr_list[env_id], self.actor_list[env_id])
                for i in range(dof_num):
                    self.obj_actor_dof_max_torque_tensor[env_id, i] = torch.tensor(dof_props['effort'][i],
                                                                                    device=self.device)
                    self.obj_actor_dof_upper_limits_tensor[env_id, i] = torch.tensor(dof_props['upper'][i],
                                                                                    device=self.device)
                    self.obj_actor_dof_lower_limits_tensor[env_id, i] = torch.tensor(dof_props['lower'][i],
                                                                                    device=self.device)

                self.mechanism_flag[env_id] = 0

        self.gripper = False

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        if reseted:
            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state_tensor)
            )
            self.gym.set_actor_root_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.root_tensor)
            )

    def _refresh_observation(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg["env"]["enableForceSensors"]:
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.render_all_camera_sensors(self.sim)
        if self.cfg["env"]["driveMode"] == "ik":
            self.gym.refresh_jacobian_tensors(self.sim)
        if self.cfg["env"]["driveMode"] == "osc":
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

    def step(self, actions):
        self._perform_actions(actions)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if not self.headless:
            self.render()
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.step_graphics(self.sim)

        self.progress_buf += 1
        self._refresh_observation()
        # if self.progress_buf[0] > 500:
        #     self.cal_success()
        self.open_bottle_stage = ((torch.abs(self.two_dof_tensor[:, 0]) >= 0.85 * 
                                   (self.obj_actor_dof_upper_limits_tensor[:, 1] - self.obj_actor_dof_lower_limits_tensor[:, 1])))
        self.refresh_mechanism()
        done = self.reset_buf.clone()
        success = self.success.clone()
        # self._partial_reset(self.reset_buf)

        self.extras["successes"] = success
        self.extras["success_rate"] = self.success_rate
        return self.obs_buf, self.rew_buf, done, None
    
    def _perform_actions(self, actions):
        # self.actions = actions.clone()
        # Deploy control based on pos i.e. ik
        if self.cfg["env"]["driveMode"] == "pos":
            joints = self.franka_num_dofs - 2
            self.pos_act[:, :joints] = self.pos_act[:, :joints] + actions[:, 0:joints] * self.dt * self.action_speed_scale
            self.pos_act[:, :joints] = tensor_clamp(
                self.pos_act[:, :joints], self.franka_dof_lower_limits_tensor[:joints],
                self.franka_dof_upper_limits_tensor[:joints])
        else:  # osc
            joints = self.franka_num_dofs - 2
            pos_err = actions[:,:3] - self.hand_rigid_body_tensor[:, :3]
            orn_cur = self.hand_rigid_body_tensor[:, 3:7]
            # orn_des = quat_from_euler_xyz(actions[:,-1],actions[:,-2],actions[:,-3])
            orn_err = orientation_error(actions[:,3:7], orn_cur)
            dpose = torch.cat([pos_err, orn_err], -1)
            mm_inv = torch.inverse(self.mm)
            m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
            m_eef = torch.inverse(m_eef_inv)
            # ipdb.set_trace()
            u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (self.kp * dpose).unsqueeze(-1) - self.kv * self.mm @ self.franka_dof_tensor[:,:joints,1].unsqueeze(-1)
            j_eef_inv = m_eef @ self.j_eef @ mm_inv
            u_null = self.kd_null * -self.franka_dof_tensor[:,:,1].unsqueeze(-1) + self.kp_null * (
                    (self.initial_dof_states[:,:self.franka_num_dofs,0].view(self.num_envs,-1,1) - self.franka_dof_tensor[:,:,0].unsqueeze(-1) + np.pi) % (2 * np.pi) - np.pi)
            u_null = u_null[:,:joints]
            u_null = self.mm @ u_null
            u += (torch.eye(joints, device=self.device).unsqueeze(0) - torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null
            self.eff_act[:,:joints] = tensor_clamp(u.squeeze(-1), -self.franka_dof_max_torque_tensor[:joints],self.franka_dof_max_torque_tensor[:joints])
        close = 0.0*torch.ones((self.num_envs, 2), device=self.device)
        open = 0.04*torch.ones((self.num_envs, 2), device=self.device)

        if self.gripper:
            self.pos_act[:,-4:-2] = close
        else:
            self.pos_act[:,-4:-2] = open        

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act.view(-1)))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.eff_act.view(-1)))

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None):
        tensor += torch.rand(tensor.shape, device=self.device) * dist * 2 - dist
        if lower is not None and upper is not None:
            torch.clamp_(tensor, min=lower, max=upper)

    def draw_line_all(self, src, dst, color, cpu=False):
        if not cpu:
            line_vec = np.concatenate([src.cpu().numpy(), dst.cpu().numpy()], axis=1).astype(np.float32)
        else:
            line_vec = np.concatenate([src, dst], axis=1).astype(np.float32)
        # ipdb.set_trace()
        self.gym.clear_lines(self.viewer)
        for env_id in range(self.num_envs):
            self.gym.add_lines(
                self.viewer,
                self.env_ptr_list[env_id],
                1,
                line_vec[env_id, :],
                color
            )
    
    def pc_normalize(self, pc):
        # print("normalize pc", pc.shape)
        # print("max_test", torch.max(pc))
        # print("min_test", torch.min(pc))
        center = torch.mean(pc, dim=-2, keepdim=True)
        pc = pc - center
        m = torch.max(torch.sqrt(torch.sum(pc**2, axis=2)), axis=1, keepdim=True)[0]
        pc = pc / m.unsqueeze(1)
        # print("max_test", torch.max(torch.norm(pc, p=2, dim=-1)))
        # print("min_test", torch.min(torch.norm(pc, p=2, dim=-1)))    
        return pc

    def collect_diff_data(self, flag=True):
        pc = self.compute_point_cloud_state(depth_bar = 2.5, type="fixed")
        rgb_tensor = self.compute_rgb_image_state()
        normalize_flag = False
        if flag:
            pc = self.pc_normalize(pc)
        # self._refresh_pointcloud_visualizer(pc[0])
        joints = self.franka_num_dofs
        robotqpose = (2 * (self.franka_dof_tensor[:, :joints, 0]-self.franka_dof_lower_limits_tensor[:joints])/
                      (self.franka_dof_upper_limits_tensor[:joints] - self.franka_dof_lower_limits_tensor[:joints])) - 1
        robotqvel = self.franka_dof_tensor[:, :joints, 1]
        hand_pos = self.hand_rigid_body_tensor[:,:3]
        hand_rot = self.hand_rigid_body_tensor[:,3:7]
        proprioception_info = torch.cat([robotqpose, robotqvel, hand_pos, hand_rot], dim = -1)
        prev_actions = self.actions
        dof_state = torch.cat([self.one_dof_tensor[:,0].unsqueeze(-1), self.two_dof_tensor[:,0].unsqueeze(-1)], dim=-1)

        obs = {"pc": pc, 
               "proprioception": proprioception_info, 
               "dof_state": dof_state, 
               "prev_action": prev_actions, 
               "rgb": rgb_tensor}
        return obs
    
    def collect_single_diff_data(self, env_id):
        pc = self.compute_single_point_cloud(depth_bar = 2.5, env_id=env_id)
        # self._refresh_pointcloud_visualizer(pc[0,:,:])
        joints = self.franka_num_dofs
        robotqpose = (2 * (self.franka_dof_tensor[env_id, :joints, 0]-self.franka_dof_lower_limits_tensor[:joints])/
                        (self.franka_dof_upper_limits_tensor[:joints] - self.franka_dof_lower_limits_tensor[:joints])) - 1
        robotqvel = self.franka_dof_tensor[env_id, :joints, 1]
        hand_pos = self.hand_rigid_body_tensor[env_id,:3]
        hand_rot = self.hand_rigid_body_tensor[env_id,3:7]
        proprioception_info = torch.cat([robotqpose, robotqvel, hand_pos, hand_rot], dim = -1)
        prev_actions = self.actions[env_id]
        dof_state = torch.cat([self.one_dof_tensor[env_id,0].unsqueeze(-1), self.two_dof_tensor[env_id,0].unsqueeze(-1)], dim=-1)
        obs = {"pc": pc.squeeze(0), "proprioception": proprioception_info, "dof_state": dof_state, "prev_action": prev_actions}
        return obs
        
    def compute_single_point_cloud(self, depth_bar, env_id):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg["env"]["cam"]["width"]
        camera_props.height = self.cfg["env"]["cam"]["height"]
        camera_u = torch.arange(0, camera_props.width, device=self.device)
        camera_v = torch.arange(0, camera_props.height, device=self.device)
        camera_v2, camera_u2 = torch.meshgrid(camera_v, camera_u, indexing='ij')
        self.gym.start_access_image_tensors(self.sim)
        env_ptr = self.env_ptr_list[env_id]
        all_points = None
        for cam_id in range(self.num_cam):
            fixed_camera_handle = self.fixed_camera_handle_list[env_id][cam_id]
            cam_vinv = self.fixed_camera_vinv_list[env_id][cam_id]
            cam_proj = self.fixed_camera_proj_list[env_id][cam_id]
            cam_array = self.gym.get_camera_image(self.sim, env_ptr, fixed_camera_handle, gymapi.IMAGE_DEPTH)
            cam_tensor = torch.tensor(cam_array, device=self.device)
            points = self.depth_image_to_point_cloud_GPU(cam_tensor, cam_vinv,
                                                        cam_proj, camera_u2, camera_v2,
                                                        camera_props.width, camera_props.height, depth_bar)
            if all_points is None:
                all_points = points
            else:
                all_points = torch.cat([all_points, points], dim=0)
        selected_points = self.sample_points(all_points, sample_num=self.PointDownSampleNum, sample_method='furthest')
        
        selected_points = selected_points - self.fixed_env_origin_list[env_id]
        return selected_points
    
    def compute_rgb_image_state(self):
        """
        (This is a new helper function)
        Mirrors the logic of compute_point_cloud_state, but fetches RGB images.
        """
        cam_height = self.cfg["env"]["cam"]["height"]
        cam_width = self.cfg["env"]["cam"]["width"]
        
        rgb_images_list = []
        
        # 1. Start accessing image tensors
        self.gym.start_access_image_tensors(self.sim)
        
        # 2. Iterate through all environments (using the correct list: self.env_ptr_list)
        for env_id, env_ptr in enumerate(self.env_ptr_list): 
            
            # 3. Get camera handle (assuming the 0th fixed camera for recording)
            cam_handle = self.fixed_camera_handle_list[env_id][0] 
            
            # 4. Get "IMAGE_COLOR" (RGBA image)
            rgba_image = self.gym.get_camera_image(self.sim, 
                                                    env_ptr, 
                                                    cam_handle, 
                                                    gymapi.IMAGE_COLOR)
            
            # 5. Reshape and append to list
            rgba_image_formatted = rgba_image.reshape(cam_height, cam_width, 4)
            rgb_images_list.append(rgba_image_formatted)
        
        # 6. Stop accessing image tensors
        self.gym.end_access_image_tensors(self.sim)

        # 7. Stack the list into one large (num_envs, H, W, 4) tensor
        rgb_tensor = torch.from_numpy(np.stack(rgb_images_list)).to(self.device)
        
        return rgb_tensor

    def compute_point_cloud_state(self, depth_bar, type="fixed"):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg["env"]["cam"]["width"]
        camera_props.height = self.cfg["env"]["cam"]["height"]
        camera_u = torch.arange(0, camera_props.width, device=self.device)
        camera_v = torch.arange(0, camera_props.height, device=self.device)
        camera_v2, camera_u2 = torch.meshgrid(camera_v, camera_u, indexing='ij')
        # self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        # fixed point cloud
        point_clouds = torch.zeros((self.num_envs, self.PointDownSampleNum, 3), device=self.device)
        if type == "fixed":
            for i in range(self.num_envs):
                env_ptr = self.env_ptr_list[i]
                all_points = None
                for cam_id in range(self.num_cam):
                    fixed_camera_handle = self.fixed_camera_handle_list[i][cam_id]
                    cam_vinv = self.fixed_camera_vinv_list[i][cam_id]
                    cam_proj = self.fixed_camera_proj_list[i][cam_id]
                    cam_array = self.gym.get_camera_image(self.sim, env_ptr, fixed_camera_handle, gymapi.IMAGE_DEPTH)
                    cam_tensor = torch.tensor(cam_array, device=self.device)
                    points = self.depth_image_to_point_cloud_GPU(cam_tensor, cam_vinv,
                                                                cam_proj, camera_u2, camera_v2,
                                                                camera_props.width, camera_props.height, depth_bar)

                    if all_points is None:
                        all_points = points
                    else:
                        all_points = torch.cat([all_points, points], dim=0)

                selected_points = self.sample_points(all_points, sample_num=self.PointDownSampleNum, sample_method='furthest')
                point_clouds[i] = selected_points - self.fixed_env_origin_list[i]
        else:
            for i in range(self.num_envs):
                env_ptr = self.env_ptr_list[i]
                hand_camera_handle = self.hand_camera_handle_list[i]
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, hand_camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, hand_camera_handle), device=self.device)
                cam_array = self.gym.get_camera_image(self.sim, env_ptr, hand_camera_handle, gymapi.IMAGE_DEPTH)
                cam_tensor = torch.tensor(cam_array, device=self.device)
                points = self.depth_image_to_point_cloud_GPU(cam_tensor, cam_vinv,
                                                            cam_proj, camera_u2, camera_v2,
                                                            camera_props.width, camera_props.height, 0.5)

                selected_points = self.sample_points(points, sample_num=self.PointDownSampleNum, sample_method='furthest')
                
                point_clouds[i] = selected_points - self.fixed_env_origin_list[i]

        self.gym.end_access_image_tensors(self.sim)

        return point_clouds

    def sample_points(self, points, sample_num=1000, sample_method='random'):

        eff_points = points[points[:, 2] > 0.1].contiguous()
        # eff_points = points.contiguous()
        eff_points = eff_points.reshape(1, *eff_points.shape)
        sampled_points, idx = sample_farthest_points(points = eff_points, K=sample_num)

        return sampled_points

    def mkdir(self, path):
        import os
        path = path.strip()
        path = path.rstrip("\\")

        isExists = os.path.exists(path)

        if not isExists:
            os.makedirs(path)
            return True
        else:
            return False

    def depth_image_to_point_cloud_GPU(self, camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width,
                                       height,
                                       depth_bar):
        # time1 = time.time()
        depth_buffer = camera_tensor.to(self.device)
        # Get the camera view matrix and invert it to transform points from camera to world space
        vinv = camera_view_matrix_inv.to(self.device)
        # Get the camera projection matrix and get the necessary scaling
        # coefficients for deprojection
        proj = camera_proj_matrix.to(self.device)
        fu = 2 / proj[0, 0]
        fv = 2 / proj[1, 1]
        centerU = width / 2
        centerV = height / 2
        Z = depth_buffer
        X = -(u - centerU) / width * Z * fu
        Y = (v - centerV) / height * Z * fv
        Z = Z.view(-1)
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)
        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[:, valid]
        position = position.permute(1, 0)
        position = position @ vinv
        points = position[:, 0:3]
        return points

    def _refresh_pointcloud_visualizer(self, point_clouds) :

        if isinstance(point_clouds, list) :
            points = np.concatenate([a.detach().cpu().numpy() for a in point_clouds], axis=0)
        else :
            points = point_clouds.detach().cpu().numpy()
        
        import open3d as o3d

        self.o3d_pc.points = o3d.utility.Vector3dVector(points)
        self.o3d_pc.paint_uniform_color([0, 1, 1])

        if self.pointCloudVisualizerInitialized == False :
            self.pointCloudVisualizer.add_geometry(self.o3d_pc)
            self.pointCloudVisualizerInitialized = True
        else :
            self.pointCloudVisualizer.update(self.o3d_pc)

    def create_camera(self, env_ptr, env_id):
        # fixed camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg["env"]["cam"]["width"]
        camera_props.height = self.cfg["env"]["cam"]["height"]
        camera_props.far_plane = self.cfg["env"]["cam"]["cam_far_plane"]
        camera_props.near_plane = self.cfg["env"]["cam"]["cam_near_plane"]
        camera_props.horizontal_fov = self.cfg["env"]["cam"]["cam_horizontal_fov"]

        camera_props.enable_tensors = True
        num_cam = len(self.cfg["env"]["cam"]["cam_start"])
        #print(self.cfg["env"]["cam"]["cam_start"][0])
        self.num_cam = num_cam

        # add fixed camera
        for i in range(num_cam):
            fixed_camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            camera_start = gymapi.Vec3(self.cfg["env"]["cam"]["cam_start"][i][0], self.cfg["env"]["cam"]["cam_start"][i][1], self.cfg["env"]["cam"]["cam_start"][i][2])
            cemera_target = gymapi.Vec3(self.cfg["env"]["cam"]["cam_target"][i][0], self.cfg["env"]["cam"]["cam_target"][i][1], self.cfg["env"]["cam"]["cam_target"][i][2])
            fixed_camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_location(fixed_camera_handle, env_ptr, camera_start, cemera_target)
            self.fixed_camera_handle_list[env_id].append(fixed_camera_handle)
        
            fixed_cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, fixed_camera_handle)))).to(self.device)
            fixed_cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, fixed_camera_handle), device=self.device)
            self.fixed_camera_vinv_list[env_id].append(fixed_cam_vinv)
            self.fixed_camera_proj_list[env_id].append(fixed_cam_proj)

        # hand camera
        hand_camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        camera_offset = gymapi.Vec3(0.08, 0, 0) #0.08, 0.0, 0.0
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(-90))
        actor_handle = self.gym.get_actor_handle(env_ptr, 0)
        # print("actor_handle: {}".format(actor_handle))
        hand_handle = self.gym.get_actor_rigid_body_handle(env_ptr, actor_handle, 8)
        
        self.gym.attach_camera_to_body(hand_camera_handle, env_ptr, hand_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
        self.hand_camera_handle_list.append(hand_camera_handle)

        self.gym.step_graphics(self.sim)

        fixed_cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, fixed_camera_handle)))).to(self.device)
        fixed_cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, fixed_camera_handle), device=self.device)
        fixed_origin = self.gym.get_env_origin(env_ptr)
        env_origin = torch.zeros((1, 3), device=self.device)
        env_origin[0, 0] = fixed_origin.x
        env_origin[0, 1] = fixed_origin.y
        env_origin[0, 2] = fixed_origin.z
        
        # self.fixed_camera_vinv_list.append(fixed_cam_vinv)
        # self.fixed_camera_proj_list.append(fixed_cam_proj)
        self.fixed_env_origin_list.append(env_origin)