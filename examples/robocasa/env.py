"""RoboCasa environment adapter for OpenPI."""

from typing import Dict, Any, Optional
import numpy as np
import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import json
import h5py
import robosuite

# Import robocasa dependencies
try:
    import robocasa
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
    from robocasa.utils.env_utils import create_env
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
except ImportError as e:
    raise ImportError(
        "RoboCasa not found. Please install RoboCasa in your environment. "
        "See: https://github.com/robocasa/robocasa"
    ) from e

import random
from typing import Optional

def switch_scenes(env: "RoboCasaEnvironment", ep: Optional[int] = None):
    # 收集所有 demo_xxx key

    # 根据 ep 选择 demo
    if ep is None:
        demo_key = random.choice(env.demo_keys)
    else:
        demo_key = f"demo_{ep}"
        if demo_key not in env.demo_keys:
            raise ValueError(f"指定的 demo {demo_key} 不存在")

    # 读取数据
    demo_data = env.f['data'][demo_key]
    states = env.f[f"data/{demo_key}/states"][()]
    ep_meta = json.loads(demo_data.attrs['ep_meta'])
    prompt = ep_meta["lang"]
    
    print("prompt : ",prompt)
    initial_state = dict(states=states[0])
    initial_state["model"] = env.f[f"data/{demo_key}"].attrs["model_file"]
    initial_state["ep_meta"] = ep_meta

    # 设置环境初始状态
    env.initial_state = initial_state
    env.prompt = prompt
    env.std_action = env.f[f"data/{demo_key}/actions"]

from pathlib import Path

def get_initial_state_dir(task_category: str, task_name: str) -> str:
    base_dir = Path("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/datasets/robocasa_minigen/datasets/v0.1/single_stage")
    task_dir = base_dir / task_category / task_name / "mg"

    # 找到 mg 目录下的所有子目录（日期文件夹）
    date_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        raise FileNotFoundError(f"没有找到日期目录: {task_dir}")

    # 如果有多个日期，可以按需要选择，这里我取最新的
    latest_dir = max(date_dirs, key=lambda d: d.name)

    # 拼接最终文件路径
    file_path = latest_dir / "demo_gentex_im128_randcams.hdf5"
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    return str(file_path)
    
class RoboCasaEnvironment(_environment.Environment):
    """An environment adapter for RoboCasa kitchen tasks."""

    def __init__(
        self,
        task_name: str = "CloseDrawer",
        task_catagory: str = "kitchen_drawer",
        robots: str = "PandaMobile",
        camera_names: list = None,
        camera_widths: int = 224,
        camera_heights: int = 224,
        render_height: int = 224,
        render_width: int = 224,
        seed: Optional[int] = None,
        **env_kwargs
    ) -> None:
        """
        Initialize RoboCasa environment.
        
        Args:
            task_name: Name of the RoboCasa task (e.g., "PnPCounterToCab")
            robots: Robot type to use
            camera_names: List of camera names for observations
            camera_widths: Width for camera observations
            camera_heights: Height for camera observations  
            render_height: Height for image rendering to policy
            render_width: Width for image rendering to policy
            seed: Random seed
            **env_kwargs: Additional environment arguments
        """
        self.initial_state_dir = get_initial_state_dir(task_catagory,task_name)
        self.task_name = task_name
        self.render_height = render_height
        self.render_width = render_width
        
        # Default camera setup - use agentview cameras like in documentation
        if camera_names is None:
            camera_names = [
                "robot0_agentview_left",
                "robot0_agentview_right", 
                "robot0_eye_in_hand",
            ]
        self.camera_names = camera_names
        
        # Create the RoboCasa environment using their utility function
        self._env = create_env(
            env_name=task_name,
            robots=robots,
            camera_names=camera_names,
            camera_widths=camera_widths,
            camera_heights=camera_heights,
            seed=seed,
            render_onscreen=False,  # We'll handle rendering separately
            **env_kwargs
        )
        self.f = h5py.File(self.initial_state_dir, "r")
        self.demo_keys = [k for k in self.f['data'].keys() if k.startswith('demo_')]
        switch_scenes(self)
        reset_to(self._env,self.initial_state)
        
        with open("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tmp/actions.json", "r", encoding="utf-8") as action_json:
            self.converted_action = json.load(action_json)
        
        self._last_obs = None
        self._episode_steps = 0
        self._done = False
        self._success = False
        self._success_times = 0
        self._test_times = 0
        

        self.accumulate_action = np.zeros(11, dtype=np.float32)

    
    
    
    @override
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        switch_scenes(self)
        reset_to(self._env,self.initial_state)
        if hasattr(self, '_last_obs') and self._last_obs is not None:
            self._test_times += 1
        raw_obs = self._env.reset()
        self._last_obs = self._convert_observation(raw_obs)
        self._episode_steps = 0
        self._done = False
        self._success = False
        print(f"current result:{self._success_times}/{self._test_times}")
        
        
        

    @override
    def is_episode_complete(self) -> bool:
        """Check if the episode is complete."""
        # return False
        #if self._done or self._success:
        #    self._test_times += 1
        return self._done or self._success

    @override
    def get_observation(self) -> dict:
        """Get the current observation."""
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")
        return self._last_obs

    @override
    def apply_action(self, action: dict) -> None:
        """Apply an action to the environment."""
        # Extract actions from the action dict
        # robot_actions = action["actions"][:11] - self.accumulate_action
        self.accumulate_action = action["actions"][:11].copy()
        
        robot_actions = action["actions"][:11].copy()
        # robot_actions[7:] = 0
        # Step the environment
        action_frame = self._episode_steps
        # if action_frame >= self.std_action.shape[0] :
        #     robot_actions = np.zeros(11, dtype=np.float32)
        # else:
        #     robot_actions = self.std_action[action_frame][:11]

        # robot_actions = self.converted_action[action_frame][:11]
            
        raw_obs, reward, done, info = self._env.step(robot_actions)
        # Convert observation to OpenPI format
        self._last_obs = self._convert_observation(raw_obs)
        self._episode_steps += 1
        self._done = done
        if self._env._check_success() :
            if self._success == False:
                print("episode successed")
                self._success_times += 1
            self._success = True
            
        # print("success:",self._success)

    def _convert_observation(self, raw_obs: dict) -> dict:
        """Convert RoboCasa observation to OpenPI format."""
        import math
        
        # Initialize 12-dimensional state array
        # dim 0/1/2/3/4/5: arm translation and rotation
        # dim 6: gripper
        # dim 7/8/9/10: base
        # dim 11: mode controlling whether to use arm or base
        state = np.zeros(12, dtype=np.float32)
        
        # dim 0/1/2: arm translation (end-effector position)
        if "robot0_eef_pos" in raw_obs:
            state[0:3] = raw_obs["robot0_eef_pos"]
        
        # dim 3/4/5: arm rotation (convert quaternion to euler angles)
        if "robot0_eef_quat" in raw_obs:
            x, y, z, w = raw_obs["robot0_eef_quat"]
            
            # quat_to_euler conversion
            # roll (x-axis rotation)
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(t0, t1)

            # pitch (y-axis rotation)
            t2 = 2.0 * (w * y - z * x)
            if t2 > 1.0: t2 = 1.0
            if t2 < -1.0: t2 = -1.0
            pitch = math.asin(t2)

            # yaw (z-axis rotation)
            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(t3, t4)
            
            state[3:6] = [roll, pitch, yaw]

        # dim 6: gripper (difference between gripper joint positions)
        if "robot0_gripper_qpos" in raw_obs:
            gripper_qpos = raw_obs["robot0_gripper_qpos"]
            if len(gripper_qpos) >= 2:
                state[6] = 0.5 * (gripper_qpos[0] - gripper_qpos[1])
            else:
                # If only one gripper value, use it directly
                state[6] = gripper_qpos[0]
        
        # dim 7/8/9/10: base quaternion
        if "robot0_base_quat" in raw_obs:
            state[7:11] = raw_obs["robot0_base_quat"]
        elif "robot0_quat" in raw_obs:
            # Alternative key name for base quaternion
            state[7:11] = raw_obs["robot0_quat"]
        
        # dim 11: mode control (-1.0 for default arm control mode)
        state[11] = -1.0

        
        
        # Process camera images to match training config expectations
        images = {}
        
        # Map cameras according to training config:
        # cam_left_wrist -> robot0_agentview_left
        # cam_right_wrist -> robot0_agentview_right  
        # cam_high -> robot0_eye_in_hand
        camera_mapping = {
            "cam_left_wrist": "robot0_agentview_left",
            "cam_right_wrist": "robot0_agentview_right",
            "cam_high": "robot0_eye_in_hand"
        }
        
        for openpi_cam_name, robocasa_cam_name in camera_mapping.items():
            if f"{robocasa_cam_name}_image" in raw_obs:
                img = raw_obs[f"{robocasa_cam_name}_image"]
                
                # Convert to uint8 and resize
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, self.render_height, self.render_width)
                )
                # import matplotlib.pyplot as plt
                # plt.imshow(img)
                # plt.axis("off")   # 不显示坐标轴
                # plt.show()
                # Convert from [H, W, C] to [C, H, W]
                img = np.flip(img, axis=0)
                img = einops.rearrange(img, "h w c -> c h w")
                images[openpi_cam_name] = img
        
        return {
            "state": state,
            "images": images,
            "prompt": self.prompt
        }

    def get_env_info(self) -> dict:
        """Get environment information."""
        return {
            "task_name": self.task_name,
            "action_dim": 12,  # Fixed to match training config
            "state_dim": 12,   # State dimension should match action dimension
            "action_spec": (self._env.action_spec[0], self._env.action_spec[1]),  # (low, high)
            "cameras": ["cam_left_wrist", "cam_right_wrist", "cam_high"],  # OpenPI camera names
            "robocasa_cameras": self.camera_names,  # Original RoboCasa camera names
        }

def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = state["ep_meta"]
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None        

if __name__ == "__main__":
    robocasa_env = RoboCasaEnvironment(
        task_name="PnPCounterToCab",
        robots="PandaMobile",
        seed=0
    )
    action = np.random.randn(11) * 0.1
    obs, reward, done, info = robocasa_env._env.step(action)
    print(obs)
