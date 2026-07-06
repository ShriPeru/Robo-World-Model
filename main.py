import torch
from pathlib import Path
import mujoco
import mujoco.viewer
def check_torch():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def check_mujoco(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    print("MuJoCo version:", mujoco.mj_versionString())
    print("Model loaded from:", model_path.name)
    mjData = mujoco.MjData(model)
    print("Data initialized, time:", mjData.time)
    return model, mjData

def step_mujoco(model, joint_name, data, n_steps=100):
    joint_info = model.joint(joint_name)
    qpos_start_address = joint_info.qposadr[0]
    qvel_start_address = joint_info.dofadr[0]

    list_qpos = []
    list_qvel = []
    list_time = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break
            mujoco.mj_step(model, data)
            z_pos = data.qpos[qpos_start_address + 2]
            z_vel = data.qvel[qvel_start_address + 2]
            x_pos = data.qpos[qpos_start_address + 0]
            x_vel = data.qvel[qvel_start_address + 0]
            y_pos = data.qpos[qpos_start_address + 1]
            y_vel = data.qvel[qvel_start_address + 1]
            print(f"Step {i+1}:")
            print("  Time:", data.time)
            print("  Position of falling sphere:", round(z_pos, 4))
            print("  Velocity of falling sphere:", round(z_vel, 4))
            print("  Position of falling sphere (x):", round(x_pos, 4))
            print("  Velocity of falling sphere (x):", round(x_vel, 4))
            print("  Position of falling sphere (y):", round(y_pos, 4))
            print("  Velocity of falling sphere (y):", round(y_vel, 4))
            list_qpos.append(z_pos)
            list_qvel.append(z_vel)
            list_time.append(data.time)
            viewer.sync()

    
    return list_qpos, list_qvel, list_time

def close_mujoco(model, data):
    del data
    del model

if __name__ == "__main__":
    check_torch()
    base_dir = Path(__file__).resolve().parent
    xml_path = base_dir / "src/assets/test_scene.xml"
    model, data = check_mujoco(xml_path)
    joint_name = "falling_sphere_joint"
    qpos, qvel, time = step_mujoco(model, joint_name, data, n_steps=10000)
    close_mujoco(model, data)
