import torch
from pathlib import Path
import mujoco
import mujoco.viewer

# import data_logger module
from src.simulation import data_logger

data_logger = data_logger.TrajectoryLogger(output_dir=Path("src/simulation/trajectory_data"), buffer_size=10010)

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

def step_mujoco(model, joint_name, data, n_steps=100, logger=None, episode_id=None):
    joint_info = model.joint(joint_name)
    qpos_start_address = joint_info.qposadr[0]
    qvel_start_address = joint_info.dofadr[0]

    list_qpos = []
    list_qvel = []
    list_time = []

    def get_state():
        pos = [
            data.qpos[qpos_start_address + 0],  # x
            data.qpos[qpos_start_address + 1],  # y
            data.qpos[qpos_start_address + 2],  # z
        ]
        vel = [
            data.qvel[qvel_start_address + 0],  # x
            data.qvel[qvel_start_address + 1],  # y
            data.qvel[qvel_start_address + 2],  # z
        ]
        return [pos, vel]  # matches before_state[0]/[1] shape in log()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break

            before_state = get_state()   # snapshot BEFORE stepping
            mujoco.mj_step(model, data)
            after_state = get_state()    # snapshot AFTER stepping

            if logger is not None:
                logger.log(
                    episode_id=episode_id,
                    timestep=i,
                    before_state=before_state,
                    after_state=after_state,
                    action=None,
                    reward=None,
                )

            z_pos = after_state[0][2]
            z_vel = after_state[1][2]
            print(f"Step {i+1}:  Time: {data.time}  z_pos: {round(z_pos,4)}  z_vel: {round(z_vel,4)}")

            list_qpos.append(after_state[0][2])
            list_qvel.append(after_state[1][2])
            list_time.append(data.time)
            viewer.sync()

    return list_qpos, list_qvel, list_time

def run_mujoco_simulations(model, joint_name, data, n_simulations=5, n_steps=100):
    for sim in range(n_simulations):
        data.time = 0.0
        mujoco.mj_resetData(model, data)
        print(f"Running simulation {sim + 1}/{n_simulations}")
        print(f"Initial conditions for simulation {sim + 1}:")
        randomize_initial_conditions(model, data, shouldrandomize=True)
        step_mujoco(model, joint_name, data, n_steps, logger=data_logger, episode_id=sim + 1)
        data_logger.save_to_parquet(
            data_logger.output_dir / f"trajectory_data_ep{sim + 1}.parquet"
        )
        print(f"Simulation {sim + 1} completed and data saved.")

# every simulation runs with randomized conditions if specified, otherwise it runs with the same initial conditions
def randomize_initial_conditions(model, data, shouldrandomize=True):
    joint_info = model.joint("falling_sphere_joint")
    qpos_start_address = joint_info.qposadr[0]
    qvel_start_address = joint_info.dofadr[0]

    if shouldrandomize:
        # Randomize the initial position and velocity of the joint in the MuJoCo state.
        data.qpos[qpos_start_address + 2] = 1.0 + 0.5 * (2 * torch.rand(1).item() - 1)  # z position in [0.5, 1.5]
        data.qvel[qvel_start_address + 2] = 0.5 * (2 * torch.rand(1).item() - 1)  # z velocity in [-0.5, 0.5]
        data.qpos[qpos_start_address + 0] = 1.0 + 0.5 * (2 * torch.rand(1).item() - 1)  # x position in [0.5, 1.5]
        data.qvel[qvel_start_address + 0] = 0.5 * (2 * torch.rand(1).item() - 1)  # x velocity in [-0.5, 0.5]
        data.qpos[qpos_start_address + 1] = 1.0 + 0.5 * (2 * torch.rand(1).item() - 1)  # y position in [0.5, 1.5]
        data.qvel[qvel_start_address + 1] = 0.5 * (2 * torch.rand(1).item() - 1)  # y velocity in [-0.5, 0.5]
    else:
        data.qpos[qpos_start_address + 0] = 1.0
        data.qvel[qvel_start_address + 0] = 0.0
        data.qpos[qpos_start_address + 1] = 1.0
        data.qvel[qvel_start_address + 1] = 0.0
        data.qpos[qpos_start_address + 2] = 1.0
        data.qvel[qvel_start_address + 2] = 0.0

    mujoco.mj_forward(model, data)
    print(
        f"Randomized initial conditions: qpos[2]={data.qpos[qpos_start_address + 2]}, "
        f"qvel[2]={data.qvel[qvel_start_address + 2]}"
    )

def close_mujoco(model, data):
    del data
    del model

if __name__ == "__main__":
    check_torch()
    base_dir = Path(__file__).resolve().parent
    xml_path = base_dir / "src/assets/test_scene.xml"
    model, data = check_mujoco(xml_path)
    joint_name = "falling_sphere_joint"
    qpos, qvel, time = step_mujoco(model, joint_name, data, n_steps=1000)
    run_mujoco_simulations(model, joint_name, data, n_simulations=5, n_steps=1000)
    close_mujoco(model, data)
