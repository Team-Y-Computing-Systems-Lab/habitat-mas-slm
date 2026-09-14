from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import habitat
import habitat.gym
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig


# Load the RearrangePick task and stop after at most 20 actions.
config = habitat.get_config(
    "benchmark/rearrange/skills/pick.yaml",
    overrides=["habitat.environment.max_episode_steps=20"],
)

# The default pick configuration only contains a depth camera. Add a
# third-person RGB camera so that the robot can be viewed without opening an
# interactive OpenGL window (which is useful when working through VNC).
with habitat.config.read_write(config):
    agent_config = get_agent_config(config.habitat.simulator)
    agent_config.sim_sensors["third_rgb_sensor"] = ThirdRGBSensorConfig(
        height=512,
        width=512,
    )
    config.habitat.gym.obs_keys.append("third_rgb")

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)
png_path = output_dir / "pick_first_frame.png"
gif_path = output_dir / "pick_episode.gif"

env = habitat.gym.make_gym_from_config(config)

try:
    observations = env.reset()
    print("Environment initialized")
    print(f"Observation keys: {list(observations.keys())}")

    first_frame = np.asarray(observations["third_rgb"])[..., :3]
    frames = [first_frame]
    imageio.imwrite(png_path, first_frame)

    terminal = False
    step = 0

    # Random actions demonstrate the simulator; they are not a trained policy.
    while not terminal:
        action = env.action_space.sample()
        observations, reward, terminal, info = env.step(action)
        frames.append(np.asarray(observations["third_rgb"])[..., :3])
        step += 1
        print(f"step={step:02d}, reward={float(reward):.4f}, done={terminal}")

    imageio.mimsave(gif_path, frames, duration=0.15, loop=0)
    print(f"Saved first frame to: {png_path}")
    print(f"Saved episode animation to: {gif_path}")
finally:
    env.close()
