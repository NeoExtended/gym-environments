from gym.envs.registration import register


register(
    id="GravityUnloading-v0",
    entry_point="gym_environments.gravity_packing.gravity_env:UnloadingEnvironment",
    max_episode_steps=200,
)
