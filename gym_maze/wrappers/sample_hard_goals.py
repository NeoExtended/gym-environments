import logging

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
    VecEnv,
)


class VecHardGoalSampleWrapper(VecEnvWrapper):
    def __init__(
        self, venv: VecEnv,
    ):
        super(VecHardGoalSampleWrapper, self).__init__(venv)

        self.episode_returns = None
        self.episode_lengths = None
        self.goal_locations = venv.get_attr("locations", 0)[0]
        maze = venv.get_attr("maze", 0)[0]
        self.avg_rewards = np.zeros(maze.shape)
        self.goal_sampled = np.zeros(maze.shape)
        self.avg_rewards[self.goal_locations[:, 0], self.goal_locations[:, 1]] = np.inf
        self.goal_sampled[self.goal_locations[:, 0], self.goal_locations[:, 1]] = 0

        self.alpha = 0.65
        self.all_goals_sampled = False

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                goal = self.venv.get_attr("goal", i)[0]
                if self.avg_rewards[goal[1], goal[0]] != np.inf:
                    self.avg_rewards[goal[1], goal[0]] = (
                        self.alpha * self.avg_rewards[goal[1], goal[0]]
                        + (1 - self.alpha) * episode_return
                    ) / 2.0
                else:
                    self.avg_rewards[goal[1], goal[0]] = episode_return

                if not self.all_goals_sampled:
                    self.goal_sampled[goal[1], goal[0]] = 1
                    logging.info(np.sum(self.goal_sampled))
                    if np.sum(self.goal_sampled) == len(self.goal_locations):
                        self.all_goals_sampled = True
                else:
                    rewards = self.avg_rewards[
                        self.goal_locations[:, 0], self.goal_locations[:, 1]
                    ]
                    probs = rewards / np.sum(rewards)
                    self.venv.env_method("update_goal_probs", probs, i)

                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return obs, rewards, dones, new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()
