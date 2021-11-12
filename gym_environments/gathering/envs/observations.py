from abc import ABC
from typing import Tuple

import cv2
import gym
import numpy as np

from gym_environments.gathering.envs import maze_base


class ObservationGenerator(ABC):
    def __init__(
        self, maze: np.ndarray, random_goal: bool, goal_range: int, noise: float = 0.0
    ):
        self.np_random = np.random.random.__self__
        self.observation_space = None
        self.random_goal = random_goal
        self.goal_range = goal_range
        self.noise = noise
        pass

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        pass

    def render_particles(self, particles: np.ndarray, maze: np.ndarray, out=None):
        out = out if out is not None else np.zeros(maze.shape)
        out[particles[:, 0], particles[:, 1]] = maze_base.PARTICLE_MARKER

        if self.noise > 0.0:
            n_salt = np.ceil(self.noise * out.size)
            coords = [self.np_random.randint(0, i - 1, int(n_salt)) for i in out.shape]
            out[tuple(coords)] = maze_base.PARTICLE_MARKER

            coords = [self.np_random.randint(0, i - 1, int(n_salt)) for i in out.shape]
            out[tuple(coords)] = 0

            out = out * (1 - maze)  # Restrict noise to the maze area
        return out

    def render_maze(self, maze):
        return maze * 255

    def render_goal(self, maze: np.ndarray, goal: Tuple[int, int], out=None):
        out = out if out is not None else np.zeros(maze.shape)
        cv2.circle(out, tuple(goal), self.goal_range, (maze_base.GOAL_MARKER))
        out[
            goal[1] - 1 : goal[1] + 1, goal[0] - 1 : goal[0] + 1
        ] = maze_base.GOAL_MARKER
        return out

    def seed(self, np_random: np.random.Generator):
        self.np_random = np_random


class SingleChannelObservationGenerator(ObservationGenerator):
    def __init__(
        self, maze: np.ndarray, random_goal: bool, goal_range: int, noise: float = 0.0
    ):
        super(SingleChannelObservationGenerator, self).__init__(
            maze, random_goal, goal_range, noise
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, 1), dtype=np.uint8
        )

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        observation = np.zeros(maze.shape)
        observation = self.render_particles(particles, maze, out=observation)

        if self.random_goal:
            observation = self.render_goal(maze, goal, out=observation)

        return observation[:, :, np.newaxis]  # Convert to single channel image


class MultiChannelObservationGenerator(ObservationGenerator):
    def __init__(
        self, maze: np.ndarray, random_goal: bool, goal_range: int, noise: float = 0.0
    ):
        super(MultiChannelObservationGenerator, self).__init__(
            maze, random_goal, goal_range, noise
        )

        self.n_channels = 3 if random_goal else 2
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, self.n_channels), dtype=np.uint8
        )

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        observation = np.zeros((*maze.shape, self.n_channels))
        observation[:, :, 0] = self.render_maze(maze)
        observation[:, :, 1] = self.render_particles(particles, maze)
        if self.random_goal:
            observation[:, :, 2] = self.render_goal(maze, goal)

        return observation
