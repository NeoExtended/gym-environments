from abc import ABC
from typing import Tuple, Optional

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
        self.np_random = np.random.random.__self__

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        pass

    def render_particles(self, particles: np.ndarray, maze: np.ndarray, out=None):
        out = out if out is not None else np.zeros(maze.shape)
        out[particles[:, 0], particles[:, 1]] = maze_base.PARTICLE_MARKER
        return out

    def generate_noise(
        self, image, maze: Optional[np.ndarray] = None, noise_type: str = "s&p"
    ):
        out = image
        if self.noise > 0.0:
            if noise_type == "s&p":
                out = self.salt_and_pepper_noise(image, self.noise)
            elif noise_type == "gauss":
                out = self.gaussian_noise(image, self.noise)
            else:
                raise NotImplementedError(f"Unknown noise type {noise_type}")

            # Restrict noise to the maze area
            if maze is not None:
                out = image * (1 - maze)
        return out

    def gaussian_noise(self, image, strength):
        row, col = image.shape
        mean = 0
        var = strength
        sigma = var ** 0.5
        gauss = self.np_random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = np.clip(image + gauss * 255, 0, 255)
        return noisy

    def salt_and_pepper_noise(self, image, strength):
        n_salt = np.ceil(strength * image.size)
        coords = [self.np_random.randint(0, i - 1, int(n_salt)) for i in image.shape]
        image[tuple(coords)] = maze_base.PARTICLE_MARKER

        coords = [self.np_random.randint(0, i - 1, int(n_salt)) for i in image.shape]
        image[tuple(coords)] = 0
        return image

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

    def reset(self):
        pass

    def seed(self, np_random: np.random.Generator):
        self.np_random = np_random


class SingleChannelRealWorldObservationGenerator(ObservationGenerator):
    def __init__(
        self,
        maze: np.ndarray,
        random_goal: bool,
        goal_range: int,
        noise: float = 0.0,
        dirt_noise: float = 0.0,
        real_world_fac: float = 2,
        max_displacement: int = 5,
        max_crop: int = 5,
    ):
        super(SingleChannelRealWorldObservationGenerator, self).__init__(
            maze, random_goal, goal_range, noise
        )
        self.real_world_fac = real_world_fac
        self.real_world_size = tuple([int(d * self.real_world_fac) for d in maze.shape])
        self.displacement = (0, 0)
        self.crop = (0, 0, 0, 0)
        self.dirt = np.ndarray([])
        self.max_displacement = max_displacement
        self.max_crop = max_crop
        self.dirt_noise = dirt_noise
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, 1), dtype=np.uint8
        )

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        observation = np.zeros(maze.shape)
        observation = self.render_particles(particles, maze, out=observation)
        observation = self.distort(observation, maze)

        if self.random_goal:
            observation = self.render_goal(maze, goal, out=observation)

        return observation[:, :, np.newaxis]  # Convert to single channel image

    def distort(self, observation, maze):
        output_shape = observation.shape

        # Scale up
        observation = cv2.resize(
            observation, self.real_world_size, interpolation=cv2.INTER_AREA
        )

        # Add stationary dirt
        observation = np.clip(observation + self.dirt, 0, 255)

        # Threshold
        ret, particles = cv2.threshold(observation, 200, 255, cv2.THRESH_BINARY)

        # Random Crop
        y, x = particles.shape
        trim_left, trim_right, trim_top, trim_bot = self.crop
        particles = particles[trim_top : y - trim_bot, trim_left : x - trim_right]

        # Translate
        particles = self.shift(particles, self.displacement[0], self.displacement[1],)

        # Add Noise
        noisy = self.generate_noise(particles, noise_type="s&p")
        noisy = self.generate_noise(noisy, noise_type="gauss")

        # Downscale
        downscaled = cv2.resize(noisy, output_shape, interpolation=cv2.INTER_AREA)

        # Threshold
        ret, out = cv2.threshold(downscaled, 80, 255, cv2.THRESH_BINARY)

        # Restrict noise to maze area + 2 pixels
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.dilate((1 - maze), kernel, iterations=2)
        out = out * opened

        return out

    def shift(self, image, tx, ty):
        # The number of pixels
        num_rows, num_cols = image.shape[:2]

        # Creating a translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        # Image translation
        return cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))

    def reset(self):
        self.displacement = (
            self.np_random.randint(-self.max_displacement, self.max_displacement),
            self.np_random.randint(-self.max_displacement, self.max_displacement),
        )

        self.crop = [self.np_random.randint(0, self.max_crop) for _ in range(4)]

        self.dirt = self.salt_and_pepper_noise(
            np.zeros(self.real_world_size), self.dirt_noise
        )


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
        observation = self.generate_noise(observation, maze)

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
        particle_image = self.render_particles(particles, maze)
        particle_image = self.generate_noise(particle_image, maze)
        observation[:, :, 1] = particle_image

        if self.random_goal:
            observation[:, :, 2] = self.render_goal(maze, goal)

        return observation
