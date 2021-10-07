import gym
import numpy as np

from gym_environments.gathering.envs.maze_base import PARTICLE_MARKER


class InvisibleParticleNoiseWrapper(gym.ObservationWrapper):
    """
    Removes a random number of particles from the observation.
    """

    def __init__(self, env, probability=0.0001, max_particles=20):
        super(InvisibleParticleNoiseWrapper, self).__init__(env)
        self.probability = probability
        self.max_particles = max_particles

    def observation(self, observation):
        particles = np.where(observation == PARTICLE_MARKER)
        random_particles = np.random.randint(0, self.max_particles)
        choice = np.random.choice(
            len(particles[0]), min(len(particles[0]), random_particles), replace=False
        )
        coords = np.asarray(particles)[:, choice]
        observation[tuple(coords)] = 0
        return observation


class FakeParticleNoiseWrapper(gym.ObservationWrapper):
    """
    Adds a fake particle into the observation at random black pixels.
    """

    def __init__(self, env, probability=0.0001):
        super(FakeParticleNoiseWrapper, self).__init__(env)
        self.probability = probability

    def observation(self, observation):
        global_mask = self.np_random.choice(
            [0, 1],
            self.observation_space.shape,
            p=[1 - self.probability, self.probability],
        )
        return np.where(
            np.logical_and(global_mask, observation == 0), PARTICLE_MARKER, observation
        )
