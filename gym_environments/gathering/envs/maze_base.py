from typing import Tuple, Union, Type, Dict, Optional, List, Callable

import cv2
import gym
import numpy as np
from gym.utils import seeding

from gym_environments.gathering.envs.observations import (
    SingleChannelObservationGenerator,
    MultiChannelObservationGenerator,
    SingleChannelRealWorldObservationGenerator,
)
from gym_environments.gathering.maze_generators import InstanceGenerator, InstanceReader
from gym_environments.gathering.rewards import GENERATORS
from gym_environments.gathering.envs.step_modifier import (
    StepModifier,
    SimpleMovementModifier,
    FuzzyMovementModifier,
    RandomMovementModifier,
    PhysicalMovementModifier,
)

PARTICLE_MARKER = 255
GOAL_MARKER = 255
BACKGROUND_COLOR = (15, 30, 65)
MAZE_COLOR = (90, 150, 190)
PARTICLE_COLOR = (250, 250, 100)


class MazeBase(gym.Env):
    """
    Base class for a maze-like environment for particle navigation tasks.
    :param instance: (str or list) *.csv file containing the map data. May be a list for random randomized maps.
    :param goal: (Tuple[int, int]) A point coordinate in form [x, y] ([column, row]).
        In case of random or multiple maps needs to be None for a random goal position.
    :param goal_range: (int) Circle radius around the goal position that should be counted as goal reached.
    :param reward_generator: (str) The type of RewardGenerator to use for reward generation. (e.g. "goal" or "continuous")
        based on the current state.
    :param n_particles: (int) Number of particles to spawn in the maze.
        Can be set to -1 for a random number of particles.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        instance: Union[str, Type[InstanceGenerator]],
        goal: Union[Tuple[int, int], Callable],
        goal_range: int,
        reward_generator: str,
        reward_kwargs: Optional[Dict] = None,
        n_particles: int = 256,
        allow_diagonal: bool = True,
        instance_kwargs: Optional[Dict] = None,
        step_type: str = "simple",
        step_kwargs: Optional[Dict] = None,
        observation_type: str = "simple",
        observation_kwargs: Optional[Dict] = None,
    ) -> None:

        self.np_random = None
        self.reward_kwargs = {} if reward_kwargs is None else reward_kwargs
        self.instance_kwargs = {} if instance_kwargs is None else instance_kwargs
        self.observation_kwargs = (
            {} if observation_kwargs is None else observation_kwargs
        )
        self.goal_range = goal_range
        self.locations = None  # Nonzero freespace - not particle locations!
        self.done = False

        if allow_diagonal:
            # self.action_map = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), # {S, SE, E, NE, N, NW, W, SW}
            #                    4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
            self.action_map = {
                0: (0, 1),
                1: (1, 1),
                2: (1, 0),
                3: (1, -1),  # {E, SE, S, SW, W, NW, N, NE}
                4: (0, -1),
                5: (-1, -1),
                6: (-1, 0),
                7: (-1, 1),
            }
        else:
            self.action_map = {
                0: (0, 1),
                1: (1, 0),
                2: (0, -1),
                3: (-1, 0),
            }  # {E, S, W, N}

        self.rev_action_map = {v: k for k, v in self.action_map.items()}
        self.actions = list(self.action_map.keys())
        self.action_space = gym.spaces.Discrete(len(self.action_map))
        self.step_modifiers = []  # type: List[StepModifier]
        self._create_modifiers(step_type, {} if step_kwargs is None else step_kwargs)

        if n_particles < 0:
            self.randomize_n_particles = True
            self.n_particles = 0
        else:
            self.randomize_n_particles = False
            self.n_particles = n_particles

        self.reward_generator_class = GENERATORS[reward_generator]
        self.reward_generator = None

        if isinstance(instance, str):
            self.map_generator = InstanceReader(instance)
        else:
            self.map_generator = instance(**self.instance_kwargs)

        self.map_index = -1
        self.goal_proposition = goal
        if callable(goal):
            self._load_map(None)
        else:
            self._load_map(goal)

        self.n_channels = 1
        self._create_obs_generator(observation_type)
        self.observation_space = self.obs_generator.observation_space

        self.particle_locations = np.array([])
        self.seed()
        self.reset()

    def _create_obs_generator(self, observation_type):
        if observation_type == "simple":
            self.obs_generator = SingleChannelObservationGenerator(
                self.maze,
                self.randomize_goal,
                self.goal_range,
                **self.observation_kwargs
            )
        elif observation_type == "multichannel":
            self.obs_generator = MultiChannelObservationGenerator(
                self.maze,
                self.randomize_goal,
                self.goal_range,
                **self.observation_kwargs
            )
        elif observation_type == "real-world":
            self.obs_generator = SingleChannelRealWorldObservationGenerator(
                self.maze,
                self.randomize_goal,
                self.goal_range,
                **self.observation_kwargs
            )

    def _create_modifiers(self, step_type, step_kwargs):
        if step_type == "simple":
            self.step_modifiers.append(
                SimpleMovementModifier(self.action_map, **step_kwargs)
            )
        elif step_type == "fuzzy":
            self.step_modifiers.append(
                RandomMovementModifier(self.action_map, **step_kwargs)
            )
            self.step_modifiers.append(
                FuzzyMovementModifier(self.action_map, **step_kwargs)
            )
        elif step_type == "physical":
            self.step_modifiers.append(
                PhysicalMovementModifier(self.action_map, **step_kwargs)
            )

    def _load_map(self, goal):
        # Load map if necessary
        self.freespace = self.map_generator.generate(
            success=self.done
        )  # 1: Passable terrain, 0: Wall
        self.maze = (
            np.ones(self.freespace.shape, dtype=np.uint8) - self.freespace
        )  # 1-freespace: 0: Passable terrain, 1: Wall
        self.height, self.width = self.maze.shape
        self.cost = None
        self.locations = np.transpose(np.nonzero(self.freespace))

        if goal:
            self.randomize_goal = False
            self.goal_probability = None
            self.update_goal(goal)
        else:
            self.randomize_goal = True
            self.goal_probability = np.full(
                (len(self.locations),), 1 / len(self.locations)
            )  # Uniform distribution
            self.goal = [0, 0]
            # Random goals require a dynamic cost map which will be calculated on each reset.

    def reset(self):
        if self.map_generator.has_next():
            self._load_map(self.goal_proposition)

        # Randomize number of particles if necessary
        self._randomize_n_particles(self.locations)

        # Randomize goal position if necessary
        self._randomize_goal_position(self.locations)

        # Reset particle positions
        self._randomize_particle_locations(self.locations)

        # Reset modifiers
        for modifier in self.step_modifiers:
            modifier.reset(self.particle_locations, self.maze, self.freespace)

        # Reset observation generator
        self.obs_generator.reset()

        self.done = False
        return self._generate_observation()

    def _randomize_particle_locations(self, locations):
        """
        Computes new locations for all particles.
        :param locations: (list) Number of possible locations for the particles.
        """
        choice = self.np_random.choice(len(locations), self.n_particles, replace=False)
        self.particle_locations = locations[
            choice, :
        ]  # Particle Locations are in y, x (row, column) order
        self.reward_generator.reset(self.particle_locations)

    def _randomize_goal_position(self, locations):
        """
        Computes a new random goal position.
        :param locations: (list) List of possible goal locations to choose from.
        """
        if self.randomize_goal:
            if callable(self.goal_proposition):
                new_goal = self.goal_proposition(locations, self.goal)
            else:
                goal_idx = self.np_random.choice(
                    len(locations), p=self.goal_probability
                )
                new_goal = locations[goal_idx]
            self.update_goal([new_goal[1], new_goal[0]])

    def _randomize_n_particles(self, locations, max_particles=4500):
        """
        Computes a random number of particles for the current map.
        :param locations: (list) Number of free locations for particles
        :param fan_out: (int) Parameter to control the maximum number of particles as a fraction of possible particle locations.
        """
        if self.randomize_n_particles:
            self.n_particles = self.np_random.randint(
                1, min(len(locations), max_particles)
            )

            # If the goal is randomized the reward generator will be replaced on each reset in the update_goal() function.
            if not self.randomize_goal:
                self.reward_generator.set_particle_count(self.n_particles)

    def update_goal(self, goal):
        self.goal = goal
        self.reward_generator = self.reward_generator_class(
            self.maze,
            self.goal,
            self.goal_range,
            self.n_particles,
            self.action_map,
            **self.reward_kwargs
        )

    def update_goal_probs(self, probs: np.ndarray):
        assert len(probs) == len(self.locations)
        self.goal_probability = probs

    def step(self, action):
        info = {}
        location_update = np.copy(self.particle_locations)

        for modifier in self.step_modifiers:
            location_update += modifier.step(action, self.particle_locations)

        valid_locations = self._update_locations(location_update)

        # Inform modifiers about update
        for modifier in self.step_modifiers:
            modifier.step_done(valid_locations)

        done, reward = self.reward_generator.step(action, self.particle_locations)
        if done:
            self.done = True
        return (self._generate_observation(), reward, done, info)

    def render(self, mode="human"):
        rgb_image = np.full((*self.maze.shape, 3), BACKGROUND_COLOR, dtype=int)
        maze_rgb = np.full((*self.maze.shape, 3), MAZE_COLOR, dtype=int)
        rgb_image = np.where(
            np.stack((self.freespace,) * 3, axis=-1), maze_rgb, rgb_image
        )
        rgb_image[
            self.particle_locations[:, 0], self.particle_locations[:, 1]
        ] = PARTICLE_COLOR

        rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        cv2.circle(
            rgb_image, tuple(self.goal), self.goal_range, (255, 0, 0), thickness=1
        )
        rgb_image = np.clip(rgb_image, 0, 255)

        if mode == "human":  # Display image
            rgb_image = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow("image", rgb_image)
            cv2.waitKey(25)
        # rgb_image = cv2.resize(rgb_image.astype(np.uint8), (100, 100), interpolation=cv2.INTER_AREA)
        return rgb_image.astype(np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        for modifier in self.step_modifiers:
            modifier.seed(self.np_random)

        if self.obs_generator is not None:
            self.obs_generator.seed(self.np_random)
        return [seed]

    def _generate_observation(self):
        return self.obs_generator.observation(
            self.maze, self.particle_locations, self.goal
        )

    def _update_locations(self, new_locations):
        """
        Updates the position for each particle based on the new locations, if the new location is valid (not blocked).
        :param new_locations (int)
        """
        # validate_locations = (np.array([self.freespace[tuple(new_loc.T)]]) & (0 <= new_loc[:, 0]) & (new_loc[:, 0] < self.height) & (0 <= new_loc[:, 1]) & (new_loc[:, 1] < self.width)).transpose()
        # Border does not need to be checked as long as all maps have borders.
        valid_locations = self.freespace.ravel()[
            (new_locations[:, 1] + new_locations[:, 0] * self.freespace.shape[1])
        ]
        valid_locations = valid_locations[:, np.newaxis]
        self.particle_locations = np.where(
            valid_locations, new_locations, self.particle_locations
        )
        return valid_locations
