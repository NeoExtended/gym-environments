import cv2
import gym
import numpy as np
from gym.utils import seeding


class UnloadingEnvironment(gym.Env):
    def __init__(
        self, space: int = 100, n_container: int = 10, container_max: int = 10
    ):
        assert n_container > 0 and container_max > 0

        self.space = space
        self.container_max = container_max
        self.n_container = n_container

        self.cargo = None
        self.free = None
        self.containers = []
        self.container_len = []
        self.current = 0
        self.gravity_min = 0
        self.gravity_max = 0
        self.gravity_range = 0
        self.gravity_penalty = 0.2

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, space, 1), dtype=np.uint8
        )

        self.action_space = gym.spaces.Discrete(3)  # Left, Right, Unload

        self.seed()

    def render(self, mode="human"):
        image = self.observation()

        rgb_image = np.stack([image] * 3, axis=2)
        if mode == "human":  # Display image
            rgb_image = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow("image", rgb_image)
            cv2.waitKey(25)
        return rgb_image

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.cargo = np.zeros((self.space,))
        self.free = np.ones((self.space,))
        self.containers = []
        self.container_len = []

        for i in range(self.container_max):
            start = self.np_random.choice(np.nonzero(self.free)[0])
            size = self.np_random.randint(1, self.container_max)

            while np.sum(self.free[start : start + size]) != size:
                size = size // 2

            self.cargo[start : start + size] = 1
            self.containers.append(start)
            self.container_len.append(size)

            if start > 0 and start + size + 1 < self.space:
                self.free[start - 1 : start + size + 1] = 0
            else:
                if start == 0 and start + size + 1 < self.space:
                    self.free[start : start + size + 1] = 0
                else:
                    self.free[start - 1 : start + size] = 0

        order = sorted(
            zip(self.containers, self.container_len), key=lambda pair: pair[0]
        )
        self.containers, self.container_len = [list(t) for t in zip(*order)]

        self.gravity_min = self.gravity_max = self.center_of_gravity()
        self.gravity_range = 0
        return self.observation()

    def center_of_gravity(self):
        weight = np.sum(self.cargo)

        if weight > 0:
            return 1 / weight * np.sum(np.nonzero(self.cargo))
        else:
            return self.space / 2

    def observation(self):
        obs = np.zeros((4, self.space), dtype=np.uint8)
        obs[3, :] = self.cargo
        obs[1, int(self.center_of_gravity())] = 1
        image[0, int(self.gravity_min)] = 255
        image[0, int(self.gravity_max)] = 255

        if len(self.containers) > 0:
            obs[2, self.containers[self.current]] = 1

        return obs[:, :, np.newaxis] * 255

    def step(self, action):
        reward = 0
        done = False
        info = {}
        if action == 0:
            self.current = max(0, self.current - 1)
        elif action == 1:
            self.current = min(len(self.containers) - 1, self.current + 1)
        elif action == 2:
            start = self.containers[self.current]
            size = self.container_len[self.current]
            self.cargo[start : start + size] = 0
            reward += 1
            self.containers.pop(self.current)
            self.container_len.pop(self.current)

            cog = self.center_of_gravity()
            if cog < self.gravity_min:
                reward -= self.gravity_penalty * (self.gravity_min - cog)
                self.gravity_min = cog
                self.gravity_range = self.gravity_max - self.gravity_min
            elif cog > self.gravity_max:
                reward -= self.gravity_penalty * (cog - self.gravity_max)
                self.gravity_max = cog
                self.gravity_range = self.gravity_max - self.gravity_min

            if len(self.containers) > 0:
                self.current = min(len(self.containers) - 1, self.current)
            else:
                reward += 2
                done = True
                info["gravity_max"] = self.gravity_max
                info["gravity_min"] = self.gravity_min
                info["gravity_range"] = self.gravity_range

        return self.observation(), reward, done, info
