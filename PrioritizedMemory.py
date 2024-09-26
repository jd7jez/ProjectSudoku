import random
import numpy as np


class PrioritizedMemory:

    def __init__(self, capacity=500, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.memory_size = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def memorize(self, experience):
        max_priority = self.priorities.max() if self.memory_size > 0 else 1.0
        if self.memory_size < self.capacity:
            self.memory_size += 1
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.memory_size == 0:
            raise ValueError("Memory is empty, cannot sample an empty memory.")

        probs = self.priorities[:self.memory_size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.memory_size, batch_size, p=probs)
        experiences = [self.memory[i] for i in indices]

        weights = (self.memory_size * probs[indices]) ** (-1 * beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = error + 0.00000001
