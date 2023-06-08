import random
import numpy as np
import torch
import scipy.stats
from util.transforms import build_se3_transform


def low_variance_resampling_torch(particles, weights, num_samples=None):
    """
    Low variance resampling algorithm used in particle filters.

    :param particles: tensor, shape (N, 4, 4), the set of particles
    :param weights: tensor, shape (N,), the weights of particles
    :param num_samples: int, optional, the number of particles to sample (default is None, which samples N particles)
    :return: tensor, shape (num_samples, D), the resampled particles
    """
    N = weights.size(0)  # Number of particles
    if num_samples is None:
        num_samples = N

    resampled_particles = torch.zeros((num_samples, 4, 4))

    # Calculate the cumulative sum of the weights
    cum_weights = torch.cumsum(weights, dim=0)

    # Generate a random starting point, uniformly distributed in [0, 1/num_samples]
    start = torch.rand(1).item() / num_samples

    # Calculate equally spaced pointers in the range of [start, start + 1]
    pointers = torch.arange(start, 1 + start, 1 / num_samples)

    # Perform the low variance resampling
    i, j = 0, 0
    while i < num_samples:
        while cum_weights[j] < pointers[i]:
            j += 1
        if j >= len(cum_weights):
            break
        resampled_particles[i] = particles[j]
        i += 1

    return resampled_particles


class ParticleFilter:
    """
    A particle filter for localization on a XYYawFeatGrid
    Each particle is presented by (x, y, yaw) and a weight
    """

    def __init__(
        self, n_particles: int, resample_rand_ratio=0.025
    ) -> None:
        """
        n_particles: number of particles
        grid: the grid map
        resample_rand_ratio: the ratio of particles uniformly resampled over the grid in each resampling step
        """
        self.n_particles = n_particles
        self.particles = torch.Tensor()  # (N, 3) #x, y, yaw
        self.weights = torch.Tensor()  # (N,)
        self.resample_rand_ratio = resample_rand_ratio

        self.init_particles()

    def random_particles(self, n: int) -> torch.Tensor:
        torch.manual_seed(0)
        x = torch.randint(0,200, (n,1)) # Random values bw 0 and 50
        # y = torch.randint(-10,10, (n,1)) # Random values bw 0 and 50
        y = torch.randint(-100,100, (n,1)) # Random values bw 0 and 50
        yaw = torch.zeros(n, 1) # Assuming no yaw error
        particles = []
        
        for i in range(n):
            particle_tf = build_se3_transform([x[i,0], y[i,0], 0, 0, 0, yaw[i,0]])
            particle_tf = torch.tensor(particle_tf)
            particles.append(particle_tf)
        
        particles = torch.stack(particles)
        return particles

    def init_particles(self) -> None:
        """
        Randomly initialize the particles on the map, with uniform weights
        """
        self.particles = self.random_particles(self.n_particles)

    def motion_update(self, motion_model) -> None:
        """
        Update the particles according to the motion model
        """
        self.particles = motion_model(self.particles)

    def observation_update(self, obs_model) -> None:
        """
        Update the weights according to the observation model
        """
        self.weights = obs_model(self.particles)

    def resample(self) -> None:
        """
        Resample the particles according to the weights
        """
        n_uniform = int(self.n_particles * self.resample_rand_ratio)
        n_resample = self.n_particles - n_uniform
        particles = []
        
        # TODO: Verify is this is required
        # # unifomrly resample some particles
        # if n_uniform > 0:
        #     particles.append(self.random_particles(n_uniform))

        # resample the rest particles from the existing ones
        particles.append(
            low_variance_resampling_torch(self.particles, self.weights, n_resample)
        )

        self.particles = torch.cat(particles, dim=0)
        # self.particles = low_variance_resampling_torch(self.particles, self.weights)

    def get_kde_estimate(self) -> torch.Tensor:
        """
        Get the particle with max KDE density as the estimate
        """
        kde = scipy.stats.gaussian_kde(self.particles.T, weights=self.weights)
        density = kde(self.particles.T)
        return self.particles[density.argmax()]