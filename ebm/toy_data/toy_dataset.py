import math
import numpy as np

import torch
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

### Data
def return_dataset(dataset_name: str, n_training_samples: int, n_validation_samples:int=100, n_test_samples:int=1000):
    training_data = TensorDataset(sample_2d_data(dataset_name, n_training_samples))
    validation_data = TensorDataset(sample_2d_data(dataset_name, n_validation_samples))
    test_data = TensorDataset(sample_2d_data(dataset_name, n_test_samples))
    return training_data, validation_data, test_data

def sample_2d_data(dataset_name: str, n_samples: int) -> torch.Tensor:
    """generate samples from 2D toy distributions
    Code borrowed from https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py"""
    z = torch.randn(n_samples, 2)

    if dataset_name == '8gaussians': # 8gaussians
        scale = 4
        sq2 = 1/math.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        x = sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])
        return x
    elif dataset_name == '2spirals': # 2spirals
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    elif dataset_name == 'checkerboard': # checkerboard
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset_name == 'rings': # 'rings'
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))
    
    elif dataset_name == 'pinwheel': # pinwheel
        rng = np.random.RandomState()
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        
        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return torch.as_tensor(data, dtype=torch.float32)

    else:
        raise ValueError('Invalid dataset name.')

### Visualization
def vis_result(X_data, X_model, model, device):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.axis('equal')
    ax2.axis('equal')
    ax1.set_title('Samples')
    ax2.set_title('Energy map')
    plot_2d_samples(ax1, X_data[:][0])
    plot_2d_samples(ax1, X_model)
    plot_2d_energy_map(ax2, model, xlim=(-5, 5), ylim=(-5, 5), n_grid=100, plot_bar=True, device=device)
    plot_2d_samples(ax2, X_data[:][0])
    ax1.legend(['data samples', 'generated samples'])
    fig.tight_layout()
    plt.show()

def plot_2d_samples(ax: Axes, data: torch.Tensor):
    """Plot 2D samples.

    Args:
        data (torch.Tensor): shape (N, 2)
    """
    ax.scatter(data[:, 0], data[:, 1], s=1)

def plot_2d_samples_with_langevin_dynamics(ax: Axes, data: torch.Tensor, dynamics: torch.Tensor):
    """Plot 2D samples with Langevin dynamics.

    Args:
        data (torch.Tensor): shape (N, 2)
        dynamics (torch.Tensor): shape (N, 2)
    """
    ax.quiver(data[:, 0], data[:, 1], dynamics[:, 0], dynamics[:, 1], scale=1, scale_units='xy', angles='xy', color='m')

def plot_2d_energy_map(ax: Axes, model: torch.nn.Module, xlim: tuple, ylim: tuple, n_grid: int=100, plot_bar=False, device:str='cpu'):
    """Plot 2D energy map.

    Args:
        model (torch.nn.Module): An energy-based model
        xlim (tuple): x-axis range
        ylim (tuple): y-axis range
        n_grid (int): The number of grid points
    """
    x = torch.linspace(xlim[0], xlim[1], n_grid)
    y = torch.linspace(ylim[0], ylim[1], n_grid)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = model(torch.stack([X, Y], dim=-1).to(device).view(-1, 2)).view(n_grid, n_grid).detach().to('cpu').numpy()
    cf = ax.contourf(X, Y, Z, 100, cmap='jet')
    ax.contour(X, Y, Z, 10, colors='black', linewidths=0.5)
    
    if plot_bar:
        cbar = plt.colorbar(cf)
        cbar.ax.set_ylabel('Energy', rotation=270, labelpad=15)