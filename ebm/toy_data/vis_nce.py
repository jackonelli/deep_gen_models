import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_nce(dataset, energy, noise, device, plot_name):
    n_pts = 700
    range_lim = 4

    # construct test points
    test_grid = setup_grid(range_lim, n_pts, device)

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(12,4.3), subplot_kw={'aspect': 'equal'})
    plot_samples(dataset, axs[0], range_lim, n_pts)
    plot_noise(noise, axs[1], test_grid, n_pts)
    plot_energy(energy, axs[2], test_grid, n_pts)

    # format
    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    plt.tight_layout()

    # save
    print('Saving image to images/....')
    plt.show()
    plt.savefig(plot_name)
    plt.close()

    # move back to cuda if cpu temporarily has been used
    energy.to(device)

@torch.no_grad()
def plot_cnce(dataset, energy, cnce, device, plot_name):
    n_pts = 700
    range_lim = 4

    # construct test points
    test_grid = setup_grid(range_lim, n_pts, device)

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(12,4.3), subplot_kw={'aspect': 'equal'})
    plot_samples(dataset, axs[0], range_lim, n_pts)
    plot_cnce_samples(dataset, cnce, axs[1], range_lim, n_pts)
    plot_energy(energy, axs[2], test_grid, n_pts)

    # format
    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    plt.tight_layout()

    # save
    print('Saving image to images/....')
    plt.show()
    plt.savefig(plot_name)
    plt.close()

    # move back to cuda if cpu temporarily has been used
    energy.to(device)

def setup_grid(range_lim, n_pts, device):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x), indexing='ij')
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(device)

def plot_samples(dataset, ax, range_lim, n_pts):
    samples = dataset.cpu().numpy()
    ax.hist2d(samples[:,0], samples[:,1], range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=n_pts, cmap=plt.cm.jet)
    ax.set_title('Target samples')

def plot_cnce_samples(dataset, cnce, ax, range_lim, n_pts):
    noise_samples = cnce.add_noise(dataset).numpy()
    ax.hist2d(noise_samples[:,0], noise_samples[:,1], range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=n_pts, cmap=plt.cm.jet)
    ax.set_title('Noised samples')

def plot_energy(energy, ax, test_grid, n_pts):
    xx, yy, zz = test_grid
    # workaround, move to cpu when plotting due to test_grid being large
    log_prob = energy.cpu()(zz.cpu())
    prob = log_prob.exp().cpu()
    # plot
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('Energy density')

def plot_noise(noise, ax, test_grid, n_pts):
    xx, yy, zz = test_grid
    log_prob = noise.log_prob(zz)
    prob = log_prob.exp().cpu()
    # plot
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('Noise density')

def format_ax(ax, range_lim):
    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()