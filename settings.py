import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from util.data import Dataset
from mpl_toolkits.mplot3d import Axes3D


class Settings:
    normalized_popularity = None

    loss_alpha: float = None
    loss_beta: float = None
    loss_scale: float = None
    loss_percentile: float = None

    metrics_alpha: float = None
    metrics_beta: float = None
    metrics_gamma: float = None
    metrics_scale: float = None
    metrics_percentile: float = None
    max_y_aux_popularity: float = None

    loss_type = 2  # 0 original, 1 CNR first version, 2 CNR second version
    k = 100
    k_trainable = False

    low_popularity_threshold = 0.05
    high_popularity_threshold = 0.25


def get_percentile(array, k):
    # print('Luciano > popularity_array:', popularity_array)
    sorted_array = np.sort(array)
    index = int(round(array.shape[0] * k / 100))
    # print('Luciano > len:', popularity_array.shape[0])
    # print('Luciano > index:', index)
    percentile = sorted_array[index]
    # print('Luciano > percentile:', percentile)
    return percentile


def set_parameters(
        normalized_popularity,

        loss_alpha,
        loss_beta,
        loss_scale,
        loss_percentile,

        metrics_alpha,
        metrics_beta,
        metrics_gamma,
        metrics_scale,
        metrics_percentile,

        loss_type,
        k,
        k_trainable,
        low_popularity_threshold,
        high_popularity_threshold
):
    Settings.normalized_popularity = normalized_popularity

    Settings.loss_alpha = loss_alpha
    Settings.loss_beta = loss_beta
    Settings.loss_scale = loss_scale
    Settings.loss_percentile = loss_percentile

    Settings.metrics_alpha = metrics_alpha
    Settings.metrics_beta = metrics_beta
    Settings.metrics_gamma = metrics_gamma
    Settings.metrics_scale = metrics_scale
    Settings.metrics_percentile = metrics_percentile

    Settings.loss_type = loss_type
    Settings.k = k
    Settings.k_trainable = k_trainable

    Settings.low_popularity_threshold = low_popularity_threshold
    Settings.high_popularity_threshold = high_popularity_threshold

    domain = np.linspace(0, 1, 1000)
    codomain = [y_aux_popularity(x) for x in domain]
    Settings.max_y_aux_popularity = max(codomain)

    print('Config parameters:', normalized_popularity,

          loss_alpha,
          loss_beta,
          loss_scale,
          loss_percentile,

          metrics_alpha,
          metrics_beta,
          metrics_gamma,
          metrics_scale,
          metrics_percentile,

          loss_type,
          k,
          sep='\n- ')


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def y_aux_popularity(x):
    f = 1 / (Settings.metrics_beta * np.sqrt(2 * np.pi))
    y = np.tanh(Settings.metrics_alpha * x) + \
        Settings.metrics_scale * f * np.exp(-1 / (2 * (Settings.metrics_beta ** 2)) * (x - Settings.metrics_percentile) ** 2)
    return y


def y_popularity(x):
    y = y_aux_popularity(x) / Settings.max_y_aux_popularity
    return y


def y_position(x, cutoff):
    y = sigmoid(-x * Settings.metrics_gamma / cutoff) + 0.5
    return y


def y_custom(popularity, position, cutoff):
    y = y_popularity(popularity) * y_position(position, cutoff)
    return y


if __name__ == "__main__":
    print("Testing settings")

    dataset = Dataset('data/pinterest.npz')

    set_parameters(
        normalized_popularity=dataset.normalized_popularity,
        loss_alpha=200,
        loss_beta=0.02,
        loss_scale=1,
        loss_percentile=get_percentile(dataset.normalized_popularity, 45),
        metrics_alpha=100,
        metrics_beta=0.03,
        metrics_gamma=5,
        metrics_scale=1 / 15,
        metrics_percentile=0.45,
        loss_type=2
    )

    print('dataset.normalized_popularity[21]:', dataset.normalized_popularity[21])

    cutoff = 5
    points = 1000

    x_1 = np.linspace(0, 1, points)
    x_2 = np.linspace(0, cutoff, points)

    y_1 = y_popularity(x_1)
    y_2 = y_position(x_2, cutoff)

    plt.figure()
    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x_1, x_2 = np.meshgrid(x_1, x_2)
    z = y_custom(x_1, x_2, cutoff)
    # print(z)
    print('max_value:', np.max(z))

    # Plot the surface.
    surf = ax.plot_surface(x_1, x_2, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=2)

    plt.show()
