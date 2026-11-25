import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Painter:
    def __init__(self):
            self.cmap = "inferno"

    def plot_solution(self, data, ax = None, title = None, vmin = None, vmax = None, cmap = None,
                        has_axis = True, has_colorbar = False, cbar_label = "Temp"):
        if ax is None:
            _, ax = plt.subplots(figsize = (6, 5))

        im = ax.imshow(data, cmap = cmap or self.cmap, origin = "lower",
                        extent = [0, 1, 0, 1], vmin = vmin, vmax = vmax)
        if title is not None:
            ax.set_title(title)

        if has_axis:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if has_colorbar:
            ax.figure.colorbar(im, ax = ax, label = cbar_label)
        return im

class Panel:
    def __init__(self, data, cmap = None, vmin = 0.0, vmax = 1.0):
        self.data = np.asarray(data)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.im = None

    def frame(self, k):
        if (self.data.ndim == 2):
            return self.data
        else:
            return self.data[k]

    def draw(self, ax, k = 0):
        self.im = Painter.plot_solution(self.frame(k), ax = ax, cmap = self.cmap,
                                        vmin = self.vmin, vmax = self.vmax, has_axis = False)

    def update(self, k):
        self.im.set_data(self.frame(k))