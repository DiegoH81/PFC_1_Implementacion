import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Painter:
    cmap = "inferno"

    @staticmethod
    def plot_solution(data, ax = None, title = None, vmin = None, vmax = None, cmap = None,
                        has_axis = True, has_colorbar = False, cbar_label = "Temp"):
        if ax is None:
            _, ax = plt.subplots(figsize = (6, 5))

        im = ax.imshow(data, cmap = cmap or Painter.cmap, origin = "lower",
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
        
        
    
class Grid:
    def __init__(self, num_rows, num_cols,
                 row_labels, col_labels, color_bar_label,title = None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.title = title
        self.color_bar_label = color_bar_label
        
        self.panels = []
        
        for i in range(self.num_rows):
            self.panels.append([])
            for j in range(self.num_cols):
                self.panels[i].append([])
    
    def set_panel(self, in_row, in_col, in_panel):
        self.panels[in_row][in_col] = in_panel
    
    
    def build(self, n_instant = 0, padding = 0.9):
        fig, axes = plt.subplots(self.num_rows, self.num_cols, squeeze = False,
                                 figsize = (1.6 * self.num_cols + 1.2, 1.6 * self.num_rows + padding),
                                 gridspec_kw = {"wspace": 0.06, "hspace": 0.06})
        
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.panels[i][j].draw(axes[i, j], n_instant)
                
                if i == 0 and self.col_labels is not None:
                    axes[i, j].set_title(self.col_labels[j], fontsize = 10)
                if j == 0 and self.row_labels is not None:
                    axes[i, j].set_ylabel(self.row_labels[i], fontsize = 11)
        
        fig.colorbar(self.panels[-1][-1].im, ax = axes,
                     fraction = 0.02, pad = 0.015,
                     label = self.color_bar_label)
        
        if self.title is not None:
            self.suptitle = fig.suptitle(self.title, fontsize = 12)
        return fig
        
        
    def save(self, path, n_instant = 0, dpi = 300):
        fig = self.build(n_instant)
        fig.savefig(f"{path}.png", dpi = dpi, bbox_inches = "tight")
        plt.close(fig)
        
