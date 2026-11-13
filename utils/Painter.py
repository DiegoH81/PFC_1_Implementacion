import numpy as np
import matplotlib.pyplot as plt

class HeatEquationPainter:
    def __init__(self):
        self.cmap = "inferno"

    def plot_solution(self, simulation_data, time_step, title):
        
        fig, ax = plt.subplots(figsize = (6, 5))
        
        frame_data = simulation_data[time_step]
        im = ax.imshow(frame_data, cmap = self.cmap, origin = "lower", aspect = "auto")
        
        fig.colorbar(im, ax=ax, label="Temp")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        plt.tight_layout()
        plt.show()

    def _finish(self, fig, save_path, show):
        if save_path is not None:
            fig.savefig(f"{save_path}.png", dpi = 300, bbox_inches = "tight")
            fig.savefig(f"{save_path}.pdf", bbox_inches = "tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_comparison_grid(self, fields, row_labels, col_titles,
                             vmin = 0.0, vmax = 1.0, cbar_label = "Temp",
                             save_path = None, show = False):
        """fields[i][j]: campo 2D (y, x) de la fila i y la columna j."""
        n_rows, n_cols = len(fields), len(fields[0])
        fig, axes = plt.subplots(n_rows, n_cols, squeeze = False,
                                 figsize = (1.6 * n_cols + 1.2, 1.6 * n_rows + 0.9),
                                 gridspec_kw = {"wspace": 0.06, "hspace": 0.06})

        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                im = ax.imshow(fields[i][j], cmap = self.cmap, origin = "lower",
                               extent = [0, 1, 0, 1], vmin = vmin, vmax = vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(col_titles[j], fontsize = 10)
                if j == 0:
                    ax.set_ylabel(row_labels[i], fontsize = 11)

        fig.colorbar(im, ax = axes, fraction = 0.02, pad = 0.015, label = cbar_label)
        self._finish(fig, save_path, show)

    def plot_profiles(self, x, curves, titles, labels, styles = None,
                      xlabel = "X", ylabel = "Temp", n_cols = 4,
                      save_path = None, show = False):
        """curves[k][c]: valores de la curva c en el panel k (sobre x)."""
        n_panels = len(curves)
        n_rows = int(np.ceil(n_panels / n_cols))
        styles = styles or [{}] * len(labels)

        fig, axes = plt.subplots(n_rows, n_cols, figsize = (12, 2.8 * n_rows), sharey = True)
        axes = np.atleast_1d(axes).ravel()

        for k in range(n_panels):
            ax = axes[k]
            for c, label in enumerate(labels):
                ax.plot(x, curves[k][c], label = label, **styles[c])
            ax.set_title(titles[k], fontsize = 10)
            ax.set_xlabel(xlabel)
            ax.grid(alpha = 0.3)
            if k % n_cols == 0:
                ax.set_ylabel(ylabel)

        for k in range(n_panels, len(axes)):
            axes[k].axis("off")

        handles, lbls = axes[0].get_legend_handles_labels()
        if n_panels < len(axes):
            axes[-1].legend(handles, lbls, loc = "center", fontsize = 11, frameon = False)
        else:
            fig.legend(handles, lbls, loc = "lower center", ncol = len(labels), frameon = False)

        fig.tight_layout()
        self._finish(fig, save_path, show)
