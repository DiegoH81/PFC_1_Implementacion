import numpy as np
import matplotlib.pyplot as plt

class HeatEquationPainter:
    def __init__(self):
        self.cmap = "inferno"

    def plot_solution(self, simulation_data, time_step, title):
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        frame_data = simulation_data[time_step]
        im = ax.imshow(frame_data, cmap=self.cmap, origin="lower", aspect="auto")
        
        fig.colorbar(im, ax=ax, label="Temp")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        plt.tight_layout()
        plt.show()