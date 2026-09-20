import abc
import numpy as np

class Equation():
    def __init__ (self, spatial_size, 
                  temporal_size, n_steps,
                  alpha):
        
        self.temporal_size = temporal_size
        self.alpha = alpha
        
        self.spatial_size = tuple(spatial_size)
        self.ndim = len(self.spatial_size)

    @abc.abstractmethod
    def initialize_equation(self):
        pass
    
    @abc.abstractmethod
    def solve(self):
        pass


class HeatEquation2D(Equation):
    def __init__(self, spatial_size, temporal_size, n_steps, alpha):
        super().__init__(spatial_size, temporal_size, n_steps, alpha)
        self.nx = self.ny = self.spatial_size[0], self.spatial_size[1]
        self.dx = 1.0 / (self.nx - 1)
        self.dy = 1.0 / (self.ny - 1)
        
    def initialize_equation(self):
        u0 = np.ones((self.ny, self.nx)) * 10
        
        return u0
    
    def solve(self):
        u0 = self.initialize_conditions()
        u = np.zeros((self.temporal_size, self.ny, self.nx))
        u[0] = u0
        
        return u