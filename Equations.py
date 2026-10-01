import abc
import numpy as np
import math

class Equation():
    def __init__ (self, spatial_size,
                  total_time: float, temporal_size: float, alpha: float):
        
        self.total_time = total_time
        self.temporal_size = temporal_size
        self.alpha = alpha
        
        self.dt = self.total_time / self.temporal_size
        
        self.spatial_size = tuple(spatial_size)
        self.ndim = len(self.spatial_size)

    @abc.abstractmethod
    def initialize_equation(self):
        pass
    
    @abc.abstractmethod
    def solve(self):
        pass

class HeatEquation2D(Equation):
    def __init__(self, spatial_size,
                 total_time: float, temporal_size: float, alpha: float):
        
        super().__init__(spatial_size, total_time, temporal_size, alpha)
        self.nx, self.ny = self.spatial_size[0], self.spatial_size[1]
        
        self.dx = 1.0 / (self.nx - 1)
        self.dy = 1.0 / (self.ny - 1)
        
    def initialize_equation(self):
        u0 = np.ones((self.ny, self.nx)) * 10
        
        corners = [(0, 0), (0, self.ny-1), (self.nx-1, 0), (self.nx-1, self.ny-1)]
        for (i, j) in corners:
            u0[j, i] = np.random.uniform(-25, 25)
            
        u0[:, 0] = np.linspace(u0[0, 0], u0[-1, 0], self.ny)
        u0[:, -1] = np.linspace(u0[0, -1], u0[-1, -1], self.ny)
        u0[0, :] = np.linspace(u0[0, 0], u0[0, -1], self.nx)
        u0[-1, :] = np.linspace(u0[-1, 0], u0[-1, -1], self.nx)
        
        return u0
    
    def solve(self):
        u0 = self.initialize_equation()
        u = np.zeros((self.temporal_size, self.ny, self.nx))
        u[0] = u0

        dt_max = 0.5 / (self.alpha * (1/self.dx**2 + 1/self.dy**2))
        n_sub = math.ceil(self.dt / (0.9 * dt_max))
        h = self.dt / n_sub
        cx = self.alpha * h / self.dx**2
        cy = self.alpha * h / self.dy**2

        cur = u0.copy()
        for t in range(1, self.temporal_size):
            for _ in range(n_sub):
                next = cur.copy()
                next[1:-1, 1:-1] = (cur[1:-1, 1:-1]
                                    + cx * (cur[2:, 1:-1] - 2*cur[1:-1, 1:-1] + cur[:-2, 1:-1])
                                    + cy * (cur[1:-1, 2:] - 2*cur[1:-1, 1:-1] + cur[1:-1, :-2]))
                cur = next
            u[t] = cur
        return u
    
class HeatEquationDataset:
    def __init__(self, equation_class, num_samples: int, **kwargs):
        self.num_samples = num_samples
        self.equation_class = equation_class
        self.kwargs = ( kwargs )
        
        self.data = self._generate_data()

    def _generate_data(self):
        dataset = []
        for _ in range(self.num_samples):
            eq = self.equation_class(**self.kwargs)
            
            sol = eq.solve()
            dataset.append(sol)

        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]