import abc
import numpy as np

class Equation():
    def __init__ (self, spatial_size,
                  total_time: float, temporal_size: int, alpha: float):
        
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
                 total_time: float, temporal_size: int, alpha: float, mode: str):
        
        super().__init__(spatial_size, total_time, temporal_size, alpha)
        self.num_x, self.num_y = self.spatial_size[0], self.spatial_size[1]
        
        self.dx = 1.0 / (self.num_x - 1)
        self.dy = 1.0 / (self.num_y - 1)
        
        self.mode = mode
        
    def initialize_equation(self):
        if self.mode == "CONST":
            T0 = np.random.uniform(0, 1)
            u0 = np.full((self.num_y, self.num_x), T0)
            u0[0, :] = 0.0
            u0[-1, :] = 0.0
            u0[:, 0] = 0.0
            u0[:, -1] = 0.0
            return u0
        else:
            u0 = np.ones((self.num_y, self.num_x)) * 10
            
            corners = [(0, 0), (0, self.num_y-1), (self.num_x-1, 0), (self.num_x-1, self.num_y-1)]

            for (i, j) in corners:
                u0[j, i] = np.random.uniform(-25, 25)
                
            u0[:, 0] = np.linspace(u0[0, 0], u0[-1, 0], self.num_y)
            u0[:, -1] = np.linspace(u0[0, -1], u0[-1, -1], self.num_y)
            u0[0, :] = np.linspace(u0[0, 0], u0[0, -1], self.num_x)
            u0[-1, :] = np.linspace(u0[-1, 0], u0[-1, -1], self.num_x)
        
        return u0
    
    def apply_boundary(self, u_next, u0):
        if self.mode == "CONST":
            u_next[0, :] = 0.0
            u_next[-1, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0
        else:
            u_next[:, 0] = u0[:, 0]
            u_next[:, -1] = u0[:, -1]
            u_next[0, :] = u0[0, :]
            u_next[-1, :] = u0[-1, :]
        return u_next
    
    def solve(self):
        u0 = self.initialize_equation()
        u = np.zeros((self.temporal_size, self.num_y, self.num_x))
        u[0] = u0

        dt_out = self.total_time / (self.temporal_size - 1)

        dt_max = 0.9 * 0.5 / (self.alpha * (1 / self.dx**2 + 1 / self.dy**2))
        n_sub = int(np.ceil(dt_out / dt_max))
        dt = dt_out / n_sub

        cx = self.alpha * dt / self.dx**2
        cy = self.alpha * dt / self.dy**2

        cur = u0.copy()
        for step in range(1, self.temporal_size):
            for _ in range(n_sub):
                nxt = cur.copy()
                nxt[1:-1, 1:-1] = (cur[1:-1, 1:-1]
                                + cx * (cur[2:, 1:-1] - 2 * cur[1:-1, 1:-1] + cur[:-2, 1:-1])
                                + cy * (cur[1:-1, 2:] - 2 * cur[1:-1, 1:-1] + cur[1:-1, :-2]))
                cur = self.apply_boundary(nxt, u0)
            u[step] = cur

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