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
                 total_time: float, temporal_size: int, alpha: float):
        
        super().__init__(spatial_size, total_time, temporal_size, alpha)
        self.num_x, self.num_y = self.spatial_size[0], self.spatial_size[1]
        
        self.dx = 1.0 / (self.num_x - 1)
        self.dy = 1.0 / (self.num_y - 1)
        
    def initialize_equation(self):
        u0 = np.ones((self.num_y, self.num_x)) * 10
        
        corners = [(0, 0), (0, self.num_y-1), (self.num_x-1, 0), (self.num_x-1, self.num_y-1)]

        for (i, j) in corners:
            u0[j, i] = np.random.uniform(-25, 25)
            
        u0[:, 0] = np.linspace(u0[0, 0], u0[-1, 0], self.num_y)
        u0[:, -1] = np.linspace(u0[0, -1], u0[-1, -1], self.num_y)
        u0[0, :] = np.linspace(u0[0, 0], u0[0, -1], self.num_x)
        u0[-1, :] = np.linspace(u0[-1, 0], u0[-1, -1], self.num_x)
        
        return u0
    
    def solve(self):
        u0 = self.initialize_equation()
        u = np.zeros((self.temporal_size, self.num_y, self.num_x))
        u[0] = u0

        for t in range(1, self.temporal_size):
            u_prev = u[t-1]
            u_next = u_prev.copy()
            
            u_next[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + self.alpha * self.dt / self.dx**2 * (
                                 u_prev[2:, 1:-1] - 2*u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1]
                                 ) + self.alpha * self.dt / self.dy**2 * (u_prev[1:-1, 2:] - 2*u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2] )
            
            u_next[:, 0] = u0[:, 0]
            u_next[:, -1] = u0[:, -1]
            u_next[0, :] = u0[0, :]
            u_next[-1, :] = u0[-1, :]
            
            u[t] = u_next
            
        return u

class HeatEquation2DConst(Equation):
    def __init__(self, spatial_size, total_time, temporal_size, alpha, n_modes = 500):
        super().__init__(spatial_size, total_time, temporal_size, alpha)
        self.num_x, self.num_y = self.spatial_size
        self.n_modes = n_modes

    def initialize_equation(self):
        self.T0 = np.random.uniform(0, 1)
        u0 = np.full((self.num_x, self.num_y), self.T0)
        u0[0, :] = u0[-1, :] = 0.0
        u0[:, 0] = u0[:, -1] = 0.0
        return u0

    def _X(self, s, t):
        m = np.arange(1, 2 * self.n_modes, 2)
        decay = np.exp(-self.alpha * np.pi**2 * np.outer(t, m**2))
        return (4.0 / (np.pi * m)) * decay @ np.sin(np.pi * np.outer(s, m)).T

    def solve(self):
        u0 = self.initialize_equation()
        t = np.linspace(0, self.total_time, self.temporal_size)
        x = np.linspace(0, 1, self.num_x)
        y = np.linspace(0, 1, self.num_y)
        u = self.T0 * np.einsum('tx,ty->txy', self._X(x, t), self._X(y, t))
        u[0] = u0
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