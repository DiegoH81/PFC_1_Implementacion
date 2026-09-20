import Equations as eq

if __name__ == "__main__":
    dataset = eq.HeatEquationDataset( equation_class = eq.HeatEquation2D,
                                     num_samples = 10, spatial_size=(20, 20),
                                     total_time=1.0, temporal_size=50, alpha=0.5 )
    
    