import numpy as np
import torch

class DegradationSimulator:
    """
    Simulates standardized time-series data for solar panels based on defect types.
    Generates Voltage (V), Current (I), Temperature (T), and Irradiance (G) series.
    """
    def __init__(self, days=30, steps_per_day=1):
        self.days = days
        self.steps = days * steps_per_day
        
        # Nominal Panel Specs (e.g., standard 300W panel)
        self.nominal_voc = 40.0  # Open Circuit Voltage
        self.nominal_isc = 10.0  # Short Circuit Current
        self.nominal_temp = 25.0
        self.nominal_irr = 1000.0

    def get_base_conditions(self):
        """Generates base environmental conditions (Irradiance/Temp) with seasonality/noise."""
        # Simple simulation: Irradiance fluctuates, Temp follows Irradiance
        time = np.linspace(0, self.days, self.steps)
        
        # Irradiance: Base sine wave + random cloud cover
        irradiance = 800 + 200 * np.sin(time * 0.5) 
        # Add random drops for clouds
        cloud_mask = np.random.choice([1, 0.8, 0.4], size=self.steps, p=[0.8, 0.15, 0.05])
        irradiance *= cloud_mask
        
        # Temperature: Correlated with irradiance + baseline
        temperature = 20 + (irradiance / 1000.0) * 30 + np.random.normal(0, 2, self.steps)
        
        return irradiance, temperature

    def generate_series(self, defect_type, standardize=True):
        """
        Generates (V, I, T, G) time series.
        
        Args:
            defect_type (str): 'normal', 'cellular_crack', 'soiling', 'hotspot'
            standardize (bool): If True, scales outputs to 0-1 range.
            
        Returns:
            np.ndarray: Shape (steps, 4) -> [Voltage, Current, Temp, Irradiance]
        """
        irr, temp = self.get_base_conditions()
        
        # Base Performance Model (simplified diode equation logic)
        # V decreases with Temp, I increases slightly with Temp, I proportional to Irr
        voltage = self.nominal_voc * (1 - 0.005 * (temp - 25))
        current = self.nominal_isc * (irr / 1000.0)
        
        # Apply Defect Physics
        if defect_type == 'cellular_crack':
            # Cracks cause intermittent voltage drops and increased resistance (lower V)
            # Progressive degradation factor
            degradation = np.linspace(1.0, 0.90, self.steps) # drops to 90%
            noise = np.random.normal(0, 2.0, self.steps) # High Volatility
            voltage = voltage * degradation + noise
            
        elif defect_type == 'soiling':
            # Soiling blocks light -> Current drops gradually
            # Resets randomly (simulating rain or cleaning)
            dust_factor = np.linspace(1.0, 0.7, self.steps) 
            # Simulate a rain event restoring efficiency halfway
            if np.random.random() > 0.5:
                rain_day = int(self.steps * np.random.uniform(0.3, 0.7))
                dust_factor[rain_day:] = np.linspace(1.0, 0.8, self.steps - rain_day)
            
            current = current * dust_factor
            
        elif defect_type == 'hotspot':
            # Hotspots: High temp local spikes (not fully captured in global panel temp)
            # But results in sharp voltage drops due to bypass diode activation
            drop_mask = np.random.choice([1.0, 0.6], size=self.steps, p=[0.9, 0.1])
            voltage *= drop_mask
            
        elif defect_type == 'normal_degradation' or defect_type == 'normal':
            # Standard very slow aging (negligible in 30 days)
            pass
            
        else: 
            # Default to normal if unknown
            pass

        # Add Gaussian Sensor Noise to all
        voltage += np.random.normal(0, 0.5, self.steps)
        current += np.random.normal(0, 0.1, self.steps)

        # Clip negative values
        voltage = np.maximum(voltage, 0)
        current = np.maximum(current, 0)

        data = np.stack([voltage, current, temp, irr], axis=1)

        if standardize:
            # Scalers based on maximum expected values (for safety 1.2x nominal)
            # V: 60V, I: 15A, T: 100C, G: 1200 W/m2
            max_vals = np.array([60.0, 15.0, 100.0, 1200.0])
            data = data / max_vals
            
        return data.astype(np.float32)

if __name__ == "__main__":
    # Quick Test
    sim = DegradationSimulator()
    data = sim.generate_series('cellular_crack')
    print("Generated shape:", data.shape)
    print("First 5 steps:\n", data[:5])
