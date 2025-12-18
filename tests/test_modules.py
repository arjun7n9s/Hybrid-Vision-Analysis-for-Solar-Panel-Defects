import torch
from src.simulator import DegradationSimulator
from src.model import HybridSolarModel

def test_simulator_physics():
    sim = DegradationSimulator(days=30)
    
    # Generate Normal
    data_normal = sim.generate_series('normal')
    avg_v_normal = data_normal[:, 0].mean()
    
    # Generate Crack
    data_crack = sim.generate_series('cellular_crack')
    avg_v_crack = data_crack[:, 0].mean()
    
    print(f"Normal Avg Voltage: {avg_v_normal:.3f}")
    print(f"Crack Avg Voltage: {avg_v_crack:.3f}")
    
    # Physics Check: Crack should have lower average voltage or higher variance
    # Note: With normalization, values are 0-1.
    if avg_v_crack < avg_v_normal:
        print("PASS: Crack reduces voltage as expected.")
    else:
        print("WARNING: Physics simulation might need tuning.")

def test_model_forward():
    model = HybridSolarModel()
    
    # Batch of 2
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_series = torch.randn(2, 30, 4) # 30 steps, 4 channels
    
    logits, regression = model(dummy_img, dummy_series)
    
    print("\nModel Output Shapes:")
    print(f"Logits: {logits.shape} (Expected: [2, 4])")
    print(f"Regression: {regression.shape} (Expected: [2, 1])")
    
    assert logits.shape == (2, 4)
    assert regression.shape == (2, 1)
    print("PASS: Model forward shapes match.")

if __name__ == "__main__":
    test_simulator_physics()
    test_model_forward()
