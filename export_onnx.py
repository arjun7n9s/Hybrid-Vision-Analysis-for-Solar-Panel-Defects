import torch
import torch.onnx
import onnx
from src.model import HybridSolarModel

def export_to_onnx():
    # 1. Load Model
    model = HybridSolarModel()
    try:
        model.load_state_dict(torch.load("hybrid_solar_model.pth"))
        print("Loaded trained weights.")
    except FileNotFoundError:
        print("Warning: Weights not found, exporting initialized model.")
    
    model.eval()
    
    # 2. Define Dummy Inputs (Matching the Standard Schema)
    # Visual: [Batch, 3, 224, 224] - Normalized Float32
    dummy_img = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # Time-Series: [Batch, 30, 4] - Normalized Float32 (V, I, T, G)
    dummy_series = torch.randn(1, 30, 4, requires_grad=True)
    
    # 3. Export
    output_path = "hybrid_solar_model.onnx"
    torch.onnx.export(
        model,
        (dummy_img, dummy_series),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_image', 'input_series'],
        output_names=['class_logits', 'degradation_prediction'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'input_series': {0: 'batch_size'},
            'class_logits': {0: 'batch_size'},
            'degradation_prediction': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    # 4. Verification
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX integrity check passed.")

if __name__ == "__main__":
    export_to_onnx()
