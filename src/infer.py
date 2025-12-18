import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import sys
import sys
import os

# Add project root to path to allow 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our definition
from src.model import HybridSolarModel
from src.simulator import DegradationSimulator

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import sys
import sys
import os

# Add project root to path to allow 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our definition
from src.model import HybridSolarModel
from src.simulator import DegradationSimulator

def predict_image(image_path, model_path='hybrid_solar_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path} to {device}...")
    
    # 1. Load Model
    model = HybridSolarModel(num_classes=6).to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Make sure the model was trained with the current architecture (6 classes + severity head).")
            return
    else:
        print("Error: Model weights not found. Please train first.")
        return

    model.eval()
    
    # 2. Prepare Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        raw_img = Image.open(image_path).convert('RGB')
        input_img = transform(raw_img).unsqueeze(0).to(device) # [1, 3, 224, 224]
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    # 3. Prepare Time-Series (Imputation)
    # Since we only have an image, we use a "Reference" time series (Normal conditions)
    # This enables the model to run, but the degradation prediction will be a baseline estimate.
    # In a real app, users would upload CSV data alongside the image.
    print("Generating reference time-series for inference context...")
    sim = DegradationSimulator(days=30)
    ref_series = sim.generate_series('normal', standardize=True) 
    input_series = torch.from_numpy(ref_series).float().unsqueeze(0).to(device) # [1, 30, 4]
    
    # 4. Inference
    with torch.no_grad():
        logits, deg_pred, sev_pred = model(input_img, input_series)
        probs = torch.softmax(logits, dim=1)
    
    # 5. Interpret
    # Must match the sorted order in dataset.py
    classes = sorted(['Clean', 'Dusty', 'Bird-drop', 'Snow-Covered', 'Electrical-damage', 'Physical-Damage'])
    
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()
    
    # Regression Outputs
    severity_score = sev_pred.item() * 100
    remaining_health = deg_pred.item() * 100
    
    print("\n" + "=" * 40)
    print(f" HYBRID MODEL ANALYSIS REPORT")
    print("=" * 40)
    print(f"Image:              {os.path.basename(image_path)}")
    print("-" * 40)
    print(f"Detected Condition: {classes[pred_idx]}")
    print(f"Classification Conf: {confidence:.1%}")
    print("-" * 40)
    print(f"Defect Severity:    {severity_score:.1f}% (Estimated Extent)")
    print(f"Predicted Health:   {remaining_health:.1f}% (Performance after 30 days)")
    print("=" * 40)
    
    if classes[pred_idx] == 'Clean':
        print("STATUS: OPERATIONAL")
    else:
        print(f"STATUS: MAINTENANCE REQUIRED ({classes[pred_idx]})")
        if severity_score > 50:
            print("URGENCY: HIGH - Defect Severity Critical")
        elif severity_score > 20:
            print("URGENCY: MODERATE - Monitor or Schedule Cleaning")
        else:
            print("URGENCY: LOW - Routine Check suggested")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
