import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
from src.simulator import DegradationSimulator

class HybridDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Root directory containing class subfolders.
            transform (callable): Image transforms.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.simulator = DegradationSimulator(days=30)
        
        # Standard Classes from 'djdhairya' dataset
        self.classes = sorted(['Clean', 'Dusty', 'Bird-drop', 'Snow-Covered', 'Electrical-damage', 'Physical-Damage'])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Severity Mapping (Heuristic for "Percentage Defect")
        # Used to train the 'Severity Head'
        self.severity_map = {
            'Clean': 0.0,
            'Dusty': 0.20,
            'Bird-drop': 0.30,
            'Snow-Covered': 0.60,
            'Electrical-damage': 0.80,
            'Physical-Damage': 1.00
        }

        # Mapping visual classes to physics simulator types
        self.sim_map = {
            'Clean': 'normal',
            'Dusty': 'soiling',
            'Bird-drop': 'soiling',
            'Snow-Covered': 'soiling',
            'Electrical-damage': 'hotspot',
            'Physical-Damage': 'cellular_crack'
        }

        # Scan directory
        self.samples = []
        if os.path.exists(img_dir):
            for cls_name in self.classes:
                cls_folder = os.path.join(img_dir, cls_name)
                if os.path.isdir(cls_folder):
                    files = glob.glob(os.path.join(cls_folder, "*"))
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.samples.append((f, cls_name))
        
        # Fallback dummy data if empty
        if len(self.samples) == 0:
            print(f"Warning: No images found in {img_dir}. Using dummy data.")
            for _ in range(10):
                self.samples.append((None, 'Clean'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_name = self.samples[idx]
        
        # 1. Load Image
        if path and os.path.exists(path):
            try:
                image = Image.open(path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224))
        else:
            image = Image.new('RGB', (224, 224)) # Dummy black image
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        # 2. Simulator (Bridging Vision -> Physics)
        sim_type = self.sim_map.get(cls_name, 'normal')
        series_np = self.simulator.generate_series(sim_type, standardize=True)
        series_tensor = torch.from_numpy(series_np).float()
        
        # 3. Targets
        # Class Label
        label = self.class_to_idx.get(cls_name, 0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Severity Target (0.0 - 1.0)
        severity = self.severity_map.get(cls_name, 0.0)
        severity_tensor = torch.tensor([severity], dtype=torch.float32)
        
        # Degradation Target (Future Health)
        # We assume higher severity = lower future health
        deg_target = 1.0 - (severity * 0.8) # Heuristic: max degradation 0.2 health for worst case
        deg_tensor = torch.tensor([deg_target], dtype=torch.float32)
        
        return image, series_tensor, label_tensor, severity_tensor, deg_tensor

def get_transforms(phase='train'):
    """Standard ImageNet normalization + Resize"""
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
