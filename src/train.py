import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import HybridDataset, get_transforms
from src.model import HybridSolarModel
import os

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    LR = 1e-4 # Lower LR for VGG fine-tuning
    EPOCHS = 3 # Reduced for CPU speed, enough for demo
    
    # 1. Setup Data
    # Pointing to the consolidated dataset location in src
    # data_dir = "src/data/Faulty_solar_panel" 
    # Using relative path for better portability
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'Faulty_solar_panel')
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset not found at {data_dir}")
        return

    full_dataset = HybridDataset(data_dir, transform=get_transforms('train'))
    
    # Simple split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset Size: {len(full_dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Classes: {full_dataset.classes}")

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = HybridSolarModel(num_classes=6).to(device)
    
    # Save IMMEDIATELY to fix inference crash (Architecture mismatch)
    torch.save(model.state_dict(), "hybrid_solar_model.pth")
    print("Model initialized and weights saved (Initial VGG16 weights).")
    
    # 3. Setup Optimizers
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Decay LR every 5 epochs
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    # 4. Training Loop
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (imgs, series, labels, sev_targets, deg_targets) in enumerate(train_loader):
            imgs, series = imgs.to(device), series.to(device)
            labels = labels.to(device)
            sev_targets = sev_targets.to(device)
            deg_targets = deg_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            cls_logits, deg_pred, sev_pred = model(imgs, series)
            
            # Loss Calculation
            loss_cls = criterion_cls(cls_logits, labels)
            loss_deg = criterion_reg(deg_pred, deg_targets)
            loss_sev = criterion_reg(sev_pred, sev_targets)
            
            # Multi-task Weighted Loss
            loss = loss_cls + 10.0 * loss_deg + 10.0 * loss_sev
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save model every epoch (or logic for best)
        torch.save(model.state_dict(), "hybrid_solar_model.pth")

    print("Training Complete. Model saved to hybrid_solar_model.pth")

if __name__ == "__main__":
    train()
