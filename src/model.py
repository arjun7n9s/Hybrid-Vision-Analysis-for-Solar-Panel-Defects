import torch
import torch.nn as nn
import torchvision.models as models

class HybridSolarModel(nn.Module):
    def __init__(self, num_classes=6, input_channels=4, lstm_hidden=128, lstm_layers=2):
        """
        Hybrid Model: VGG16 for Images + LSTM for Time-Series.
        Matching 'djdhairya' architecture which uses VGG16 Transfer Learning.
        
        Args:
            num_classes (int): Number of defect classes.
            input_channels (int): Number of features in time-series (V, I, T, G)
            lstm_hidden (int): Hidden size for LSTM
            lstm_layers (int): Number of LSTM layers
        """
        super(HybridSolarModel, self).__init__()
        
        # --- Visual Branch (VGG16) ---
        # Using VGG16 as per djdhairya's notebook approach
        # VGG16 output features: 512 x 7 x 7
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.visual_extractor = vgg.features
        
        # Adaptive pooling to ensure fixed output size regardless of input
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Flatten VGG output: 512 * 7 * 7 = 25088
        visual_dim = 512 * 7 * 7 
        
        # --- Time-Series Branch (LSTM) ---
        # Input: (Batch, Seq_Len, Features=4) -> Voltage, Current, Temp, Irradiance
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.lstm_fc = nn.Linear(lstm_hidden, 128) 
        
        # --- Fusion & Heads ---
        fusion_dim = visual_dim + 128
        
        # Intermediate FC layers (VGG style classifier adaptation)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Head A: Defect Classification (6 Classes)
        self.class_head = nn.Linear(4096, num_classes)
        
        # Head B: Degradation Prediction
        self.deg_head = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

        # Head C: Severity Prediction
        self.severity_head = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, image, time_series):
        # Visual Path
        vis_feat = self.visual_extractor(image) # (B, 512, 7, 7)
        vis_feat = self.avgpool(vis_feat)
        vis_feat = torch.flatten(vis_feat, 1) # (B, 25088)
        
        # Time-Series Path
        lstm_out, (hn, cn) = self.lstm(time_series)
        ts_feat = self.lstm_fc(hn[-1]) # (B, 128)
        
        # Fusion
        combined = torch.cat((vis_feat, ts_feat), dim=1) 
        embedding = self.fusion_fc(combined) 
        
        # Outputs
        class_logits = self.class_head(embedding)
        deg_pred = self.deg_head(embedding)
        severity_pred = self.severity_head(embedding)
        
        return class_logits, deg_pred, severity_pred
