import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256, num_layers=2):
        super(CNNLSTM, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the last fully connected layer to use it as a feature extractor
        # ResNet50 final layer is 'fc'
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        
        # Freeze CNN parameters (optional, can be unfreezed for fine-tuning)
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # LSTM layer
        # ResNet50 output dim is 2048
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape for CNN: (batch_size * seq_len, C, H, W)
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features
        # Output shape: (batch_size * seq_len, 2048, 1, 1)
        features = self.cnn(c_in)
        
        # Reshape for LSTM: (batch_size, seq_len, 2048)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM forward
        # out shape: (batch_size, seq_len, hidden_dim)
        # _ (hidden state, cell state) are ignored
        lstm_out, _ = self.lstm(features)
        
        # Take the output of the last time step for classification
        # shape: (batch_size, hidden_dim)
        last_out = lstm_out[:, -1, :]
        
        # Classification
        # shape: (batch_size, num_classes)
        output = self.fc(last_out)
        
        return output

