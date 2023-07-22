import torch
import torch.nn as nn

class StoppingModel(nn.Module):

    def __init__(self, input_dim, dropout=0.05):
        super(StoppingModel, self).__init__()
        self.dropout = dropout



        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  nn.BatchNorm1d(input_dim // 2), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(input_dim // 2, input_dim // 4), nn.BatchNorm1d(input_dim // 4), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(input_dim // 4,  input_dim // 8), nn.BatchNorm1d(input_dim // 8), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(input_dim // 8, 2)
        )

    def forward(self, query, img):
        
        x = torch.cat((query, img), dim=-1).float()
        return self.mlp(x)
        



