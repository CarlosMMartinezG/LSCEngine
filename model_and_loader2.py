# model_and_loader.py
import torch
import torch.nn as nn

# ===========================
# Modelo BiLSTM para landmarks
# ===========================

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=512, num_layers=3, num_classes=47):
        """
        input_dim: dimensiones por frame (21 landmarks * 3 coords = 63)
        hidden_dim: neuronas LSTM
        num_layers: número de capas LSTM
        num_classes: número de clases (palabras/señas)
        """
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x: tensor [batch, seq_len, 21, 3] → landmarks
        """
        B, T, V, C = x.shape
        x = x.view(B, T, V*C)  # aplanamos landmarks por frame
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # tomar output del último timestep
        out = self.fc(out)
        return out

# ===========================
# Dataset auxiliar
# ===========================
from torch.utils.data import Dataset
import pickle
import torch

class LandmarkSequenceDataset(Dataset):
    def __init__(self, pkl_file):
        """
        Carga secuencias de landmarks desde un pickle generado por preprocess.py
        """
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in data["sequences"]]
        self.labels = torch.tensor(data["labels"], dtype=torch.long)
        self.class_map = data["class_map"]
        print("[LandmarkSequenceDataset] Loaded:")
        print(f"  Total samples: {len(self.sequences)}")
        print(f"  Num classes : {len(self.class_map)}")
        print(f"  Classes     : {self.class_map}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ===========================
# Función para cargar modelo entrenado
# ===========================
def load_model(path, device="cpu", input_dim=63, hidden_dim=512, num_layers=3, num_classes=47):
    model = BiLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim,
                        num_layers=num_layers, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

