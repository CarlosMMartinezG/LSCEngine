import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataset_lsc_landmarks import LandmarkSequenceDataset


# ======================================
# CONFIGURACIÓN
# ======================================

# ================ HIPERPARÁMETROS ÓPTIMOS ================ obtained via hyper.py
#'lr': 0.001, 'hidden_dim': 512, 
# 'num_layers': 3, 'batch_size': 32, 
# 'accuracy': 0.803051525395875
# =====================================================


PKL_PATH = "c:/IA_CarlosMartinez/preprocessed/sequences_all.pkl"
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 100
HIDDEN_DIM = 512
NUM_LAYERS = 3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[DEVICE]", DEVICE)


# ======================================
# DATASET
# ======================================


# Dataset base (sin augment)
base_dataset = LandmarkSequenceDataset(PKL_PATH, augment=False)
num_classes = len(base_dataset.class_map)

train_idx, val_idx = train_test_split(
    list(range(len(base_dataset))),
    test_size=0.2,
    shuffle=True,
    stratify=base_dataset.labels
)

# Dataset con augment SOLO para entrenamiento
train_dataset = LandmarkSequenceDataset(PKL_PATH, augment=True)
val_dataset   = LandmarkSequenceDataset(PKL_PATH, augment=False)

# Reemplazar secuencias según índices
train_dataset.sequences = [base_dataset.sequences[i] for i in train_idx]
train_dataset.labels    = [base_dataset.labels[i]    for i in train_idx]

val_dataset.sequences = [base_dataset.sequences[i] for i in val_idx]
val_dataset.labels    = [base_dataset.labels[i]    for i in val_idx]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


# ======================================
# MODELO BiLSTM
# ======================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=512, num_layers=3, num_classes=47):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, 6, 21, 3)
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)  # reshape a (B,6,63)
        out, _ = self.lstm(x)
        out = out[:, -1, :]     # último timestep
        return self.fc(out)


model = BiLSTMModel(
    input_dim=63,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes
).to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ======================================
# FUNCIONES DE EVALUACIÓN
# ======================================
def compute_accuracy(loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ======================================
# TRAINING LOOP con EARLY STOPPING
# ======================================
train_acc_hist = []
val_acc_hist = []
best_val_acc = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)

    # accuracy de entrenamiento y validación
    train_acc = compute_accuracy(train_loader)
    val_acc = compute_accuracy(val_loader)

    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    print(f"Epoch {epoch}/{EPOCHS} | Loss: {running_loss/len(train_dataset):.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    torch.save(model.state_dict(), "best_biLSTM_model.pth")


print("\nEntrenamiento finalizado. Mejor Val Accuracy:", best_val_acc)


# ======================================
# GRAFICAR ACCURACY
# ======================================
plt.figure(figsize=(8,4))
plt.plot(train_acc_hist, label="Accuracy Entrenamiento")
plt.plot(val_acc_hist, label="Accuracy Validación")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Accuracy de entrenamiento vs validación")
plt.legend()
plt.grid()
plt.savefig("accuracy_plot.png")
plt.show()

print("Gráfica guardada como accuracy_plot.png")


