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
PKL_PATH = "c:/IA_CarlosMartinez/preprocessed/sequences_all.pkl"
BATCH_SIZE = 16
LR = 1e-2
EPOCHS = 100
HIDDEN_DIM = 128
NUM_LAYERS = 1
EARLY_STOPPING_PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[DEVICE]", DEVICE)


# ======================================
# DATASET
# ======================================

# dataset = LandmarkSequenceDataset(PKL_PATH)
# num_classes = len(dataset.class_map)

# # dividir en train / validation (estratificado)
# train_idx, val_idx = train_test_split(
#     list(range(len(dataset))),
#     test_size=0.2,
#     shuffle=True,
#     stratify=dataset.labels
# )

# train_dataset = Subset(dataset, train_idx)
# val_dataset = Subset(dataset, val_idx)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=1, num_classes=47):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
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
    # # early stopping
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     patience_counter = 0
    #     torch.save(model.state_dict(), "best_biLSTM_model.pth")
    #     print("    -> Nuevo mejor modelo guardado")
    # else:
    #     patience_counter += 1
    #     if patience_counter >= EARLY_STOPPING_PATIENCE:
    #         print("    -> Early stopping activado")
    #         break


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



#========================

# # train_final.py
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import numpy as np

# from dataset_lsc_landmarks import LandmarkSequenceDataset

# # CONFIG
# PKL_PATH = "preprocessed/sequences_all.pkl"
# BATCH_SIZE = 16
# LR = 1e-2
# EPOCHS = 60
# HIDDEN_DIM = 128
# NUM_LAYERS = 1
# EARLY_STOPPING_PATIENCE = 8
# WEIGHT_DECAY = 1e-5
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("[DEVICE]", DEVICE)

# # Dataset base (no augment) for splits
# base_dataset = LandmarkSequenceDataset(PKL_PATH, augment=False)
# num_classes = len(base_dataset.class_map)

# # stratified split
# indices = list(range(len(base_dataset)))
# train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=True, stratify=base_dataset.labels, random_state=42)

# # create datasets where train has augment=True and val augment=False
# train_dataset = LandmarkSequenceDataset(PKL_PATH, augment=True)
# val_dataset   = LandmarkSequenceDataset(PKL_PATH, augment=False)

# # subset sequences/labels by indices (preserve augment flag)
# train_dataset.sequences = [base_dataset.sequences[i] for i in train_idx]
# train_dataset.labels    = [base_dataset.labels[i]    for i in train_idx]

# val_dataset.sequences = [base_dataset.sequences[i] for i in val_idx]
# val_dataset.labels    = [base_dataset.labels[i]    for i in val_idx]

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
# val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# # Model: BiLSTM improved
# class BiLSTMModel(nn.Module):
#     def __init__(self, input_dim=63, hidden_dim=128, num_layers=1, num_classes=47, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
#                             batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim*2, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, num_classes)
#         )
#     def forward(self, x):
#         B,T,V,C = x.shape
#         x = x.view(B,T,V*C)  # (B,6,63)
#         out, _ = self.lstm(x)  # (B,T,2*H)
#         # use mean pooling across time (more robust than last timestep)
#         out = out.mean(dim=1)
#         out = self.dropout(out)
#         return self.fc(out)

# model = BiLSTMModel(input_dim=63, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=num_classes, dropout=0.3).to(DEVICE)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# # training helpers
# def compute_accuracy(loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for seqs, labels in loader:
#             seqs = seqs.to(DEVICE)
#             labels = labels.to(DEVICE)
#             out = model(seqs)
#             preds = out.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     return correct/total if total>0 else 0.0

# train_acc_hist = []
# val_acc_hist = []
# best_val = 0.0
# patience = 0

# for epoch in range(1, EPOCHS+1):
#     model.train()
#     running_loss = 0.0
#     for seqs, labels in train_loader:
#         seqs = seqs.to(DEVICE)
#         labels = labels.to(DEVICE)
#         optimizer.zero_grad()
#         out = model(seqs)
#         loss = criterion(out, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * seqs.size(0)
#     avg_loss = running_loss / len(train_dataset)

#     train_acc = compute_accuracy(train_loader)
#     val_acc = compute_accuracy(val_loader)
#     train_acc_hist.append(train_acc)
#     val_acc_hist.append(val_acc)

#     print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

#   # early stopping + save best
#     if val_acc > best_val:
#         best_val = val_acc
#         patience = 0
#         torch.save(model.state_dict(), "best_biLSTM_model.pth")
#         print("  -> Best model saved.")
#     else:
#         patience += 1
#         if patience >= EARLY_STOPPING_PATIENCE:
#             print("  -> Early stopping triggered.")
#             break

# print("Training finished. Best val acc:", best_val)

# # plot accuracies
# plt.figure(figsize=(8,4))
# plt.plot(train_acc_hist, label="Train Accuracy")
# plt.plot(val_acc_hist, label="Val Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid()
# plt.savefig("accuracy_plot.png")
# plt.show()
# print("Saved accuracy_plot.png")
