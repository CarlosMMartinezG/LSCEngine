import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_lsc_landmarks import LandmarkSequenceDataset


# ======================================
# CONFIGURACIÓN
# ======================================

# ================ HIPERPARÁMETROS ÓPTIMOS ================ obtained via hyper.py
#'lr': 0.001, 'hidden_dim': 512, 'num_layers': 3, 'batch_size': 32, 'accuracy': 0.803051525395875

PKL_PATH = "c:/IA_CarlosMartinez/preprocessed/sequences_all.pkl"
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 100
HIDDEN_DIM = 512
NUM_LAYERS = 3
EARLY_STOPPING_PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[DEVICE]", DEVICE)


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


def compute_auc(loader, num_classes):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # AUC por clase (One-vs-Rest)
    auc_per_class = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
    # AUC promedio (macro)
    auc_avg = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    # Curvas ROC por clase
    fpr = {}
    tpr = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels, all_probs[:, i], pos_label=i)

    return auc_per_class, auc_avg, fpr, tpr


# ======================================
# TRAINING LOOP con EARLY STOPPING
# ======================================
train_acc_hist = []
val_acc_hist = []
train_auc_hist = []
val_auc_hist = []
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

    # AUC de entrenamiento y validación
    train_auc_per_class, train_auc_avg, _, _ = compute_auc(train_loader, num_classes)
    val_auc_per_class, val_auc_avg, _, _ = compute_auc(val_loader, num_classes)

    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)
    train_auc_hist.append(train_auc_avg)
    val_auc_hist.append(val_auc_avg)

    print(f"Epoch {epoch}/{EPOCHS} | Loss: {running_loss/len(train_dataset):.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Train AUC: {train_auc_avg:.4f} | Val AUC: {val_auc_avg:.4f}")
    torch.save(model.state_dict(), "best_biLSTM_model.pth")


print("\nEntrenamiento finalizado. Mejor Val Accuracy:", best_val_acc)


# ======================================
# CALCULAR AUC FINAL POR CLASE
# ======================================
final_train_auc_per_class, final_train_auc_avg, train_fpr, train_tpr = compute_auc(train_loader, num_classes)
final_val_auc_per_class, final_val_auc_avg, val_fpr, val_tpr = compute_auc(val_loader, num_classes)

print(f"AUC promedio Train: {final_train_auc_avg:.4f}")
print(f"AUC promedio Val: {final_val_auc_avg:.4f}")


# ======================================
# GRAFICAR ACCURACY
# ======================================
plt.figure(figsize=(8, 4))
plt.plot(train_acc_hist, label="Accuracy Entrenamiento")
plt.plot(val_acc_hist, label="Accuracy Validación")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Accuracy de entrenamiento vs validación")
plt.legend()
plt.grid()
plt.savefig("accuracy_plot.png")
plt.show()

print("Gráfica de accuracy guardada como accuracy_plot.png")


# ======================================
# GRAFICAR AUC PROMEDIO
# ======================================
plt.figure(figsize=(8, 4))
plt.plot(train_auc_hist, label="AUC Entrenamiento")
plt.plot(val_auc_hist, label="AUC Validación")
plt.xlabel("Época")
plt.ylabel("AUC")
plt.title("AUC promedio de entrenamiento vs validación")
plt.legend()
plt.grid()
plt.savefig("auc_plot.png")
plt.show()

print("Gráfica de AUC guardada como auc_plot.png")


# ======================================
# GRAFICAR CURVAS ROC POR CLASE (DIVIDIDAS EN 5 GRÁFICOS)
# ======================================
class_names = list(base_dataset.class_map.keys())
num_classes = len(class_names)
classes_per_plot = 10  # Aproximadamente 47 / 5 ≈ 9.4, usar 10 para 5 plots
num_plots = 5

for plot_idx in range(num_plots):
    start_class = plot_idx * classes_per_plot
    end_class = min((plot_idx + 1) * classes_per_plot, num_classes)
    
    plt.figure(figsize=(10, 8))
    for i in range(start_class, end_class):
        plt.plot(val_fpr[i], val_tpr[i], label=f'{class_names[i]} (AUC = {final_val_auc_per_class[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curvas ROC - Clases {start_class} a {end_class-1}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f"roc_curves_plot_{plot_idx+1}.png")
    plt.show()

print("Gráficas de curvas ROC guardadas como roc_curves_plot_1.png a roc_curves_plot_5.png")


