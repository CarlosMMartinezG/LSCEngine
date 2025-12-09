import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

from dataset_lsc_landmarks import LandmarkSequenceDataset

# =====================================================
# CONFIGURACIÓN
# =====================================================
PKL_PATH = "preprocessed/sequences_all.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 5  # K-fold cross validation

# =====================================================
# GRID DE HIPERPARÁMETROS
# =====================================================
param_grid = {
    "lr":        [0.01, 0.001, 0.0001],
    "hidden_dim": [128, 256, 512],
    "num_layers": [1, 2, 3],
    "batch_size": [16, 32, 64]
}

# =====================================================
# CARGAR DATASET (base sin augment y otro con augment)
# =====================================================
# Base dataset (no augment) used for validation and for indexing
base_dataset = LandmarkSequenceDataset(PKL_PATH, augment=False)
# Augmented dataset used only for training
aug_dataset = LandmarkSequenceDataset(PKL_PATH, augment=True)
num_classes = len(base_dataset.class_map)

# =====================================================
# DEFINIR MODELO (INDPENDIENTE DE TRAIN.PY)
# =====================================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
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
        B, T, V, C = x.size()  # (batch,6,21,3)
        x = x.reshape(B, T, V*C)  # (batch,6,63)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # último paso
        return self.fc(out)

# =====================================================
# FUNCIÓN PARA ENTRENAR Y VALIDAR UN SPLIT
# =====================================================
def train_and_eval(train_idx, val_idx, params):

    # For training use the augmented dataset; for validation use the base (no augment)
    train_subset = Subset(aug_dataset, train_idx)
    val_subset = Subset(base_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False)

    model = BiLSTMModel(
        input_dim=63,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        num_classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    EPOCHS = 50 # pequeño para búsquedas de hiperparámetros

    # Helpers to compute accuracy on a loader
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
        return correct / total if total > 0 else 0.0

    # Historial por época
    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []

    # ---------- ENTRENAMIENTO por épocas (registrando historial) ----------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for seq, lbl in train_loader:
            seq = seq.to(DEVICE)
            lbl = lbl.to(DEVICE)

            optimizer.zero_grad()
            out = model(seq)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * seq.size(0)

        avg_loss = running_loss / len(train_subset)
        train_loss_hist.append(avg_loss)

        # calcular accuracies (entrenamiento y validación)
        train_acc = compute_accuracy(train_loader)
        val_acc = compute_accuracy(val_loader)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # progreso mínimo por época
        print(f"    Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

    # Devolver accuracy final (última época de validación) y el historial
    final_val_acc = val_acc_hist[-1] if val_acc_hist else 0.0
    history = {"train_loss": train_loss_hist, "train_acc": train_acc_hist, "val_acc": val_acc_hist}
    return final_val_acc, history


# =====================================================
# GRID SEARCH + K-FOLD
# =====================================================
kf = KFold(n_splits=K, shuffle=True, random_state=42)
results = []

# Usar índices del dataset base para K-Fold (base_dataset fue creado más arriba)
indices = list(range(len(base_dataset)))

for lr in param_grid["lr"]:
    for hd in param_grid["hidden_dim"]:
        for nl in param_grid["num_layers"]:
            for bs in param_grid["batch_size"]:

                params = {"lr": lr, "hidden_dim": hd, "num_layers": nl, "batch_size": bs}
                print(f"\nProbando params: {params}")

                fold_scores = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
                    print(f"  Fold {fold+1}/{K}")
                    acc, history = train_and_eval(train_idx, val_idx, params)
                    fold_scores.append(acc)

                mean_acc = np.mean(fold_scores)
                # Guardar la media de accuracy; opcionalmente también se puede guardar
                # el history de la última fold si interesa analizar curvas.
                results.append({**params, "accuracy": mean_acc, "history": history})

                print(f"Accuracy promedio = {mean_acc:.4f}")

# =====================================================
# MEJORES RESULTADOS
# =====================================================
results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)

print("\n======================= RESULTADOS =======================")
for r in results_sorted[:5]:
    print(r)

best = results_sorted[0]
print("\n================ HIPERPARÁMETROS ÓPTIMOS ================")
print(best)

# =====================================================
# GRAFICO 3D LR vs HIDDEN_DIM vs ACCURACY
# =====================================================
from mpl_toolkits.mplot3d import Axes3D

lrs = [r["lr"] for r in results]
hds = [r["hidden_dim"] for r in results]
accs = [r["accuracy"] for r in results]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lrs, hds, accs)
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Hidden Dim")
ax.set_zlabel("Accuracy")
plt.title("Grid Search Results")
plt.show()





# =====================================================
# HEATMAPS DE PRECISIÓN
# =====================================================
import seaborn as sns
import pandas as pd

# Convertir resultados a DataFrame
df = pd.DataFrame(results)

# Generar un heatmap por cada LR
for lr in param_grid["lr"]:
    df_lr = df[df["lr"] == lr]

    print(f"\n=== HEATMAPS para LR = {lr} ===")

    # ------------------------------
    # 1. HEATMAP: hidden_dim vs batch_size
    # ------------------------------
    heat1 = df_lr.pivot_table(
        values="accuracy",
        index="hidden_dim",
        columns="batch_size"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(heat1, annot=True, cmap="viridis", fmt=".3f")
    plt.title(f"Accuracy Heatmap — Hidden Dim vs Batch Size (LR={lr})")
    plt.xlabel("Batch Size")
    plt.ylabel("Hidden Dim")
    plt.show()

    # ------------------------------
    # 2. HEATMAP: hidden_dim vs num_layers
    # ------------------------------
    heat2 = df_lr.pivot_table(
        values="accuracy",
        index="hidden_dim",
        columns="num_layers"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(heat2, annot=True, cmap="magma", fmt=".3f")
    plt.title(f"Accuracy Heatmap — Hidden Dim vs Num Layers (LR={lr})")
    plt.xlabel("Num Layers")
    plt.ylabel("Hidden Dim")
    plt.show()

    # ------------------------------
    # 3. HEATMAP: batch_size vs num_layers
    # ------------------------------
    heat3 = df_lr.pivot_table(
        values="accuracy",
        index="batch_size",
        columns="num_layers"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(heat3, annot=True, cmap="plasma", fmt=".3f")
    plt.title(f"Accuracy Heatmap — Batch Size vs Num Layers (LR={lr})")
    plt.xlabel("Num Layers")
    plt.ylabel("Batch Size")
    plt.show()
