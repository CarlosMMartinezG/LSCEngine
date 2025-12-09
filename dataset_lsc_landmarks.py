# import torch
# from torch.utils.data import Dataset
# import pickle
# import numpy as np

# class LandmarkSequenceDataset(Dataset):
#     def __init__(self, pkl_path):
#         with open(pkl_path, "rb") as f:
#             data = pickle.load(f)

#         self.sequences = data["sequences"]   # lista de listas de 6 arrays (21,3)
#         self.labels = data["labels"]         # lista de enteros
#         self.class_map = data["class_map"]   # dict clase -> id

#         print("[LandmarkSequenceDataset] Loaded:")
#         print(f"  Total samples: {len(self.sequences)}")
#         print(f"  Num classes : {len(self.class_map)}")
#         print(f"  Classes     : {self.class_map}")

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         seq = self.sequences[idx]            # list of 6 arrays (21,3)
#         label = self.labels[idx]

#         # convertir secuencia a tensor
#         seq = np.array(seq, dtype=np.float32)    # (6,21,3)
#         seq = torch.tensor(seq)                  # tensor (6,21,3)

#         return seq, label

####Desde aquí###

#dataset_lsc_landmarks.py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# ============================
# NORMALIZACIÓN
# ============================

def normalize_landmarks(seq):
    """
    seq: (6, 21, 3)
    Normaliza la secuencia:
    1) Resta respecto al landmark 0 (wrist)
    2) Escala para que la distancia máxima sea 1
    """
    seq = seq.copy()

    # Resta el punto base
    wrist = seq[:, 0:1, :]       # (6,1,3)
    seq = seq - wrist

    # Escalado por distancia máxima
    max_vals = np.max(np.linalg.norm(seq, axis=2, keepdims=True), axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1e-6
    seq = seq / max_vals

    return seq


# ============================
# DATA AUGMENTATION
# ============================

def jitter(seq, sigma=0.01):
    """Ruido gaussiano leve."""
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

def random_scale(seq, min_scale=0.9, max_scale=1.1):
    """Escalado aleatorio."""
    s = np.random.uniform(min_scale, max_scale)
    return seq * s

def random_rotate_z(seq, max_deg=20):
    """Rotación 2D en eje Z (efectiva para cámaras)."""
    theta = np.radians(np.random.uniform(-max_deg, max_deg))
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return seq @ R.T

def random_rotate_3d(seq, max_deg=10):
    """Rotación suave en 3D."""
    a = np.radians(np.random.uniform(-max_deg, max_deg, size=3))

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(a[0]), -np.sin(a[0])],
        [0, np.sin(a[0]),  np.cos(a[0])]
    ])
    Ry = np.array([
        [np.cos(a[1]), 0, np.sin(a[1])],
        [0, 1, 0],
        [-np.sin(a[1]), 0, np.cos(a[1])]
    ])
    Rz = np.array([
        [np.cos(a[2]), -np.sin(a[2]), 0],
        [np.sin(a[2]),  np.cos(a[2]), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return seq @ R.T


# ============================================================
# DATASET COMPLETO
# ============================================================

class LandmarkSequenceDataset(Dataset):
    def __init__(self, pkl_path, augment=False):
        """
        pkl_path: archivo generado por preprocess.py
        augment: aplicar data augmentation (solo en entrenamiento)
        """
        data = pickle.load(open(pkl_path, "rb"))

        self.sequences = data["sequences"]     # lista de (6,21,3)
        self.labels = data["labels"]
        self.class_map = data["class_map"]

        self.augment = augment

        print("[LandmarkSequenceDataset] Loaded:")
        print(f"  Total samples: {len(self.sequences)}")
        print(f"  Num classes : {len(self.class_map)}")
        print(f"  Classes     : {self.class_map}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = np.array(self.sequences[idx], dtype=np.float32)  # (6,21,3)

        # Siempre normalizar
        seq = normalize_landmarks(seq)

        # Augment solo en entrenamiento
        if self.augment:
            if np.random.rand() < 0.5:
                seq = jitter(seq)
            if np.random.rand() < 0.5:
                seq = random_scale(seq)
            if np.random.rand() < 0.5:
                seq = random_rotate_z(seq)
            if np.random.rand() < 0.3:
                seq = random_rotate_3d(seq)

        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.float32), label




# # dataset_lsc_landmarks.py
# import pickle
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# def normalize_landmarks(seq):
#     """
#     seq: (6,21,3)
#     - centrar por muñeca (landmark 0)
#     - escalar con UN valor por secuencia (max norm across all frames)
#     """
#     seq = seq.copy().astype(np.float32)
#     # centrar
#     seq = seq - seq[:, 0:1, :]
#     # escala global por secuencia
#     norms = np.linalg.norm(seq.reshape(-1, 3), axis=1)
#     max_val = np.max(norms)
#     if max_val < 1e-6:
#         max_val = 1e-6
#     seq = seq / max_val
#     return seq

# # Augmentations
# def jitter(seq, sigma=0.01):
#     return seq + np.random.normal(0, sigma, size=seq.shape)

# def random_scale(seq, min_scale=0.95, max_scale=1.05):
#     s = np.random.uniform(min_scale, max_scale)
#     return seq * s

# def random_rotate_z(seq, max_deg=15):
#     theta = np.radians(np.random.uniform(-max_deg, max_deg))
#     R = np.array([[np.cos(theta), -np.sin(theta), 0],
#                   [np.sin(theta),  np.cos(theta), 0],
#                   [0, 0, 1]], dtype=np.float32)
#     # apply same rotation to all frames
#     return np.einsum('ij,ftj->fti', R, seq)

# def random_rotate_3d(seq, max_deg=8):
#     a = np.radians(np.random.uniform(-max_deg, max_deg, size=3))
#     Rx = np.array([[1,0,0],[0,np.cos(a[0]),-np.sin(a[0])],[0,np.sin(a[0]),np.cos(a[0])]])
#     Ry = np.array([[np.cos(a[1]),0,np.sin(a[1])],[0,1,0],[-np.sin(a[1]),0,np.cos(a[1])]])
#     Rz = np.array([[np.cos(a[2]),-np.sin(a[2]),0],[np.sin(a[2]),np.cos(a[2]),0],[0,0,1]])
#     R = Rz @ Ry @ Rx
#     return np.einsum('ij,ftj->fti', R, seq)

# class LandmarkSequenceDataset(Dataset):
#     def __init__(self, pkl_path, augment=False):
#         data = pickle.load(open(pkl_path, "rb"))
#         self.sequences = data["sequences"]   # list of lists (6,21,3)
#         self.labels = data["labels"]
#         self.class_map = data["class_map"]
#         self.augment = augment

#         # Convert labels to list (some scripts expect list)
#         if not isinstance(self.labels, list):
#             self.labels = list(self.labels)

#         print("[LandmarkSequenceDataset] Loaded:")
#         print(f"  Total samples: {len(self.sequences)}")
#         print(f"  Num classes : {len(self.class_map)}")

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         seq = np.array(self.sequences[idx], dtype=np.float32)  # (6,21,3)
#         # Always normalize (per sequence)
#         seq = normalize_landmarks(seq)

#         # Augment only when requested (training)
#         if self.augment:
#             if np.random.rand() < 0.5:
#                 seq = jitter(seq, sigma=0.01)
#             if np.random.rand() < 0.5:
#                 seq = random_scale(seq, 0.95, 1.05)
#             if np.random.rand() < 0.5:
#                 seq = random_rotate_z(seq, max_deg=12)
#             if np.random.rand() < 0.2:
#                 seq = random_rotate_3d(seq, max_deg=6)

#         return torch.tensor(seq, dtype=torch.float32), int(self.labels[idx])
