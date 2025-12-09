import torch
import pickle
import numpy as np
import os
import cv2
import imageio

from dataset_lsc_landmarks import LandmarkSequenceDataset


# ===========================================
# CONFIG
# ===========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_biLSTM_model.pth"
PKL_PATH = "preprocessed/sequences_all.pkl"

print("[DEVICE]", DEVICE)


# ===========================================
# Cargar Dataset (solo para obtener frames reales)
# ===========================================
dataset = LandmarkSequenceDataset(PKL_PATH)
class_map = dataset.class_map
id_to_class = {v: k for k, v in class_map.items()}   # invertir diccionario

print("[INFO] Clases disponibles:", class_map.keys())


# ===========================================
# Definir MODELO (debe coincidir con train)
# ===========================================
class BiLSTMModel(torch.nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=1, num_classes=47):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)  # reshape (B,6,63)
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # tomar último frame
        return self.fc(out)


# Crear el modelo con el número correcto de clases
model = BiLSTMModel(num_classes=len(class_map)).to(DEVICE)

# Cargar pesos
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] Modelo cargado:", MODEL_PATH)


# ===========================================
# Función para generar un GIF de una seña real
# ===========================================
def show_sequence_as_gif(example_frames, save_path="output.gif", duration=0.3):
    frames = []
    for img_array in example_frames:  # img_array es landmark → buscamos imágenes reales
        pass  # (este modelo no usa imágenes reales, así que buscamos otras fuentes)


# ===========================================
# Recuperar FRAMES reales del dataset
# ===========================================
# dataset.samples contiene la tupla (secuencia_landmarks, label)
# Pero NO contiene imágenes, así que necesitamos usarlas del dataset REAL
# Creamos un mapa: clase_id → indices donde aparece en dataset
class_to_indices = {}
for idx, lbl in enumerate(dataset.labels):
    class_to_indices.setdefault(lbl, []).append(idx)


def get_example_frames_for_class(lbl_id):
    """Devuelve una secuencia real de landmarks para mostrar como GIF."""
    sample_idx = class_to_indices[lbl_id][0]  # ejemplo representativo
    landmarks_seq = dataset.sequences[sample_idx]  # lista de 6 arrays (21x3)
    return landmarks_seq


# ===========================================
# Loop de inferencia
# ===========================================
while True:
    text = input("\nIngrese palabra/letra ('exit' para salir): ").strip()
    if text.lower() == "exit":
        break

    if text not in class_map:
        print("❌ Esta palabra no está en el dataset.")
        print("Clases disponibles:", list(class_map.keys()))
        continue

    cls_id_expected = class_map[text]

    # Para predicción usamos el prototipo de esta clase
    example_idx = class_to_indices[cls_id_expected][0]
    seq = dataset.sequences[example_idx]  # landmarks de referencia

    x = torch.tensor([seq], dtype=torch.float32).to(DEVICE)  # (1,6,21,3)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    predicted_label = id_to_class[pred]

    print(f"\n🧠 Predicción del modelo → {predicted_label}")
    print(f"📌 Clase esperada → {text}")

    # Mostrar landmarks reales en un GIF
    seq = np.array(seq)  # convertir para graficar
    gif_frames = []

    for f in seq:
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        for x, y, z in f:
            px = int(x * 700 - 120)
            py = int(y * 700 - 120)
            cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
        gif_frames.append(img[:, :, ::-1])  # BGR→RGB

    save_path = f"sign_{text}.gif"
    imageio.mimsave(save_path, gif_frames, duration=0.4)

    print(f"🎞  GIF generado → {save_path}")



# import torch
# import numpy as np
# import cv2
# import imageio

# from dataset_lsc_landmarks import LandmarkSequenceDataset
# from model_and_loader2 import load_model

# # ==========================================
# # CONFIG
# # ==========================================
# PKL_PATH = "c:/IA_CarlosMartinez/preprocessed/sequences_all.pkl"
# MODEL_PATH = "best_biLSTM_model.pth"

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset = LandmarkSequenceDataset(PKL_PATH, augment=False)
# class_map = dataset.class_map
# id_to_class = {v: k for k, v in class_map.items()}
# num_classes = len(class_map)

# print("[DEVICE]", DEVICE)
# print("[INFO] Clases disponibles:", list(class_map.keys()))

# # cargar modelo
# model = load_model(
#     MODEL_PATH,
#     device=DEVICE,
#     input_dim=63,
#     hidden_dim=128,
#     num_layers=1,
#     num_classes=num_classes
# )
# print("[INFO] Modelo cargado:", MODEL_PATH)

# # mapa para hallar un ejemplo real por clase
# class_to_indices = {}
# for idx, lbl in enumerate(dataset.labels):
#     class_to_indices.setdefault(lbl, []).append(idx)

# def draw_landmarks_as_gif(seq, save_path):
#     seq = np.array(seq)

#     SCALE = 500
#     OFFSET_X = 50
#     OFFSET_Y = 50
#     EXTRA_SPREAD = 120

#     frames = []

#     for f in seq:
#         img = np.ones((600,600,3), dtype=np.uint8) * 255

#         for i, (x,y,z) in enumerate(f):
#             px = int(x * SCALE + OFFSET_X)
#             py = int(y * SCALE + OFFSET_Y)
#             py += (i // 7) * EXTRA_SPREAD
#             cv2.circle(img, (px,py), 10, (0,0,255), -1)

#         frames.append(img[:,:,::-1])

#     imageio.mimsave(save_path, frames, duration=0.4)
#     print(f"🎞 GIF generado → {save_path}")

# # ==========================================
# # LOOP DE INFERENCIA
# # ==========================================
# while True:
#     text = input("\nIngrese palabra/letra ('exit' para salir): ").strip()
#     if text.lower() == "exit":
#         break

#     if text not in class_map:
#         print("❌ Esta palabra no está en el dataset.")
#         print("Clases disponibles:", list(class_map.keys()))
#         continue

#     lbl_id = class_map[text]
#     example_idx = class_to_indices[lbl_id][0]
#     seq = dataset.sequences[example_idx]

#     x = torch.tensor([seq], dtype=torch.float32).to(DEVICE)

#     with torch.no_grad():
#         out = model(x)
#         pred = torch.argmax(out, dim=1).item()

#     predicted_label = id_to_class[pred]

#     print(f"\n🧠 Predicción del modelo → {predicted_label}")
#     print(f"📌 Clase esperada → {text}")

#     gif_path = f"sign_{text}.gif"
#     draw_landmarks_as_gif(seq, gif_path)
