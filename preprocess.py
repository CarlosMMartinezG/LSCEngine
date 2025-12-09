# # preprocess.py
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# import pickle
# from tqdm import tqdm

# DATASET_ROOT = r"C:\IA_CarlosMartinez\LSC70\LSC70"
# OUT_FILE = "preprocessed/sequences_all.pkl"

# os.makedirs("preprocessed", exist_ok=True)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# def extract_landmarks(img):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     res = hands.process(img_rgb)
#     if not res.multi_hand_landmarks:
#         return None
#     lm = res.multi_hand_landmarks[0]
#     arr = np.array([[p.x, p.y, p.z] for p in lm.landmark])
#     return arr   # shape (21,3)

# class_map = {}   # class_name -> idx
# sequences = []   # list of sequences (each = list length 6 of (21,3))
# labels = []      # class idx for each sequence

# print("Procesando dataset real...")

# for top_folder in sorted(os.listdir(DATASET_ROOT)):
#     top_path = os.path.join(DATASET_ROOT, top_folder)
#     if not os.path.isdir(top_path):
#         continue

#     # Dentro del top_folder vienen las carpetas de personas
#     for person in sorted(os.listdir(top_path)):
#         person_path = os.path.join(top_path, person)
#         if not os.path.isdir(person_path):
#             continue

#         # Dentro de cada persona vienen las clases
#         for cls in sorted(os.listdir(person_path)):
#             cls_path = os.path.join(person_path, cls)
#             if not os.path.isdir(cls_path):
#                 continue

#             # asignar ID a clase
#             if cls not in class_map:
#                 class_map[cls] = len(class_map)
#             cls_idx = class_map[cls]

#             # Leer frames
#             frame_files = sorted([
#                 f for f in os.listdir(cls_path)
#                 if f.lower().endswith((".jpg", ".png"))
#             ])
#             if len(frame_files) == 0:
#                 continue

#             seq = []
#             for fname in frame_files:
#                 img = cv2.imread(os.path.join(cls_path, fname))
#                 lm = extract_landmarks(img)
#                 if lm is None:
#                     lm = np.zeros((21,3), dtype=np.float32)
#                 seq.append(lm)

#             # Normalizar a 6 frames
#             if len(seq) < 6:
#                 seq += [np.zeros((21,3), dtype=np.float32)]*(6-len(seq))
#             elif len(seq) > 6:
#                 seq = seq[:6]

#             sequences.append(seq)
#             labels.append(cls_idx)


# hands.close()

# data = {
#     "class_map": class_map,
#     "sequences": sequences,
#     "labels": labels
# }

# with open(OUT_FILE, "wb") as f:
#     pickle.dump(data, f)

# print("Listo. Secuencias guardadas:", len(sequences))
# print("Clases detectadas:", len(class_map))



# preprocess.py
import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tqdm import tqdm

DATASET_ROOT = r"C:\IA_CarlosMartinez\LSC70\LSC70"
OUT_FILE = "preprocessed/sequences_all.pkl"
os.makedirs("preprocessed", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(img):
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
    return arr   # (21,3)

class_map = {}
sequences = []
labels = []

print("Procesando dataset real...")

# Recorremos tres niveles: top_folder -> person -> class
for top_folder in sorted(os.listdir(DATASET_ROOT)):
    top_path = os.path.join(DATASET_ROOT, top_folder)
    if not os.path.isdir(top_path):
        continue
    for person in sorted(os.listdir(top_path)):
        person_path = os.path.join(top_path, person)
        if not os.path.isdir(person_path):
            continue
        for cls in sorted(os.listdir(person_path)):
            cls_path = os.path.join(person_path, cls)
            if not os.path.isdir(cls_path):
                continue

            if cls not in class_map:
                class_map[cls] = len(class_map)
            cls_idx = class_map[cls]

            frame_files = sorted([f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".png"))])
            if len(frame_files) == 0:
                continue

            seq = []
            last_valid = None
            for fname in frame_files:
                img_path = os.path.join(cls_path, fname)
                img = cv2.imread(img_path)
                lm = extract_landmarks(img)
                if lm is None:
                    # si no se detecta, reutilizar el último frame válido (si existe)
                    if last_valid is not None:
                        seq.append(last_valid.copy())
                    else:
                        # aún no hay un frame válido: dejaremos para rellenar después
                        seq.append(None)
                else:
                    seq.append(lm)
                    last_valid = lm.copy()

            # eliminar leading None si no hay ningún frame válido
            if all(s is None for s in seq):
                # no hay hand detections en toda la secuencia -> saltar
                continue

            # rellenar None usando último válido (propagar hacia adelante)
            for i in range(len(seq)):
                if seq[i] is None:
                    # buscar anterior válido
                    if i > 0 and seq[i-1] is not None:
                        seq[i] = seq[i-1].copy()
                    else:
                        # buscar siguiente válido
                        j = i+1
                        while j < len(seq) and seq[j] is None:
                            j += 1
                        if j < len(seq) and seq[j] is not None:
                            seq[i] = seq[j].copy()
                        else:
                            # fallback: zeros (muy raro porque habíamos filtrado)
                            seq[i] = np.zeros((21,3), dtype=np.float32)

            # normalize length to 6: if fewer, repeat last valid; if more, sample uniformly
            if len(seq) < 6:
                while len(seq) < 6:
                    seq.append(seq[-1].copy())
            elif len(seq) > 6:
                # sample 6 indices uniformly across available frames to preserve temporal spread
                idxs = np.linspace(0, len(seq)-1, 6, dtype=int)
                seq = [seq[i] for i in idxs]

            # ensure dtype float32 and shape (6,21,3)
            seq = [np.array(s, dtype=np.float32) for s in seq]
            sequences.append(seq)
            labels.append(cls_idx)

hands.close()

data = {"class_map": class_map, "sequences": sequences, "labels": labels}
with open(OUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("Listo. Secuencias guardadas:", len(sequences))
print("Clases detectadas:", len(class_map))
