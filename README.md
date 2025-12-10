# LSCEngine - Reconocimiento de Lengua de Signos Colombiana (LSC)

Sistema de reconocimiento de gestos en Lengua de Signos Colombiana usando MediaPipe para extracción de landmarks y redes neuronales BiLSTM con PyTorch.

## 📋 Descripción del Proyecto

LSCEngine es un pipeline completo para:
1. **Extracción de landmarks**: Usar MediaPipe para detectar puntos clave de manos en video
2. **Preprocesamiento**: Normalizar secuencias de landmarks y generar dataset pickle
3. **Entrenamiento**: Entrenar modelo BiLSTM bidireccional con data augmentation
4. **Optimización**: Búsqueda de hiperparámetros con K-fold cross-validation
5. **Inferencia**: Generar Landmarks de seña

## 🏗️ Arquitectura del Modelo

**Modelo: BiLSTM (Bidirectional LSTM)**

```
Entrada (6 frames × 21 landmarks × 3 coords = 6×63)
    ↓
BiLSTM (512 hidden dims × 3 capas, bidireccional → 1024 características)
    ↓
Linear (1024 → 512)
    ↓
ReLU Activation
    ↓
Linear (512 → num_classes)
    ↓
Salida (probabilidades por clase)
```

**Hiperparámetros óptimos (encontrados en `hyper.py`):**
- Learning Rate: 0.001
- Hidden Dimension: 512
- Num Layers: 3
- Batch Size: 32
- Accuracy: ~80.3%

## 📁 Estructura de Archivos

```
LSCEngine/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias del proyecto
│
├── 1️⃣ preprocess.py             # [PASO 1] Extraer landmarks con MediaPipe
├── preprocessed/                # Datos preprocesados (generado automáticamente)
│   └── sequences_all.pkl        # Dataset serializado con landmarks
│
├── dataset_lsc_landmarks.py     # Dataset PyTorch + normalización + augmentation
├── model_and_loader2.py         # Definición del modelo BiLSTM
│
├── 2️⃣ hyper.py                  # [PASO 2] Búsqueda de hiperparámetros (opcional)
├── 3️⃣ train_with_validation.py  # [PASO 3] Entrenamiento con validación 
│
├── infer_biLSTM.py              # [PASO 4] Inferencia: Generación de Landmarks según texto ingresado
│
├── models/                      # Modelos guardados
│   └── best_biLSTM_model.pth
│
└── results/                     # Resultados de experimentos
    └── accuracy_plot.jpg
```

## 🚀 Orden de Ejecución

### **Opción A: Pipeline Completo (Recomendado)**

#### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

#### 2. Preprocesar dataset (extraer landmarks)
```bash
python preprocess.py
```
- Lee imágenes de `LSC70/`
- Extrae landmarks de manos con MediaPipe
- Genera `preprocessed/sequences_all.pkl`
- 

#### 3. Entrenar modelo (opcional: primero optimizar hiperparámetros)
```bash
# Opcional: búsqueda de hiperparámetros (más lento)
python hyper.py

# O entrenar directamente con hiperparámetros óptimos
python train_with_validation.py
```
- Entrena BiLSTM con validación 80/20
- Early stopping si no mejora en 10 épocas
- Guarda mejor modelo en `best_biLSTM_model.pth`
-

#### 4. Generar Landmarks a partir de texto
```bash
python infer_biLSTM.py
```

- Genera GIFs con predicciones (`sign_*.gif`)

---

### **Opción B: Solo Inferencia (Si ya tienes modelo entrenado)**
```bash
python infer_biLSTM.py
```
- Necesita `best_biLSTM_model.pth` existente
- No requiere preprocesamiento ni entrenamiento

---

## ⚙️ Configuración

### Variables principales en scripts:

**`train_with_validation.py`:**
```python
PKL_PATH = "c:/IA_CarlosMartinez/preprocessed/sequences_all.pkl"  # Ruta a dataset
BATCH_SIZE = 32          # Tamaño de lote
LR = 0.001               # Learning rate
EPOCHS = 100             # Épocas máximas
HIDDEN_DIM = 512         # Dimensión LSTM
NUM_LAYERS = 3           # Capas LSTM
DEVICE = "cuda" or "cpu" # GPU automática si disponible
```

**`preprocess.py`:**
```python
DATASET_ROOT = r"C:\IA_CarlosMartinez\LSC70\LSC70"  # Ruta dataset raw
OUT_FILE = "preprocessed/sequences_all.pkl"         # Salida
SEQUENCE_LENGTH = 6      # Frames por secuencia
```

## 📊 Datos de Entrada

**Dataset LSC70:**
- 3 variantes: LSC70AN, LSC70ANH, LSC70W
- Estructura: `Variante/Persona/Clase/Imágenes(6 frames de seña)`
- ~47 clases (gestos/palabras diferentes)

**Formato de landmarks:**
- 21 puntos por mano (MediaPipe Hands)
- Coordenadas: (x, y, z) normalizadas [0, 1]
- Entrada modelo: (batch_size, seq_len=6, 21*3=63)

## 📈 Resultados

**Mejor modelo encontrado:**
- **Accuracy**: 80.3%
- **Configuración**: lr=0.001, hidden_dim=512, num_layers=3, batch_size=32
- **Dataset**: 80% entrenamiento, 20% validación

## 🛠️ Troubleshooting

### Error: "OOM (Out of Memory)"
- Reducir `BATCH_SIZE` (e.g., 32 → 16)
- Reducir `HIDDEN_DIM` (e.g., 512 → 256)
- Usar GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Error: "No module named 'mediapipe'"
```bash
pip install mediapipe
```

### Error: "File not found: sequences_all.pkl"
- Ejecutar primero `python preprocess.py`

### Modelo entrenado lentamente
- Habilitar GPU si disponible: `torch.cuda.is_available()`
- Ejecutar en máquina con más cores CPU

## 📚 Dependencias Principales

| Librería | Versión | Uso |
|----------|---------|-----|
| `torch` | ≥2.0 | Framework deep learning |
| `torchvision` | ≥0.15 | Utilitarios visión |
| `mediapipe` | ≥0.10 | Extracción landmarks |
| `opencv-python` | ≥4.8 | Procesamiento video |
| `numpy` | ≥1.24 | Operaciones numéricas |
| `scikit-learn` | ≥1.2 | Train/val split |
| `matplotlib` | ≥3.7 | Visualización |
| `tqdm` | ≥4.66 | Barras de progreso |

Ver `requirements.txt` para versiones exactas.

## 📝 Notas

- **Preprocesamiento costoso**: Se ejecuta una sola vez; el resultado se cachea en `.pkl`
- **Data augmentation**: Solo aplicada a entrenamiento, no a validación
- **Device automático**: Usa GPU si `torch.cuda.is_available()`, else CPU


## 👨‍💻 Autor

**Carlos Mario Martínez Gómez**
- GitHub: [@CarlosMMartinezG](https://github.com/CarlosMMartinezG)
- Proyecto: Reconocimiento de Lengua de Signos Colombiana

