# Análisis Comparativo de ResNet: Impacto del Número de Bloques en el Aprendizaje

## 📋 Descripción
Este proyecto investiga cómo el número de bloques residuales en una arquitectura ResNet afecta su capacidad de aprendizaje y rendimiento. Se realizan experimentos comparativos con diferentes configuraciones de ResNet para analizar la relación entre la profundidad de la red y su efectividad.

## 🎯 Objetivos
- Comparar el rendimiento de ResNet con diferentes números de bloques residuales
- Analizar la velocidad de convergencia en el entrenamiento
- Evaluar la precisión final alcanzada por cada configuración
- Identificar la relación óptima entre profundidad y rendimiento

## 🛠️ Requisitos
```python
pytorch-lightning>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.0.0
transformers>=4.0.0
tensorboard>=2.0.0
pillow>=8.0.0
torchmetrics>=0.11.0
```

## 📁 Estructura del Proyecto
```
project/
├── lightning_modules/       # Módulos de PyTorch Lightning
│   ├── __init__.py
│   └── restnet_module.py    # Implementación de ResNet
├── models/                  # Modelos guardados
│   ���── checkpoints/        # Checkpoints durante el entrenamiento
├── results/                # Resultados y métricas
│   └── beans/             # Resultados específicos del dataset de frijoles
├── scripts/                # Scripts de entrenamiento
│   ├── fine_tuning_function.py
│   └── train_beans.py
├── utils/                  # Utilidades
│   └── data_loader.py      # Funciones de carga de datos
└── requirements.txt        # Dependencias del proyecto
```

## 🚀 Características Principales

### ResNet Transfer Learning
- Modelo base: ResNet18 pre-entrenado en ImageNet
- Capacidad de congelar bloques selectivamente para transfer learning
- Métricas implementadas:
  - Accuracy (entrenamiento y validación)
  - Pérdida (entrenamiento y validación)
- Optimización:
  - Optimizador: AdamW
  - Learning Rate: 1e-4

### Dataset
- Dataset: Beans de Hugging Face (AI-Lab-Makerere/beans)
- 3 clases diferentes de plantas de frijol
- Transformaciones de datos:
  - Redimensionamiento a 224x224
  - Normalización con medias y desviaciones estándar de ImageNet

## 💻 Uso

### Instalación
```bash
# Clonar el repositorio
git clone [URL_DEL_REPOSITORIO]
cd project

# Instalar dependencias
pip install -r requirements.txt
```

### Entrenamiento
```bash
# Entrenar el modelo con configuración por defecto
python scripts/train_beans.py

# Los parámetros configurables incluyen:
- num_classes: Número de clases (default: 3)
- batch_size: Tamaño del batch (default: 16)
- learning_rate: Tasa de aprendizaje (default: 1e-4)
- max_epochs: Número máximo de épocas (default: 20)
- freeze_blocks: Número de bloques a congelar (default: 4)
- num_workers: Número de workers para data loading (default: 4)
```

## 📊 Características del Entrenamiento

### Configuración del Modelo
- **Arquitectura Base**: ResNet18
- **Transfer Learning**: 
  - Pesos pre-entrenados de ImageNet
  - 4 bloques congelados por defecto
  - Capa final adaptada a 3 clases

### Optimización
- **Optimizador**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 16
- **Épocas Máximas**: 20

### Resultados
Los resultados del entrenamiento se guardan en:
- Directorio: `results/beans/`
- Formato: Archivos JSON con métricas y configuraciones
- Timestamp: Cada experimento se guarda con marca de tiempo única

## 📝 Licencia
Este proyecto está bajo la licencia [ESPECIFICAR_LICENCIA].