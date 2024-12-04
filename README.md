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
├── data/                    # Directorio para datasets
├── lightning_modules/       # Módulos de PyTorch Lightning
│   ├── __init__.py
│   └── restnet_module.py    # Implementación de ResNet
├── models/                  # Modelos guardados
│   └── checkpoints/        # Checkpoints durante el entrenamiento
├── notebooks/              # Jupyter notebooks para análisis
├── results/                # Resultados y métricas
├── scripts/                # Scripts de entrenamiento
│   ├── fine_tuning_function.py
│   └── train_animals.py
├── utils/                  # Utilidades
│   └── data_loader.py      # Funciones de carga de datos
└── requirements.txt        # Dependencias del proyecto
```

## 🚀 Características Principales

### ResNet Transfer Learning
- Modelo base: ResNet18 pre-entrenado en ImageNet
- Capacidad de congelar bloques selectivamente para transfer learning
- Métricas implementadas:
  - Accuracy (entrenamiento, validación y prueba)
  - Pérdida (entrenamiento, validación y prueba)
- Optimización:
  - Optimizador: AdamW
  - Learning Rate Scheduler: ReduceLROnPlateau
  - Precisión mixta para optimización de memoria

### Dataset
- Dataset: Animals de Hugging Face
- 90 clases diferentes de animales
- División automática en conjuntos de entrenamiento y validación
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
python scripts/train_animals.py

# Los parámetros configurables incluyen:
- num_classes: Número de clases (default: 90)
- batch_size: Tamaño del batch (default: 32)
- learning_rate: Tasa de aprendizaje (default: 3e-5)
- max_epochs: Número máximo de épocas (default: 30)
- freeze_blocks: Número de bloques a congelar (default: 3)
```

### Monitoreo
```bash
# Visualizar métricas en TensorBoard
tensorboard --logdir results/animals/logs
```

## 📊 Características del Entrenamiento

### Configuración del Modelo
- **Arquitectura Base**: ResNet18
- **Transfer Learning**: 
  - Pesos pre-entrenados de ImageNet
  - Capacidad de congelar hasta 4 bloques
  - Capa final adaptada a 90 clases

### Optimización
- **Optimizador**: AdamW
- **Learning Rate**: 3e-5 (configurable)
- **Scheduler**: ReduceLROnPlateau
  - Factor de reducción: 0.5
  - Paciencia: 3 épocas
  - Monitoreo: val_loss

### Callbacks
- **Model Checkpoint**: Guarda los mejores modelos basados en val_loss
- **Early Stopping**: Detiene el entrenamiento si no hay mejora
- **TensorBoard Logger**: Registra métricas de entrenamiento

## 📈 Métricas y Logging
- **Métricas de Entrenamiento**:
  - Loss (por paso y época)
  - Accuracy (por paso y época)
- **Métricas de Validación**:
  - Loss
  - Accuracy
- **Visualización**: TensorBoard para seguimiento en tiempo real

## 🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, sigue estos pasos:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📝 Licencia
Este proyecto está bajo la licencia [ESPECIFICAR_LICENCIA].