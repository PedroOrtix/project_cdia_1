# Optimizar Clasificación de Beans mediante Congelamiento Selectivo en ResNet18

## 📋 Descripción
Este proyecto investiga cómo el congelamiento selectivo de bloques en una arquitectura ResNet18 afecta su capacidad de aprendizaje y rendimiento en la tarea de clasificación de plantas de frijol. El objetivo es demostrar que no siempre es necesario entrenar todos los parámetros de un modelo preentrenado para obtener resultados óptimos, contribuyendo así a la democratización del deep learning mediante la optimización de recursos computacionales.

## 🎯 Objetivos
- Analizar el impacto del congelamiento selectivo de bloques en ResNet18
- Evaluar la relación entre parámetros entrenables y rendimiento
- Optimizar recursos computacionales sin comprometer el acierto
- Demostrar la viabilidad de fine-tuning eficiente en tareas específicas

## 📁 Estructura del Proyecto
```
project/
├── lightning_modules/          # Módulos de PyTorch Lightning para el modelo ResNet18
├── notebooks/                  # Notebooks de análisis exploratorio
├── results/                    # Resultados experimentales
│   ├── beans/                 # Resultados por configuración de congelamiento (1-4 bloques)
│   │   ├── csv_exports/       # Métricas exportadas de TensorBoard
│   │   ├── checkpoints/       # Modelos guardados durante entrenamiento
│   │   └── logs/             # Logs de TensorBoard
│   └── plots/                 # Visualizaciones (accuracy, loss, convergencia, etc.)
├── scripts/                    # Scripts principales
│   ├── train_beans.py         # Script principal de entrenamiento
│   ├── fine_tuning_function.py # Funciones de fine-tuning
│   ├── evaluate_models.py     # Evaluación de modelos
│   ├── data_loader.py        # Utilidades de carga de datos
│   ├── export_tensorboard.py # Exportación de métricas
│   └── plots.py              # Generación de visualizaciones
├── report.md                  # Análisis detallado de resultados
└── requirements.txt           # Dependencias del proyecto
```

## 🚀 Características Principales

### Modelo Base: ResNet18
- Arquitectura: ResNet18 preentrenada en ImageNet
- Configuración flexible de congelamiento de bloques
- Adaptación de la capa final para 3 clases (beans)

### Dataset: Beans (AI-Lab-Makerere)
- 3 clases de plantas de frijol
- Preprocesamiento:
  - Redimensionamiento: 224x224 píxeles
  - Normalización: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Configuración de Entrenamiento
- Batch size: 16
- Learning rate: 1e-4
- Optimizador: AdamW
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=10
- Max epochs: 30

## 💻 Uso

### Instalación
```bash
git clone https://github.com/PedroOrtix/project_cdia_1.git
cd project
pip install -r requirements.txt
```

### Entrenamiento
```python
!python scripts/train_beans.py
```

### Evaluación
```python
!python scripts/evaluate_models.py
```

## 📊 Monitoreo y Resultados

### Métricas Implementadas
- Accuracy (train/val/test)
- Loss (train/val/test)
- Parámetros entrenables vs congelados
- Tiempo de entrenamiento

### Visualizaciones
- Curvas de aprendizaje
- Análisis de convergencia
- Estudio de overfitting
- Comparativa de rendimiento por configuración

### TensorBoard
```bash
tensorboard --logdir results/beans
```

## 📈 Resultados Principales
- Rendimiento óptimo con hasta 3 bloques congelados (>95% accuracy)
- Reducción significativa de parámetros entrenables sin pérdida notable de rendimiento
- Convergencia rápida y estable en todas las configuraciones
- Ausencia de overfitting significativo