# Optimización de Recursos mediante Congelamiento Selectivo en ResNet18

## Descripción / Abstract

Este estudio desafía la creencia común de que entrenar todos los parámetros de un modelo preentrenado conduce necesariamente a mejores resultados. Utilizando una única arquitectura ResNet18, investigamos cómo el congelamiento estratégico de diferentes bloques puede mantener o incluso mejorar el rendimiento mientras se reduce significativamente el número de parámetros entrenables y los recursos computacionales necesarios. El objetivo es demostrar que una configuración más eficiente de los parámetros entrenables puede lograr resultados comparables o superiores, optimizando así el uso de recursos computacionales sin comprometer el rendimiento.

---

## Introducción

Si bien históricamente existió una tendencia a utilizar todos los parámetros disponibles en modelos preentrenados, actualmnete es bien conocida la importancia crítica de la eficiencia computacional. Este cambio de paradigma está impulsado por la necesidad de democratizar el acceso a los modelos de aprendizaje profundo, haciéndolos más accesibles y prácticos para su implementación en servicios del mundo real. 

Esta tendencia hacia la democratización se refleja en el desarrollo de diversas técnicas de optimización como:
- **LoRA** (Low-Rank Adaptation), que permite el fine-tuning eficiente mediante la factorización de matrices
- **Destilación de conocimiento** para crear modelos más compactos
- **Cuantización de pesos** para reducir los requisitos de memoria y acelerar la inferencia

En este contexto, nuestro estudio utiliza la ResNet18 como caso de estudio para demostrar que, mediante el congelamiento selectivo de capas, podemos reducir significativamente la cantidad de parámetros entrenables mientras mantenemos o mejoramos el rendimiento del modelo.

La optimización de recursos sin comprometer el rendimiento se ha convertido en un factor crucial para hacer que las soluciones de inteligencia artificial sean verdaderamente escalables y accesibles para todo el mundo, desde pequeñas empresas hasta aplicaciones de gran escala.

---

## Metodología/Modelo

### Dataset 📊

El estudio utiliza el dataset "Beans" proporcionado por AI-Lab-Makerere a través de HuggingFace, que consiste en imágenes de plantas de frijol para clasificación. Los datos se procesan mediante una pipeline de transformaciones que incluye:

- 🔄 Redimensionamiento de imágenes a 224x224 píxeles
- 📊 Normalización utilizando los valores estándar de ImageNet:
  - mean=[0.485, 0.456, 0.406]
  - std=[0.229, 0.224, 0.225]
- 🔢 Transformación a tensores para su procesamiento en PyTorch

El dataset se divide en tres conjuntos: entrenamiento, validación y test, manteniendo la estructura original proporcionada por los autores.

### Arquitectura del Modelo 🏗️

Se utiliza una ResNet18 pre-entrenada en ImageNet (IMAGENET1K_v1) como modelo base, modificada para la tarea de clasificación específica:

- **Modelo base**: ResNet18 con pesos pre-entrenados
- **Capa de clasificación**: Adaptada para 3 clases (correspondientes a las categorías de plantas de frijol)
- **Estrategia de congelamiento**: Implementación de congelamiento selectivo por bloques, permitiendo experimentar con diferentes configuraciones de parámetros entrenables

### Proceso de Entrenamiento ⚙️

#### Configuración Base:
| Parámetro | Valor |
|-----------|--------|
| Batch size | 16 |
| Learning rate | 1e-4 |
| Max epochs | 30 |
| Workers | 4 |

#### Elementos de la estrategia de training:
- ✅ Callback para guardar checkpoints con filtro en val_loss con pacience = 10
- 🛑 Early stopping con filtro en val_loss con pacience = 10
- 📊 Logger para TensorBoard
- ⚙️ Declaracacion del trainer con:
  - Precisión mixta para optimizar memoria y velocidad
  - Compatibilidad con GPU (Nvidia T4 de Google Colab)
  - deterministic=True para asegurar reproducibilidad
  - log_every_n_steps=10 para registrar métricas cada 10 steps

#### Decisiones de diseño:
- 📈 Uso de tensorboard para registrar métricas y visualizaciones
- ⚡ Uso de pytorch-lightning para agilizar el desarrollo
- 🎯 Epochs limitadas a 30 con early stopping para prevenir overfitting

---

## Resultados
### Análisis de Resultados 📊

#### Accuracy de Entrenamiento y Validación 📈

![Accuracies de Entrenamiento y Validación](./results/beans/accuracies.png)




## Conclusiones

[Las conclusiones detalladas se añadirán una vez se hayan analizado todos los resultados experimentales, enfocándose en la relación entre parámetros entrenables y rendimiento]
