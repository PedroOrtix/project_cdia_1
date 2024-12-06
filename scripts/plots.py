import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import re
import os

# Configuración global de seaborn
sns.set_theme(style="whitegrid")

def extract_block_number(exp_name):
    """Extrae el número de bloques del nombre del experimento"""
    match = re.search(r'freeze_blocks_(\d+)', exp_name)
    return int(match.group(1)) if match else float('inf')

# 1. Gráfico de barras agrupado de accuracy train vs val
def plot_accuracies(datos_experimentos):
    """
    Genera un gráfico de barras comparando accuracy de train vs validación.

    Args:
        datos_experimentos (dict): Diccionario con datos de los experimentos
    """
    accuracies_data = []
    # Ordenar los experimentos por número de bloques
    sorted_experiments = sorted(datos_experimentos.items(), 
                              key=lambda x: extract_block_number(x[0]))
    
    for exp_name, exp_data in sorted_experiments:
        # Tomamos el último valor de cada métrica
        train_acc = exp_data['train_acc_epoch']['value'].iloc[-1]
        val_acc = exp_data['val_acc']['value'].iloc[-1]
        
        accuracies_data.append({
            'Experimento': exp_name,
            'Train': train_acc,
            'Validación': val_acc
        })
    
    df_acc = pd.DataFrame(accuracies_data)
    df_melted = df_acc.melt(id_vars=['Experimento'], var_name='Tipo', value_name='Accuracy')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Experimento', y='Accuracy', hue='Tipo')
    plt.title('Accuracy de Entrenamiento vs Validación por Experimento')
    plt.xticks(rotation=45)
    plt.tight_layout()

# 2. Parámetros entrenables vs tiempo de entrenamiento
def plot_params_vs_time(datos_experimentos):
    """
    Genera gráficos comparando parámetros entrenables y tiempo de entrenamiento.

    Args:
        datos_experimentos (dict): Diccionario con datos de los experimentos
    """
    # Cargar resultados del JSON
    with open('results/beans/training_results_20241206_030322.json', 'r') as f:
        training_results = json.load(f)
    
    data = []
    # Ordenar los experimentos por número de bloques
    sorted_experiments = sorted(datos_experimentos.items(), 
                              key=lambda x: extract_block_number(x[0]))
    
    for exp_name, exp_data in sorted_experiments:
        trainable_params = exp_data['model_trainable_parameters']['value'].iloc[0]
        training_time = next(
            result['training_time'] 
            for result in training_results 
            if f"freeze_blocks_{result['freeze_blocks']}" == exp_name
        )
        
        data.append({
            'Experimento': exp_name,
            'Parámetros Entrenables (M)': trainable_params / 1e6,
            'Tiempo de Entrenamiento (min)': training_time / 60
        })
    
    df_params = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de barras para parámetros
    sns.barplot(data=df_params, x='Experimento', y='Parámetros Entrenables (M)', ax=ax1)
    ax1.set_title('Parámetros Entrenables por Experimento')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Gráfico de barras para tiempo
    sns.barplot(data=df_params, x='Experimento', y='Tiempo de Entrenamiento (min)', ax=ax2)
    ax2.set_title('Tiempo de Entrenamiento por Experimento')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()

# 3. Análisis de eficiencia (reemplaza plot_params_accuracy_ratio)
def plot_efficiency_analysis(datos_experimentos):
    data = []
    for exp_name, exp_data in datos_experimentos.items():
        trainable_params = exp_data['model_trainable_parameters']['value'].iloc[0] / 1e6  # En millones
        val_acc = exp_data['val_acc']['value'].iloc[-1] * 100  # En porcentaje
        
        data.append({
            'Experimento': exp_name,
            'Parámetros (M)': trainable_params,
            'Accuracy (%)': val_acc,
        })
    
    df_efficiency = pd.DataFrame(data)
    
    # Crear una figura con dos subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de líneas para ver la relación
    sns.lineplot(data=df_efficiency, x='Parámetros (M)', y='Accuracy (%)', 
                marker='o', markersize=10, ax=ax1)
    ax1.set_title('Accuracy vs Parámetros Entrenables')
    
    # Tabla con los valores numéricos - aumentamos la precisión a 3 decimales
    table_data = [[f"{row['Experimento']}\n{row['Parámetros (M)']:.3f}M params\n{row['Accuracy (%)']:.1f}% acc"] 
                  for _, row in df_efficiency.iterrows()]
    ax2.table(cellText=table_data, 
              colLabels=['Métricas por Experimento'],
              cellLoc='center',
              loc='center',
              bbox=[0.1, 0.1, 0.8, 0.8])
    ax2.axis('off')
    
    plt.tight_layout()

# 4. Progreso del loss train vs val
def plot_loss_progress(datos_experimentos):
    plt.figure(figsize=(12, 6))
    
    for exp_name, exp_data in datos_experimentos.items():
        train_loss = exp_data['train_loss_epoch']
        val_loss = exp_data['val_loss']
        
        plt.plot(train_loss['step'], train_loss['value'], 
                label=f'{exp_name} (train)', linestyle='--')
        plt.plot(val_loss['step'], val_loss['value'], 
                label=f'{exp_name} (val)', linestyle='-')
    
    plt.title('Progreso del Loss durante el Entrenamiento')
    plt.xlabel('Paso')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

# 5. Análisis de velocidad de convergencia
def plot_convergence_analysis(datos_experimentos):
    """Análisis de velocidad de convergencia"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Ordenar los experimentos por número de bloques
    sorted_experiments = sorted(datos_experimentos.items(), 
                              key=lambda x: extract_block_number(x[0]))
    
    # Para cada experimento, encontrar en qué época se alcanza 90% de accuracy
    convergence_data = []
    for exp_name, exp_data in sorted_experiments:
        val_acc = exp_data['val_acc']['value']
        steps = exp_data['val_acc']['step']
        
        # Encontrar primera época que supera 0.9 de accuracy
        convergence_epoch = next((step for step, acc in zip(steps, val_acc) if acc > 0.9), None)
        
        convergence_data.append({
            'Experimento': exp_name,
            'Épocas hasta 90%': convergence_epoch if convergence_epoch else max(steps),
            'Accuracy Final': val_acc.iloc[-1],
            'Estabilidad': val_acc.std()  # Desviación estándar como medida de estabilidad
        })
    
    df_convergence = pd.DataFrame(convergence_data)
    
    # Gráfico de épocas hasta convergencia
    sns.barplot(data=df_convergence, x='Experimento', y='Épocas hasta 90%', ax=ax1)
    ax1.set_title('Velocidad de Convergencia')
    ax1.set_ylabel('Épocas hasta 90% accuracy')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Gráfico de estabilidad
    sns.barplot(data=df_convergence, x='Experimento', y='Estabilidad', ax=ax2)
    ax2.set_title('Estabilidad del Entrenamiento')
    ax2.set_ylabel('Desviación Estándar del Accuracy')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()

# 6. Análisis de overfitting
def plot_overfitting_analysis(datos_experimentos):
    """Análisis de overfitting"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Ordenar los experimentos por número de bloques
    sorted_experiments = sorted(datos_experimentos.items(), 
                              key=lambda x: extract_block_number(x[0]))
    
    overfitting_metrics = []
    for exp_name, exp_data in sorted_experiments:
        train_acc = exp_data['train_acc_epoch']['value']
        val_acc = exp_data['val_acc']['value']
        
        # Calcular gap entre train y val
        train_val_gap = train_acc.iloc[-1] - val_acc.iloc[-1]
        
        # Calcular tendencia de val_loss en últimas épocas
        val_loss = exp_data['val_loss']['value']
        loss_trend = val_loss.iloc[-5:].mean() - val_loss.iloc[-10:-5].mean()
        
        overfitting_metrics.append({
            'Experimento': exp_name,
            'Train-Val Gap': train_val_gap,
            'Tendencia Loss': loss_trend
        })
    
    df_overfitting = pd.DataFrame(overfitting_metrics)
    
    # Gráfico de diferencia train-val
    sns.barplot(data=df_overfitting, x='Experimento', y='Train-Val Gap', ax=ax1)
    ax1.set_title('Gap entre Train y Validación')
    ax1.set_ylabel('Diferencia de Accuracy')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Gráfico de tendencia del loss
    sns.barplot(data=df_overfitting, x='Experimento', y='Tendencia Loss', ax=ax2)
    ax2.set_title('Tendencia del Loss en Últimas Épocas')
    ax2.set_ylabel('Δ Loss (+ indica overfitting)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()

def plot_test_results(csv_path='results/beans/test_evaluation_results.csv'):
    """
    Visualiza los resultados de evaluación en el conjunto de test.

    Args:
        csv_path (str, optional): Ruta al CSV con resultados. 
            Por defecto 'results/beans/test_evaluation_results.csv'

    Returns:
        Figure: Figura de matplotlib con los gráficos generados
    """
    # Leer los datos
    df = pd.read_csv(csv_path)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de barras
    sns.barplot(data=df, x='freeze_blocks', y='test_accuracy', ax=ax1)
    ax1.set_title('Precisión en Test por Bloques Congelados')
    ax1.set_xlabel('Número de Bloques Congelados')
    ax1.set_ylabel('Precisión en Test')
    
    # Tabla con valores numéricos
    table_data = [[f"Bloques {row['freeze_blocks']}", f"{row['test_accuracy']:.4f}"] 
                  for _, row in df.iterrows()]
    
    ax2.table(cellText=table_data,
              colLabels=['Configuración', 'Accuracy'],
              cellLoc='center',
              loc='center',
              bbox=[0.1, 0.1, 0.8, 0.8])
    ax2.axis('off')
    ax2.set_title('Resultados Detallados')
    
    plt.tight_layout()
    return fig

# Para ejecutar todas las gráficas:
def plot_all(datos_experimentos):
    plot_accuracies(datos_experimentos)
    plot_params_vs_time(datos_experimentos)
    plot_efficiency_analysis(datos_experimentos)
    plot_loss_progress(datos_experimentos)
    plot_convergence_analysis(datos_experimentos)
    plot_overfitting_analysis(datos_experimentos)
    plot_test_results()
    plt.show()

def save_all_plots(datos_experimentos, output_dir='results/plots'):
    """
    Genera y guarda todos los plots en archivos individuales
    
    Args:
        datos_experimentos: Diccionario con los datos de los experimentos
        output_dir: Directorio donde se guardarán los plots
    """
    # Crear el directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plot_accuracies(datos_experimentos)
    plt.savefig(os.path.join(output_dir, 'accuracies.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Params vs Time plot
    plt.figure(figsize=(15, 5))
    plot_params_vs_time(datos_experimentos)
    plt.savefig(os.path.join(output_dir, 'params_vs_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Efficiency analysis
    plt.figure(figsize=(15, 5))
    plot_efficiency_analysis(datos_experimentos)
    plt.savefig(os.path.join(output_dir, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss progress
    plt.figure(figsize=(12, 6))
    plot_loss_progress(datos_experimentos)
    plt.savefig(os.path.join(output_dir, 'loss_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convergence analysis
    plt.figure(figsize=(15, 5))
    plot_convergence_analysis(datos_experimentos)
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Overfitting analysis
    plt.figure(figsize=(15, 5))
    plot_overfitting_analysis(datos_experimentos)
    plt.savefig(os.path.join(output_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Test results
    plt.figure(figsize=(15, 5))
    plot_test_results()
    plt.savefig(os.path.join(output_dir, 'test_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots guardados en: {output_dir}")