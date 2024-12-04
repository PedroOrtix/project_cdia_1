import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import glob

def load_results(results_dir):
    """Cargar los resultados del entrenamiento desde el archivo JSON más reciente"""
    results_files = glob.glob(str(Path(results_dir) / 'training_results_*.json'))
    if not results_files:
        raise FileNotFoundError("No se encontraron archivos de resultados")
    
    latest_results_file = max(results_files)
    
    with open(latest_results_file, 'r') as f:
        results = json.load(f)
    
    return results

def load_tensorboard_data(log_dir):
    """Cargar datos de TensorBoard para un experimento"""
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0
        }
    )
    ea.Reload()
    
    metrics = {}
    for tag in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            metrics[tag] = {
                'values': [event.value for event in events],
                'steps': [event.step for event in events]
            }
    
    return metrics

def plot_metric_comparison(all_metrics, metric_name, title, output_dir):
    """Generar gráfico comparativo para una métrica específica"""
    plt.figure(figsize=(12, 6))
    
    for freeze_blocks, metrics in all_metrics.items():
        if metric_name in metrics:
            plt.plot(
                metrics[metric_name]['steps'],
                metrics[metric_name]['values'],
                label=f'Bloques Congelados: {freeze_blocks}'
            )
    
    plt.title(title)
    plt.xlabel('Pasos')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    # Guardar gráfico
    plt.savefig(output_dir / f"{metric_name}_comparison.png")
    plt.close()

def plot_training_times(results, output_dir):
    """Generar gráfico de tiempos de entrenamiento"""
    training_times = pd.DataFrame([
        {
            'freeze_blocks': r['freeze_blocks'],
            'training_time': r['training_time'] / 3600  # Convertir a horas
        }
        for r in results
    ])

    plt.figure(figsize=(10, 5))
    sns.barplot(data=training_times, x='freeze_blocks', y='training_time')
    plt.title('Tiempo de Entrenamiento por Configuración')
    plt.xlabel('Número de Bloques Congelados')
    plt.ylabel('Tiempo de Entrenamiento (horas)')
    plt.grid(True)
    
    plt.savefig(output_dir / "training_times.png")
    plt.close()

def generate_summary(all_metrics, results):
    """Generar resumen de los mejores resultados"""
    def get_best_metrics(metrics):
        return {
            'mejor_val_acc': max(metrics['val_acc']['values']) if 'val_acc' in metrics else None,
            'mejor_val_loss': min(metrics['val_loss']['values']) if 'val_loss' in metrics else None
        }

    summary = []
    for freeze_blocks, metrics in all_metrics.items():
        best_metrics = get_best_metrics(metrics)
        training_time = next(
            r['training_time'] for r in results 
            if r['freeze_blocks'] == freeze_blocks
        )
        
        summary.append({
            'freeze_blocks': freeze_blocks,
            'mejor_accuracy': best_metrics['mejor_val_acc'],
            'mejor_loss': best_metrics['mejor_val_loss'],
            'tiempo_entrenamiento': training_time / 3600
        })

    return pd.DataFrame(summary)

def main():
    # Configurar directorios
    results_dir = Path('results/animals')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar resultados
    results = load_results(results_dir)
    print(f"Cargados {len(results)} experimentos")
    
    # Cargar métricas de TensorBoard
    all_metrics = {}
    for result in results:
        freeze_blocks = result['freeze_blocks']
        log_dir = Path(result['config']['output_dir']) / 'logs'
        try:
            metrics = load_tensorboard_data(str(log_dir))
            all_metrics[freeze_blocks] = metrics
        except Exception as e:
            print(f"Error cargando métricas para freeze_blocks={freeze_blocks}: {str(e)}")
    
    # Generar gráficos
    for metric in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
        title_map = {
            'train_loss': 'Comparación de Pérdida en Entrenamiento',
            'val_loss': 'Comparación de Pérdida en Validación',
            'train_acc': 'Comparación de Accuracy en Entrenamiento',
            'val_acc': 'Comparación de Accuracy en Validación'
        }
        plot_metric_comparison(all_metrics, metric, title_map[metric], plots_dir)
    
    # Generar gráfico de tiempos de entrenamiento
    plot_training_times(results, plots_dir)
    
    # Generar resumen
    summary_df = generate_summary(all_metrics, results)
    summary_df = summary_df.sort_values('mejor_accuracy', ascending=False)
    
    # Encontrar la mejor configuración
    best_config = summary_df.iloc[0]
    
    # Generar markdown con resultados
    results_md = f"""# Resultados del Experimento

## Comparación de Configuraciones

{summary_df.to_markdown(index=False)}

## Mejor Configuración

- **Bloques Congelados**: {best_config['freeze_blocks']}
- **Mejor Accuracy**: {best_config['mejor_accuracy']:.4f}
- **Mejor Loss**: {best_config['mejor_loss']:.4f}
- **Tiempo de Entrenamiento**: {best_config['tiempo_entrenamiento']:.2f} horas

## Gráficos

### Métricas de Entrenamiento
![Train Loss](plots/train_loss_comparison.png)
![Train Accuracy](plots/train_acc_comparison.png)

### Métricas de Validación
![Val Loss](plots/val_loss_comparison.png)
![Val Accuracy](plots/val_acc_comparison.png)

### Tiempos de Entrenamiento
![Training Times](plots/training_times.png)
"""
    
    # Guardar resultados
    with open(results_dir / 'experiment_results.md', 'w') as f:
        f.write(results_md)
    
    print("\nAnálisis completado:")
    print(f"- Gráficos guardados en: {plots_dir}")
    print(f"- Resumen guardado en: {results_dir / 'experiment_results.md'}")

if __name__ == "__main__":
    main() 