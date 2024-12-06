import sys
import json
from pathlib import Path
import time
from datetime import datetime

# Añadir el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.fine_tuning_function import fine_tune_resnet

def train_model_with_config(freeze_blocks, base_config):
    """Entrena un modelo con una configuración específica de freeze_blocks"""
    # Crear directorio específico para este experimento
    experiment_name = f"freeze_blocks_{freeze_blocks}"
    output_dir = Path(base_config["output_dir"]) / experiment_name
    
    # Actualizar configuración
    config = base_config.copy()
    config["freeze_blocks"] = freeze_blocks
    config["output_dir"] = str(output_dir)
    
    # Registrar tiempo de inicio
    start_time = time.time()
    
    # Entrenar modelo
    best_model_path = fine_tune_resnet(**config)
    
    # Calcular tiempo total de entrenamiento
    training_time = time.time() - start_time
    
    return {
        "freeze_blocks": freeze_blocks,
        "best_model_path": str(best_model_path),
        "training_time": training_time,
        "config": config
    }

def main():
    # Configuración base
    base_config = {
        "num_classes": 3,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "max_epochs": 30,
        "num_workers": 4,
        "output_dir": "results/beans",
    }
    
    # Lista de configuraciones de freeze_blocks a probar
    freeze_blocks_configs = [1, 2, 3, 4]
    
    # Crear directorio para resultados si no existe
    results_dir = Path(base_config["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Entrenar modelos y guardar resultados
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for freeze_blocks in freeze_blocks_configs:
        print(f"\n{'='*50}")
        print(f"Entrenando modelo con freeze_blocks={freeze_blocks}")
        print(f"{'='*50}\n")
        
        try:
            result = train_model_with_config(freeze_blocks, base_config)
            results.append(result)
            
            # Guardar resultados parciales después de cada entrenamiento
            results_file = results_dir / f"training_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
                
            print(f"\nResultados guardados en: {results_file}")
            
        except Exception as e:
            print(f"Error entrenando modelo con freeze_blocks={freeze_blocks}: {str(e)}")
            continue
    
    print("\nEntrenamiento completado para todas las configuraciones.")
    print(f"Resultados finales guardados en: {results_file}")

if __name__ == "__main__":
    main() 