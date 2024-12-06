import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import argparse

def export_tensorboard_data(log_dir, output_dir):
    """
    Exporta los datos de TensorBoard a archivos CSV.

    Args:
        log_dir (str): Directorio con los logs de TensorBoard
        output_dir (str): Directorio donde guardar los archivos CSV exportados
    """
    # Inicializar el acumulador de eventos
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Obtener todas las etiquetas escalares
    tags = ea.Tags()['scalars']
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Exportar cada métrica a un archivo CSV separado
    for tag in tags:
        # Obtener los eventos para esta etiqueta
        events = ea.Scalars(tag)
        
        # Convertir a DataFrame
        df = pd.DataFrame([(e.wall_time, e.step, e.value) for e in events],
                         columns=['timestamp', 'step', 'value'])
        
        # Guardar a CSV
        safe_tag = tag.replace('/', '_')
        output_file = os.path.join(output_dir, f'{safe_tag}.csv')
        df.to_csv(output_file, index=False)
        print(f'Exportado {tag} a {output_file}')

def process_beans_directory(base_dir, output_base_dir):
    """
    Procesa todos los subdirectorios de experimentos beans para exportar sus logs.

    Args:
        base_dir (str): Directorio base que contiene los experimentos
        output_base_dir (str): Directorio base donde guardar las exportaciones
    """
    # Iterar sobre todos los subdirectorios en el directorio beans
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # Verificar que sea un directorio
        if os.path.isdir(subdir_path):
            # Buscar el directorio de logs de tensorboard
            tensorboard_dir = None
            for root, dirs, files in os.walk(subdir_path):
                if any(f.startswith('events.out.tfevents') for f in files):
                    tensorboard_dir = root
                    break
            
            if tensorboard_dir:
                # Crear directorio de salida específico para este experimento
                output_dir = os.path.join(output_base_dir, subdir, 'tensorboard_exports')
                print(f'Procesando experimento: {subdir}')
                export_tensorboard_data(tensorboard_dir, output_dir)
            else:
                print(f'No se encontraron logs de TensorBoard en {subdir_path}')

def main():
    parser = argparse.ArgumentParser(description='Exportar datos de TensorBoard a CSV')
    parser.add_argument('--beans_dir', type=str, required=True,
                       help='Directorio base de experimentos beans')
    parser.add_argument('--output', type=str, required=True,
                       help='Directorio base de salida para los archivos CSV')
    
    args = parser.parse_args()
    process_beans_directory(args.beans_dir, args.output)

if __name__ == '__main__':
    main() 