import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import argparse

def export_tensorboard_data(log_dir, output_dir):
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

def main():
    parser = argparse.ArgumentParser(description='Exportar datos de TensorBoard a CSV')
    parser.add_argument('--logdir', type=str, required=True,
                       help='Directorio de logs de TensorBoard')
    parser.add_argument('--output', type=str, required=True,
                       help='Directorio de salida para los archivos CSV')
    
    args = parser.parse_args()
    export_tensorboard_data(args.logdir, args.output)

if __name__ == '__main__':
    main() 