import sys
from pathlib import Path
import json
import torch
from tqdm import tqdm
import pandas as pd

# Añadir el directorio raíz al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lightning_modules.restnet_module import ResNetTransferLearning
from scripts.data_loader import get_datasets

def load_trained_model(checkpoint_path: str, num_classes: int = 3) -> ResNetTransferLearning:
    """Carga un modelo entrenado desde un checkpoint"""
    model = ResNetTransferLearning(num_classes=num_classes)
    # Añadir map_location para manejar la carga en CPU si no hay GPU disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def evaluate_model(model: ResNetTransferLearning, test_loader, device: str = 'cuda'):
    """Evalúa el modelo en el conjunto de test"""
    model = model.to(device)
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluando"):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
            
    return predictions, true_labels

def main():
    # Configuración
    results_dir = Path("results/beans")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar el dataset de test
    _, _, test_dataset = get_datasets()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Encontrar todos los checkpoints
    results = []
    for freeze_dir in results_dir.glob("freeze_blocks_*"):
        checkpoint_path = freeze_dir / "checkpoints" / "last.ckpt"
        if not checkpoint_path.exists():
            continue
            
        print(f"\nEvaluando modelo en: {checkpoint_path}")
        
        # Cargar y evaluar modelo
        model = load_trained_model(str(checkpoint_path))
        predictions, true_labels = evaluate_model(model, test_loader, device)
        
        # Calcular métricas
        accuracy = (torch.tensor(predictions) == torch.tensor(true_labels)).float().mean()
        
        # Guardar resultados
        result = {
            "freeze_blocks": int(freeze_dir.name.split("_")[-1]),
            "checkpoint_path": str(checkpoint_path),
            "test_accuracy": float(accuracy),
        }
        results.append(result)
        
        print(f"Precisión en test: {accuracy:.4f}")
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_accuracy', ascending=False)
    
    # Guardar como CSV
    output_file = results_dir / "test_evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResultados guardados en: {output_file}")
    
    # Mostrar resultados ordenados
    print("\nResumen de resultados:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main() 