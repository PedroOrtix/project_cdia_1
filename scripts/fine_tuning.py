import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from project.lightning_modules.restnet_module import ResNetTransferLearning
from utils.data_loader import get_datasets  # Asumiendo que tienes esta función implementada

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning de modelo ResNet para clasificación de animales')
    parser.add_argument('--data_dir', type=str, required=True, help='Directorio con las carpetas de clases de animales')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamaño del batch')
    parser.add_argument('--num_classes', type=int, default=90, help='Número de clases de animales')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Tasa de aprendizaje')
    parser.add_argument('--max_epochs', type=int, default=30, help='Número máximo de épocas')
    parser.add_argument('--num_workers', type=int, default=4, help='Número de workers para DataLoader')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directorio para guardar resultados')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proporción de datos para validación')
    parser.add_argument('--freeze_blocks', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Número de bloques a congelar (0 para full finetuning)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42)

    # Crear directorios de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preparar datasets y dataloaders
    train_dataset, val_dataset = get_datasets(args.data_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Inicializar modelo
    model = ResNetTransferLearning(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        freeze_blocks=args.freeze_blocks
    )

    # Configurar callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="resnet-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )

    # Configurar logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs"
    )

    # Configurar trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        precision=16,  # Usar precisión mixta para optimizar memoria y velocidad
        deterministic=True,
        log_every_n_steps=10,
    )

    # Entrenar modelo
    trainer.fit(model, train_loader, val_loader)

    # Guardar el mejor modelo
    best_model_path = checkpoint_callback.best_model_path
    print(f"Mejor modelo guardado en: {best_model_path}")

if __name__ == "__main__":
    main()
