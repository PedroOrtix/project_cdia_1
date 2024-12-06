import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from lightning_modules.restnet_module import ResNetTransferLearning
from utils.data_loader import get_datasets

def fine_tune_resnet(
    num_classes: int = 90,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    max_epochs: int = 30,
    num_workers: int = 4,
    output_dir: str = 'results',
    freeze_blocks: int = 0
):
    pl.seed_everything(42)

    # Crear directorios de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preparar datasets y dataloaders
    train_dataset, val_dataset, _ = get_datasets()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Inicializar modelo
    model = ResNetTransferLearning(
        num_classes=num_classes,
        learning_rate=learning_rate,
        freeze_blocks=freeze_blocks
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
        max_epochs=max_epochs,
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

    return best_model_path 