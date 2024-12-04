import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torchmetrics
from typing import Optional, Dict, Any

class ResNetTransferLearning(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 3e-5,
        weights: str = "IMAGENET1K_V1",
        freeze_blocks: int = 0
    ):
        """
        Args:
            num_classes: Número de clases para la clasificación
            learning_rate: Tasa de aprendizaje para el optimizador
            weights: Pesos pre-entrenados a utilizar
            freeze_blocks: Número de bloques a congelar (0 para full finetuning)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Inicializar el modelo ResNet18 pre-entrenado
        self.backbone = models.resnet18(weights=weights)
        
        # Obtener el número de características del modelo
        num_filters = self.backbone.fc.in_features
        
        # Reemplazar la capa de clasificación
        self.backbone.fc = nn.Linear(num_filters, num_classes)
        
        # Congelar bloques según el argumento freeze_blocks
        self.freeze_blocks(freeze_blocks)
        
        self.learning_rate = learning_rate
        
        # Definir función de pérdida
        self.criterion = nn.CrossEntropyLoss()
        
        # Definir métricas
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def freeze_blocks(self, num_blocks):
        """Congela los bloques especificados de la ResNet18"""
        blocks = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        
        # Congelar los bloques especificados
        for i in range(num_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calcular accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calcular accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calcular accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)
        
        # Logging
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        # Tasa de aprendizaje única para todo el modelo
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
