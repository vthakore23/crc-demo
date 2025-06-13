#!/usr/bin/env python3
"""
EPOC Cluster Training Script - Focused Version
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import yaml
from tqdm import tqdm

class EPOCTrainer:
    """Focused EPOC trainer for cluster deployment"""
    
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Setup distributed training
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')
        
        # Initialize model
        self.model = self.create_model()
        self.model = DDP(self.model, device_ids=[rank])
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_model(self):
        """Create enhanced molecular predictor"""
        from models.enhanced_molecular_predictor import EnhancedMolecularPredictor
        return EnhancedMolecularPredictor(num_classes=3).to(self.device)
    
    def load_data(self):
        """Load EPOC data"""
        data_path = Path(self.config['data_path'])
        manifest = pd.read_csv(data_path / 'epoc_manifest.csv')
        
        # Split data
        train_df = manifest[manifest['split'] == 'train']
        val_df = manifest[manifest['split'] == 'val']
        
        train_dataset = EPOCDataset(train_df)
        val_dataset = EPOCDataset(val_df)
        
        train_sampler = DistributedSampler(train_dataset, rank=self.rank)
        val_sampler = DistributedSampler(val_dataset, rank=self.rank)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 32),
            sampler=train_sampler,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32), 
            sampler=val_sampler,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train(self):
        """Training loop"""
        train_loader, val_loader = self.load_data()
        
        for epoch in range(self.config.get('num_epochs', 100)):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            if self.rank == 0:
                train_acc = 100. * correct / total
                self.logger.info(
                    f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, '
                    f'Val Acc: {val_acc:.2f}%'
                )
    
    def validate(self, val_loader):
        """Validation"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return val_loss / len(val_loader), 100. * correct / total


class EPOCDataset(Dataset):
    """Simple EPOC dataset"""
    
    def __init__(self, manifest_df):
        self.manifest_df = manifest_df
        self.label_map = {'Canonical': 0, 'Immune': 1, 'Stromal': 2}
    
    def __len__(self):
        return len(self.manifest_df)
    
    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        
        # Load dummy image for now
        image = torch.randn(3, 256, 256)
        label = self.label_map[row['molecular_subtype']]
        
        return image, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    trainer = EPOCTrainer(config, rank, world_size)
    trainer.train()


if __name__ == "__main__":
    main() 