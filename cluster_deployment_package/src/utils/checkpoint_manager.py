"""
Fault-Tolerant Checkpoint Manager for Distributed Training
Handles checkpoint saving, loading, and recovery with compression and integrity checks
"""

import os
import json
import time
import hashlib
import shutil
import gzip
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import asyncio

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files"""
    epoch: int
    step: int
    timestamp: float
    metrics: Dict[str, float]
    model_config: Dict[str, Any]
    file_size: int
    checksum: str
    compression: str
    distributed_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)


class FaultTolerantCheckpointManager:
    """Advanced checkpoint manager with fault tolerance and compression"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 10,
        compression: str = "gzip",
        compression_level: int = 6,
        async_save: bool = True,
        verify_integrity: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.compression_level = compression_level
        self.async_save = async_save
        self.verify_integrity = verify_integrity
        
        # Checkpoint tracking
        self.checkpoints = self._load_checkpoint_registry()
        self.best_checkpoints = {}
        self.save_lock = threading.Lock()
        
        # Background save queue for async operations
        if async_save:
            self.save_queue = asyncio.Queue()
            self.save_task = None
            
        logger.info(f"Initialized checkpoint manager at {checkpoint_dir}")
        
    def _load_checkpoint_registry(self) -> Dict[str, CheckpointMetadata]:
        """Load existing checkpoint registry"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        if not registry_path.exists():
            return {}
            
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
                
            checkpoints = {}
            for checkpoint_id, metadata_dict in registry_data.items():
                checkpoints[checkpoint_id] = CheckpointMetadata.from_dict(metadata_dict)
                
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint registry: {str(e)}")
            return {}
            
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        try:
            registry_data = {}
            for checkpoint_id, metadata in self.checkpoints.items():
                registry_data[checkpoint_id] = metadata.to_dict()
                
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint registry: {str(e)}")
            
    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data"""
        return hashlib.sha256(data).hexdigest()
        
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using specified compression algorithm"""
        if self.compression == "gzip":
            return gzip.compress(data, compresslevel=self.compression_level)
        elif self.compression == "none":
            return data
        else:
            raise ValueError(f"Unsupported compression: {self.compression}")
            
    def _decompress_data(self, data: bytes, compression: str) -> bytes:
        """Decompress data"""
        if compression == "gzip":
            return gzip.decompress(data)
        elif compression == "none":
            return data
        else:
            raise ValueError(f"Unsupported compression: {compression}")
            
    def _serialize_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bytes:
        """Serialize checkpoint data to bytes"""
        return pickle.dumps(checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL)
        
    def _deserialize_checkpoint(self, data: bytes) -> Dict[str, Any]:
        """Deserialize checkpoint data from bytes"""
        return pickle.loads(data)
        
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        checkpoint_id: Optional[str] = None
    ) -> str:
        """Save checkpoint with fault tolerance"""
        
        if metrics is None:
            metrics = {}
            
        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            timestamp = int(time.time())
            checkpoint_id = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}"
            
        # Prepare checkpoint data
        full_checkpoint_data = {
            'checkpoint_data': checkpoint_data,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'timestamp': time.time(),
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__
        }
        
        if self.async_save:
            # Queue for async saving
            asyncio.create_task(self._async_save_checkpoint(
                checkpoint_id, full_checkpoint_data, epoch, step, metrics, is_best
            ))
        else:
            # Synchronous saving
            self._sync_save_checkpoint(
                checkpoint_id, full_checkpoint_data, epoch, step, metrics, is_best
            )
            
        return checkpoint_id
        
    def _sync_save_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool
    ):
        """Synchronously save checkpoint"""
        with self.save_lock:
            try:
                # Serialize checkpoint
                serialized_data = self._serialize_checkpoint(checkpoint_data)
                
                # Compress data
                compressed_data = self._compress_data(serialized_data)
                
                # Compute checksum
                checksum = self._compute_checksum(compressed_data)
                
                # Write to temporary file first
                temp_path = self.checkpoint_dir / f"{checkpoint_id}.tmp"
                final_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
                
                with open(temp_path, 'wb') as f:
                    f.write(compressed_data)
                    
                # Verify integrity
                if self.verify_integrity:
                    with open(temp_path, 'rb') as f:
                        read_data = f.read()
                        
                    if self._compute_checksum(read_data) != checksum:
                        raise ValueError("Checkpoint integrity check failed")
                        
                # Atomic move to final location
                shutil.move(str(temp_path), str(final_path))
                
                # Create metadata
                metadata = CheckpointMetadata(
                    epoch=epoch,
                    step=step,
                    timestamp=time.time(),
                    metrics=metrics,
                    model_config=checkpoint_data.get('config', {}),
                    file_size=len(compressed_data),
                    checksum=checksum,
                    compression=self.compression,
                    distributed_info={
                        'world_size': os.environ.get('WORLD_SIZE', '1'),
                        'rank': os.environ.get('RANK', '0')
                    }
                )
                
                # Update registry
                self.checkpoints[checkpoint_id] = metadata
                
                # Handle best checkpoint
                if is_best:
                    self._update_best_checkpoint(checkpoint_id, metrics)
                    
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                # Save registry
                self._save_checkpoint_registry()
                
                logger.info(f"Saved checkpoint {checkpoint_id} (epoch {epoch}, step {step})")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint {checkpoint_id}: {str(e)}")
                
                # Cleanup temporary file
                if temp_path.exists():
                    temp_path.unlink()
                    
                raise
                
    async def _async_save_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool
    ):
        """Asynchronously save checkpoint"""
        # Run synchronous save in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_save_checkpoint,
            checkpoint_id,
            checkpoint_data,
            epoch,
            step,
            metrics,
            is_best
        )
        
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint by ID"""
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint {checkpoint_id} not found in registry")
            return None
            
        metadata = self.checkpoints[checkpoint_id]
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
            
        try:
            # Read compressed data
            with open(checkpoint_path, 'rb') as f:
                compressed_data = f.read()
                
            # Verify integrity
            if self.verify_integrity:
                checksum = self._compute_checksum(compressed_data)
                if checksum != metadata.checksum:
                    logger.error(f"Checkpoint integrity check failed for {checkpoint_id}")
                    return None
                    
            # Decompress data
            serialized_data = self._decompress_data(compressed_data, metadata.compression)
            
            # Deserialize checkpoint
            checkpoint_data = self._deserialize_checkpoint(serialized_data)
            
            logger.info(f"Loaded checkpoint {checkpoint_id} (epoch {metadata.epoch})")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {str(e)}")
            return None
            
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint ID"""
        if not self.checkpoints:
            return None
            
        # Sort by timestamp
        latest_checkpoint = max(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp
        )
        
        return latest_checkpoint[0]
        
    def get_best_checkpoint(self, metric: str = "val_f1_macro") -> Optional[str]:
        """Get the best checkpoint for a specific metric"""
        return self.best_checkpoints.get(metric)
        
    def _update_best_checkpoint(self, checkpoint_id: str, metrics: Dict[str, float]):
        """Update best checkpoint tracking"""
        for metric, value in metrics.items():
            if metric.startswith('val_'):
                current_best = self.best_checkpoints.get(metric)
                
                if current_best is None or value > self.checkpoints[current_best].metrics.get(metric, -float('inf')):
                    self.best_checkpoints[metric] = checkpoint_id
                    
                    # Create symlink for convenience
                    best_link = self.checkpoint_dir / f"best_{metric}.ckpt"
                    if best_link.exists():
                        best_link.unlink()
                        
                    try:
                        best_link.symlink_to(f"{checkpoint_id}.ckpt")
                    except OSError:
                        # Fallback to copying if symlinks not supported
                        shutil.copy2(
                            self.checkpoint_dir / f"{checkpoint_id}.ckpt",
                            best_link
                        )
                        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
            
        # Sort checkpoints by timestamp
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Keep best checkpoints
        protected_checkpoints = set(self.best_checkpoints.values())
        
        # Remove oldest checkpoints (but protect best ones)
        checkpoints_to_remove = []
        for checkpoint_id, metadata in sorted_checkpoints:
            if checkpoint_id not in protected_checkpoints:
                checkpoints_to_remove.append(checkpoint_id)
                
            # Stop if we have enough to remove
            if len(self.checkpoints) - len(checkpoints_to_remove) <= self.max_checkpoints:
                break
                
        # Remove checkpoint files and registry entries
        for checkpoint_id in checkpoints_to_remove:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                
            del self.checkpoints[checkpoint_id]
            logger.info(f"Removed old checkpoint {checkpoint_id}")
            
    def list_checkpoints(self) -> List[Tuple[str, CheckpointMetadata]]:
        """List all available checkpoints"""
        return sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint"""
        return self.checkpoints.get(checkpoint_id)
        
    def export_checkpoint(self, checkpoint_id: str, export_path: str):
        """Export checkpoint to a different location"""
        checkpoint_data = self.load_checkpoint(checkpoint_id)
        if checkpoint_data is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
        # Save in standard PyTorch format
        torch.save(checkpoint_data, export_path)
        logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")
        
    def import_checkpoint(
        self,
        import_path: str,
        checkpoint_id: Optional[str] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Import checkpoint from external file"""
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"Import file not found: {import_path}")
            
        # Load checkpoint data
        checkpoint_data = torch.load(import_path, map_location='cpu')
        
        # Extract metadata if available
        if epoch is None:
            epoch = checkpoint_data.get('epoch', 0)
            
        if metrics is None:
            metrics = checkpoint_data.get('metrics', {})
            
        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            timestamp = int(time.time())
            checkpoint_id = f"imported_checkpoint_{epoch}_{timestamp}"
            
        # Save using our format
        return self.save_checkpoint(
            checkpoint_data,
            epoch=epoch,
            metrics=metrics,
            checkpoint_id=checkpoint_id
        )
        
    def verify_all_checkpoints(self) -> Dict[str, bool]:
        """Verify integrity of all checkpoints"""
        results = {}
        
        for checkpoint_id, metadata in self.checkpoints.items():
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
            
            if not checkpoint_path.exists():
                results[checkpoint_id] = False
                continue
                
            try:
                with open(checkpoint_path, 'rb') as f:
                    data = f.read()
                    
                checksum = self._compute_checksum(data)
                results[checkpoint_id] = (checksum == metadata.checksum)
                
            except Exception:
                results[checkpoint_id] = False
                
        return results
        
    def cleanup_corrupted_checkpoints(self):
        """Remove corrupted checkpoints"""
        verification_results = self.verify_all_checkpoints()
        
        corrupted_checkpoints = [
            checkpoint_id for checkpoint_id, is_valid in verification_results.items()
            if not is_valid
        ]
        
        for checkpoint_id in corrupted_checkpoints:
            logger.warning(f"Removing corrupted checkpoint {checkpoint_id}")
            
            # Remove file if it exists
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                
            # Remove from registry
            if checkpoint_id in self.checkpoints:
                del self.checkpoints[checkpoint_id]
                
            # Remove from best checkpoints
            for metric, best_id in list(self.best_checkpoints.items()):
                if best_id == checkpoint_id:
                    del self.best_checkpoints[metric]
                    
        # Save updated registry
        self._save_checkpoint_registry()
        
        return len(corrupted_checkpoints)
        
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage usage information"""
        total_size = 0
        checkpoint_sizes = {}
        
        for checkpoint_id, metadata in self.checkpoints.items():
            total_size += metadata.file_size
            checkpoint_sizes[checkpoint_id] = metadata.file_size
            
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'num_checkpoints': len(self.checkpoints),
            'checkpoint_sizes': checkpoint_sizes,
            'directory': str(self.checkpoint_dir)
        } 