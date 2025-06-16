#!/usr/bin/env python3
"""
Production Inference Pipeline for CRC Molecular Subtype Classification
Enterprise-grade scalable inference system with load balancing and quality control
"""

import os
import json
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import uuid

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import openslide
import ray
from ray import serve
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import redis

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.distributed_wrapper import DistributedModelWrapper
from src.data.wsi_dataset_distributed import DistributedPatchExtractor
from src.utils.monitoring import PerformanceProfiler
from src.validation.epoc_validator import EPOCValidator

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Inference request data structure"""
    request_id: str
    wsi_path: str
    slide_id: str
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=urgent
    metadata: Dict[str, Any] = None
    callback_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    """Inference result data structure"""
    request_id: str
    slide_id: str
    prediction: int
    confidence: float
    uncertainty: float
    class_probabilities: List[float]
    processing_time: float
    quality_metrics: Dict[str, Any]
    patch_count: int
    attention_weights: Optional[List[float]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelManager:
    """Manages model loading and inference across multiple GPUs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.model_lock = threading.Lock()
        self.device_pool = self._initialize_device_pool()
        
    def _initialize_device_pool(self) -> List[torch.device]:
        """Initialize pool of available devices"""
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device(f'cuda:{i}'))
        else:
            devices.append(torch.device('cpu'))
        
        logger.info(f"Initialized device pool with {len(devices)} devices")
        return devices
        
    def load_model(self, model_path: str, device: torch.device) -> torch.nn.Module:
        """Load model on specified device"""
        device_key = str(device)
        
        with self.model_lock:
            if device_key not in self.models:
                # Load model
                model_wrapper = DistributedModelWrapper(
                    model_name=self.config['model']['name'],
                    num_classes=self.config['model']['num_classes'],
                    config=self.config['model']
                )
                
                model = model_wrapper.get_model()
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                # Move to device and set to eval mode
                model = model.to(device)
                model.eval()
                
                self.models[device_key] = model
                logger.info(f"Loaded model on {device}")
                
            return self.models[device_key]
            
    def get_available_device(self) -> torch.device:
        """Get device with lowest current load"""
        # Simple round-robin selection
        # In production, you'd implement actual load monitoring
        return self.device_pool[len(self.models) % len(self.device_pool)]


@ray.remote
class DistributedInferenceWorker:
    """Ray actor for distributed inference processing"""
    
    def __init__(self, worker_id: int, config: Dict[str, Any], model_path: str):
        self.worker_id = worker_id
        self.config = config
        self.model_path = model_path
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.patch_extractor = DistributedPatchExtractor(config['dataset'])
        self.profiler = PerformanceProfiler()
        
        # Load model
        self.device = self.model_manager.get_available_device()
        self.model = self.model_manager.load_model(model_path, self.device)
        
        logger.info(f"Initialized inference worker {worker_id} on {self.device}")
        
    def process_slide(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single WSI slide"""
        try:
            request_obj = InferenceRequest(**request)
            
            self.profiler.start_timer('total_processing')
            self.profiler.start_timer('patch_extraction')
            
            # Extract patches
            patches_data = ray.get(
                self.patch_extractor.extract_patches.remote(
                    request_obj.wsi_path, request_obj.slide_id
                )
            )
            
            self.profiler.end_timer('patch_extraction')
            
            if patches_data is None:
                raise ValueError(f"Failed to extract patches from {request_obj.slide_id}")
                
            # Prepare batch data
            self.profiler.start_timer('data_preparation')
            batch_data = self._prepare_batch_data(patches_data)
            self.profiler.end_timer('data_preparation')
            
            # Run inference
            self.profiler.start_timer('model_inference')
            inference_output = self._run_inference(batch_data)
            self.profiler.end_timer('model_inference')
            
            # Post-process results
            self.profiler.start_timer('post_processing')
            result = self._post_process_results(
                request_obj, patches_data, inference_output
            )
            self.profiler.end_timer('post_processing')
            
            self.profiler.end_timer('total_processing')
            
            # Add performance metrics
            result.quality_metrics.update(self.profiler.get_performance_summary())
            
            logger.info(f"Processed slide {request_obj.slide_id} in {result.processing_time:.2f}s")
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to process slide {request.get('slide_id', 'unknown')}: {str(e)}")
            return {
                'request_id': request.get('request_id', ''),
                'slide_id': request.get('slide_id', ''),
                'error': str(e),
                'timestamp': time.time()
            }
            
    def _prepare_batch_data(self, patches_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare patch data for model input"""
        patches = []
        
        for patch_info in patches_data['patches']:
            patch = patch_info['patch']
            if isinstance(patch, Image.Image):
                patch_np = np.array(patch)
            else:
                patch_np = patch
                
            # Normalize
            patch_tensor = torch.from_numpy(patch_np).float() / 255.0
            patch_tensor = patch_tensor.permute(2, 0, 1)  # HWC -> CHW
            
            # Apply normalization (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            patch_tensor = (patch_tensor - mean) / std
            
            patches.append(patch_tensor)
            
        # Stack patches
        patches_tensor = torch.stack(patches).to(self.device)  # [N, C, H, W]
        
        return {
            'patches': patches_tensor.unsqueeze(0),  # Add batch dimension [1, N, C, H, W]
            'labels': torch.tensor([0], dtype=torch.long).to(self.device)  # Dummy label
        }
        
    def _run_inference(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run model inference"""
        with torch.no_grad():
            outputs = self.model(batch_data)
            
        return outputs
        
    def _post_process_results(
        self,
        request: InferenceRequest,
        patches_data: Dict[str, Any],
        model_output: Dict[str, torch.Tensor]
    ) -> InferenceResult:
        """Post-process model outputs into final result"""
        
        # Get predictions
        logits = model_output['logits'].squeeze(0)  # Remove batch dimension
        probabilities = F.softmax(logits, dim=0)
        prediction = torch.argmax(logits).item()
        confidence = torch.max(probabilities).item()
        
        # Get uncertainty
        if 'uncertainty' in model_output:
            uncertainty = model_output['uncertainty'].squeeze(0).item()
        else:
            # Compute entropy-based uncertainty
            uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()
            
        # Get attention weights if available
        attention_weights = None
        if 'mil_weights' in model_output:
            attention_weights = model_output['mil_weights'].squeeze(0).cpu().numpy().tolist()
            
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            patches_data, model_output, probabilities
        )
        
        # Calculate processing time
        processing_time = sum([
            self.profiler.timing_data.get('patch_extraction', [0])[-1],
            self.profiler.timing_data.get('data_preparation', [0])[-1],
            self.profiler.timing_data.get('model_inference', [0])[-1],
            self.profiler.timing_data.get('post_processing', [0])[-1]
        ])
        
        return InferenceResult(
            request_id=request.request_id,
            slide_id=request.slide_id,
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
            class_probabilities=probabilities.cpu().numpy().tolist(),
            processing_time=processing_time,
            quality_metrics=quality_metrics,
            patch_count=len(patches_data['patches']),
            attention_weights=attention_weights
        )
        
    def _compute_quality_metrics(
        self,
        patches_data: Dict[str, Any],
        model_output: Dict[str, torch.Tensor],
        probabilities: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute quality control metrics"""
        
        # Basic patch statistics
        patch_count = len(patches_data['patches'])
        
        # Quality scores from patch extraction
        if patches_data['patches']:
            blur_scores = [p.get('blur_score', 0) for p in patches_data['patches']]
            tissue_ratios = [p.get('tissue_ratio', 0) for p in patches_data['patches']]
            
            avg_blur_score = np.mean(blur_scores)
            avg_tissue_ratio = np.mean(tissue_ratios)
        else:
            avg_blur_score = 0.0
            avg_tissue_ratio = 0.0
            
        # Model confidence metrics
        prediction_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()
        max_probability = torch.max(probabilities).item()
        
        # Quality flags
        quality_flags = {
            'sufficient_patches': patch_count >= self.config.get('min_patches_per_slide', 10),
            'good_image_quality': avg_blur_score >= self.config.get('blur_threshold', 50),
            'sufficient_tissue': avg_tissue_ratio >= self.config.get('tissue_threshold', 0.15),
            'confident_prediction': max_probability >= 0.7,
            'low_uncertainty': prediction_entropy <= 1.0
        }
        
        # Overall quality score
        quality_score = sum(quality_flags.values()) / len(quality_flags)
        
        return {
            'patch_count': patch_count,
            'avg_blur_score': avg_blur_score,
            'avg_tissue_ratio': avg_tissue_ratio,
            'prediction_entropy': prediction_entropy,
            'max_probability': max_probability,
            'quality_flags': quality_flags,
            'quality_score': quality_score
        }


class InferenceOrchestrator:
    """Orchestrates distributed inference across multiple workers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.request_queue = queue.PriorityQueue()
        self.results_cache = {}
        self.active_requests = {}
        
        # Initialize Redis for result caching (optional)
        self.redis_client = None
        if config.get('use_redis', False):
            try:
                self.redis_client = redis.Redis(
                    host=config.get('redis_host', 'localhost'),
                    port=config.get('redis_port', 6379),
                    db=config.get('redis_db', 0)
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {str(e)}")
                
        # Initialize Ray workers
        self.workers = self._initialize_workers()
        
        # Start processing thread
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info(f"Initialized inference orchestrator with {len(self.workers)} workers")
        
    def _initialize_workers(self) -> List:
        """Initialize Ray workers for distributed inference"""
        num_workers = self.config.get('num_workers', 4)
        model_path = self.config.get('model_path', 'models/best_model.pth')
        
        workers = []
        for i in range(num_workers):
            worker = DistributedInferenceWorker.remote(i, self.config, model_path)
            workers.append(worker)
            
        return workers
        
    def submit_request(self, request: InferenceRequest) -> str:
        """Submit inference request"""
        # Priority queue: lower numbers = higher priority
        priority_score = -request.priority  # Negate for correct ordering
        
        self.request_queue.put((priority_score, time.time(), request))
        self.active_requests[request.request_id] = {
            'status': 'queued',
            'submitted_at': time.time(),
            'request': request
        }
        
        logger.info(f"Submitted request {request.request_id} for slide {request.slide_id}")
        return request.request_id
        
    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get inference result"""
        # Check local cache first
        if request_id in self.results_cache:
            return self.results_cache[request_id]
            
        # Check Redis cache
        if self.redis_client:
            try:
                cached_result = self.redis_client.get(f"result:{request_id}")
                if cached_result:
                    result = json.loads(cached_result)
                    self.results_cache[request_id] = result
                    return result
            except Exception as e:
                logger.error(f"Failed to retrieve from Redis: {str(e)}")
                
        return None
        
    def get_status(self, request_id: str) -> Dict[str, Any]:
        """Get request status"""
        if request_id in self.active_requests:
            return self.active_requests[request_id]
        elif request_id in self.results_cache:
            return {
                'status': 'completed',
                'result': self.results_cache[request_id]
            }
        else:
            return {'status': 'not_found'}
            
    def _processing_loop(self):
        """Main processing loop"""
        while self.processing_active:
            try:
                # Get next request from queue (with timeout)
                try:
                    priority, submit_time, request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Update request status
                self.active_requests[request.request_id]['status'] = 'processing'
                
                # Submit to worker
                worker = self._get_available_worker()
                future = worker.process_slide.remote(request.to_dict())
                
                # Wait for result
                try:
                    result = ray.get(future, timeout=self.config.get('inference_timeout', 300))
                    
                    # Store result
                    self._store_result(request.request_id, result)
                    
                    # Update status
                    self.active_requests[request.request_id]['status'] = 'completed'
                    
                except ray.exceptions.RayTimeoutError:
                    logger.error(f"Request {request.request_id} timed out")
                    self._store_result(request.request_id, {
                        'error': 'timeout',
                        'request_id': request.request_id
                    })
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                
    def _get_available_worker(self):
        """Get available worker (simple round-robin)"""
        # In production, implement actual load balancing
        return self.workers[len(self.active_requests) % len(self.workers)]
        
    def _store_result(self, request_id: str, result: Dict[str, Any]):
        """Store result in cache and Redis"""
        self.results_cache[request_id] = result
        
        # Store in Redis with expiration
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"result:{request_id}",
                    self.config.get('result_cache_ttl', 3600),  # 1 hour TTL
                    json.dumps(result, default=str)
                )
            except Exception as e:
                logger.error(f"Failed to store in Redis: {str(e)}")


# FastAPI application
app = FastAPI(title="CRC Subtype Inference API", version="2.0.0")

# Global orchestrator instance
orchestrator = None


class InferenceRequestModel(BaseModel):
    wsi_path: str
    slide_id: str
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None
    callback_url: Optional[str] = None


class InferenceStatusResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global orchestrator
    
    # Load configuration
    config_path = os.environ.get('INFERENCE_CONFIG', 'deployment/cluster/config/inference_config.yaml')
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address='auto', ignore_reinit_error=True)
        
    # Initialize orchestrator
    orchestrator = InferenceOrchestrator(config)
    
    logger.info("Inference API started successfully")


@app.post("/inference/submit")
async def submit_inference(request: InferenceRequestModel) -> Dict[str, str]:
    """Submit WSI for inference"""
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Create inference request
    inference_request = InferenceRequest(
        request_id=request_id,
        wsi_path=request.wsi_path,
        slide_id=request.slide_id,
        priority=request.priority,
        metadata=request.metadata or {},
        callback_url=request.callback_url
    )
    
    # Submit to orchestrator
    orchestrator.submit_request(inference_request)
    
    return {"request_id": request_id, "status": "submitted"}


@app.get("/inference/status/{request_id}")
async def get_inference_status(request_id: str) -> InferenceStatusResponse:
    """Get inference status and result"""
    status_info = orchestrator.get_status(request_id)
    
    return InferenceStatusResponse(
        request_id=request_id,
        status=status_info['status'],
        result=status_info.get('result')
    )


@app.get("/inference/result/{request_id}")
async def get_inference_result(request_id: str) -> Dict[str, Any]:
    """Get inference result"""
    result = orchestrator.get_result(request_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
        
    return result


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    # Return basic metrics about the inference system
    return {
        "active_requests": len(orchestrator.active_requests),
        "cached_results": len(orchestrator.results_cache),
        "workers": len(orchestrator.workers),
        "queue_size": orchestrator.request_queue.qsize()
    }


if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker since we're using Ray for parallelism
        log_level="info"
    ) 