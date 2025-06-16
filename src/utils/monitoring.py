"""
Comprehensive Monitoring System for Distributed Training
Tracks GPU usage, system metrics, network performance, and training progress
"""

import os
import time
import json
import logging
import threading
import psutil
import socket
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import queue

import torch
import numpy as np
import nvidia_ml_py3 as nvml

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-level metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    load_avg: List[float]
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPUMetrics:
    """GPU-specific metrics"""
    timestamp: float
    gpu_id: int
    gpu_name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    utilization_percent: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    clock_sm_mhz: int
    clock_memory_mhz: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    timestamp: float
    rank: int
    bandwidth_mbps: float
    latency_ms: float
    packet_loss_percent: float
    connection_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Training-specific metrics"""
    timestamp: float
    epoch: int
    step: int
    loss: float
    learning_rate: float
    batch_size: int
    throughput_samples_per_sec: float
    gpu_memory_allocated_mb: float
    gradient_norm: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GPUMonitor:
    """Monitor GPU usage and performance"""
    
    def __init__(self):
        self.nvml_initialized = False
        self._initialize_nvml()
        
    def _initialize_nvml(self):
        """Initialize NVIDIA Management Library"""
        try:
            nvml.nvmlInit()
            self.nvml_initialized = True
            logger.info("NVML initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {str(e)}")
            self.nvml_initialized = False
            
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get metrics for all GPUs"""
        if not self.nvml_initialized:
            return []
            
        metrics = []
        timestamp = time.time()
        
        try:
            device_count = nvml.nvmlDeviceGetCount()
            
            for gpu_id in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Get GPU name
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = mem_info.used / 1024 / 1024
                memory_total_mb = mem_info.total / 1024 / 1024
                memory_percent = (mem_info.used / mem_info.total) * 100
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent = util.gpu
                
                # Temperature
                temperature_c = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Power
                power_draw_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit_w = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                
                # Clock speeds
                clock_sm_mhz = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                clock_memory_mhz = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                
                metrics.append(GPUMetrics(
                    timestamp=timestamp,
                    gpu_id=gpu_id,
                    gpu_name=name,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                    memory_percent=memory_percent,
                    utilization_percent=utilization_percent,
                    temperature_c=temperature_c,
                    power_draw_w=power_draw_w,
                    power_limit_w=power_limit_w,
                    clock_sm_mhz=clock_sm_mhz,
                    clock_memory_mhz=clock_memory_mhz
                ))
                
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {str(e)}")
            
        return metrics


class SystemMonitor:
    """Monitor system-level metrics"""
    
    def __init__(self):
        self.boot_time = psutil.boot_time()
        self.last_network_io = psutil.net_io_counters()
        self.last_network_time = time.time()
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        timestamp = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / 1024 / 1024 / 1024
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Network I/O
        current_network_io = psutil.net_io_counters()
        current_time = time.time()
        
        time_delta = current_time - self.last_network_time
        if time_delta > 0:
            bytes_sent_delta = current_network_io.bytes_sent - self.last_network_io.bytes_sent
            bytes_recv_delta = current_network_io.bytes_recv - self.last_network_io.bytes_recv
            
            network_sent_mb = (bytes_sent_delta / time_delta) / 1024 / 1024
            network_recv_mb = (bytes_recv_delta / time_delta) / 1024 / 1024
        else:
            network_sent_mb = 0.0
            network_recv_mb = 0.0
            
        self.last_network_io = current_network_io
        self.last_network_time = current_time
        
        # Load average
        load_avg = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            load_avg=load_avg,
            process_count=process_count
        )


class NetworkMonitor:
    """Monitor network performance for distributed training"""
    
    def __init__(self, rank: int = 0):
        self.rank = rank
        self.last_measurement_time = time.time()
        
    def measure_bandwidth(self, target_ranks: List[int] = None) -> float:
        """Measure network bandwidth between ranks"""
        # This is a simplified bandwidth measurement
        # In practice, you'd implement all-to-all communication tests
        try:
            if torch.distributed.is_initialized():
                # Simple tensor communication test
                tensor_size = 1024 * 1024  # 1MB
                test_tensor = torch.randn(tensor_size // 4).cuda()
                
                start_time = time.time()
                
                # Broadcast from rank 0
                torch.distributed.broadcast(test_tensor, src=0)
                torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Calculate bandwidth in MB/s
                data_size_mb = (tensor_size * 4) / 1024 / 1024  # Float32 = 4 bytes
                time_elapsed = end_time - start_time
                bandwidth_mbps = data_size_mb / time_elapsed
                
                return bandwidth_mbps
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to measure bandwidth: {str(e)}")
            return 0.0
            
    def measure_latency(self) -> float:
        """Measure communication latency"""
        try:
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                # Small tensor for latency test
                test_tensor = torch.tensor([1.0]).cuda()
                
                start_time = time.time()
                
                # All-reduce operation
                torch.distributed.all_reduce(test_tensor)
                torch.cuda.synchronize()
                
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                return latency_ms
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to measure latency: {str(e)}")
            return 0.0
            
    def get_network_metrics(self) -> NetworkMetrics:
        """Get current network metrics"""
        timestamp = time.time()
        
        bandwidth_mbps = self.measure_bandwidth()
        latency_ms = self.measure_latency()
        
        # Get connection count
        connection_count = len(psutil.net_connections())
        
        return NetworkMetrics(
            timestamp=timestamp,
            rank=self.rank,
            bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
            packet_loss_percent=0.0,  # Placeholder
            connection_count=connection_count
        )


class ClusterMonitor:
    """Comprehensive monitoring for distributed training clusters"""
    
    def __init__(
        self,
        log_dir: str,
        rank: int = 0,
        world_size: int = 1,
        monitor_interval: int = 60,
        save_interval: int = 300
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.rank = rank
        self.world_size = world_size
        self.monitor_interval = monitor_interval
        self.save_interval = save_interval
        
        # Initialize monitors
        self.gpu_monitor = GPUMonitor()
        self.system_monitor = SystemMonitor()
        self.network_monitor = NetworkMonitor(rank)
        
        # Metrics storage
        self.metrics_buffer = {
            'system': [],
            'gpu': [],
            'network': [],
            'training': []
        }
        
        # Monitoring control
        self.monitoring_active = False
        self.monitor_thread = None
        self.save_thread = None
        
        # Alerting
        self.alert_callbacks = []
        self.alert_thresholds = {
            'gpu_memory_percent': 95.0,
            'gpu_temperature_c': 85.0,
            'system_memory_percent': 90.0,
            'system_cpu_percent': 95.0
        }
        
        logger.info(f"Initialized cluster monitor (rank {rank}/{world_size})")
        
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start saving thread
        self.save_thread = threading.Thread(target=self._saving_loop, daemon=True)
        self.save_thread.start()
        
        logger.info("Started continuous monitoring")
        
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        if self.save_thread:
            self.save_thread.join(timeout=5)
            
        # Save remaining metrics
        self._save_metrics()
        
        logger.info("Stopped monitoring")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.get_system_metrics()
                self.metrics_buffer['system'].append(system_metrics)
                
                # Collect GPU metrics
                gpu_metrics = self.gpu_monitor.get_gpu_metrics()
                self.metrics_buffer['gpu'].extend(gpu_metrics)
                
                # Collect network metrics (less frequently)
                if len(self.metrics_buffer['system']) % 5 == 0:
                    network_metrics = self.network_monitor.get_network_metrics()
                    self.metrics_buffer['network'].append(network_metrics)
                
                # Check alerts
                self._check_alerts(system_metrics, gpu_metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                
            time.sleep(self.monitor_interval)
            
    def _saving_loop(self):
        """Save metrics to disk periodically"""
        while self.monitoring_active:
            time.sleep(self.save_interval)
            self._save_metrics()
            
    def _save_metrics(self):
        """Save metrics to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for metric_type, metrics_list in self.metrics_buffer.items():
            if not metrics_list:
                continue
                
            # Convert to dictionaries
            metrics_data = [m.to_dict() for m in metrics_list]
            
            # Save to JSON file
            filename = f"{metric_type}_metrics_rank_{self.rank}_{timestamp}.json"
            filepath = self.log_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            # Clear buffer
            self.metrics_buffer[metric_type] = []
            
        logger.debug(f"Saved metrics to {self.log_dir}")
        
    def _check_alerts(self, system_metrics: SystemMetrics, gpu_metrics: List[GPUMetrics]):
        """Check for alert conditions"""
        alerts = []
        
        # System alerts
        if system_metrics.memory_percent > self.alert_thresholds['system_memory_percent']:
            alerts.append(f"High system memory usage: {system_metrics.memory_percent:.1f}%")
            
        if system_metrics.cpu_percent > self.alert_thresholds['system_cpu_percent']:
            alerts.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
            
        # GPU alerts
        for gpu_metric in gpu_metrics:
            if gpu_metric.memory_percent > self.alert_thresholds['gpu_memory_percent']:
                alerts.append(f"High GPU {gpu_metric.gpu_id} memory: {gpu_metric.memory_percent:.1f}%")
                
            if gpu_metric.temperature_c > self.alert_thresholds['gpu_temperature_c']:
                alerts.append(f"High GPU {gpu_metric.gpu_id} temperature: {gpu_metric.temperature_c}°C")
                
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
            
    def _send_alert(self, message: str):
        """Send alert notification"""
        logger.warning(f"ALERT: {message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Failed to send alert via callback: {str(e)}")
                
    def log_training_metrics(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        batch_size: int,
        throughput: float = 0.0
    ):
        """Log training-specific metrics"""
        # Get current GPU memory usage
        gpu_memory_allocated_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            throughput_samples_per_sec=throughput,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
            gradient_norm=0.0  # Set externally if needed
        )
        
        self.metrics_buffer['training'].append(training_metrics)
        
    def log_gpu_memory(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024
                logger.debug(f"GPU {device_id} - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
                
    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add custom alert callback"""
        self.alert_callbacks.append(callback)
        
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set custom alert threshold"""
        self.alert_thresholds[metric] = threshold
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current snapshot of all metrics"""
        system_metrics = self.system_monitor.get_system_metrics()
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        network_metrics = self.network_monitor.get_network_metrics()
        
        return {
            'timestamp': time.time(),
            'rank': self.rank,
            'system': system_metrics.to_dict(),
            'gpus': [gpu.to_dict() for gpu in gpu_metrics],
            'network': network_metrics.to_dict()
        }
        
    def generate_report(self, hours: int = 1) -> Dict[str, Any]:
        """Generate monitoring report for the last N hours"""
        # This would analyze saved metrics files and generate a comprehensive report
        # Implementation would involve reading JSON files and computing statistics
        
        report = {
            'report_period_hours': hours,
            'rank': self.rank,
            'world_size': self.world_size,
            'summary': {
                'avg_gpu_utilization': 0.0,
                'avg_gpu_memory_usage': 0.0,
                'avg_system_cpu': 0.0,
                'avg_system_memory': 0.0,
                'peak_gpu_memory': 0.0,
                'peak_system_memory': 0.0,
                'network_bandwidth_avg': 0.0,
                'network_latency_avg': 0.0
            },
            'alerts_count': 0,
            'uptime_percent': 100.0
        }
        
        return report


class PerformanceProfiler:
    """Profile training performance and identify bottlenecks"""
    
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
        self.active_timers = {}
        
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.active_timers[name] = time.time()
        
    def end_timer(self, name: str):
        """End timing an operation"""
        if name not in self.active_timers:
            logger.warning(f"Timer {name} was not started")
            return
            
        elapsed = time.time() - self.active_timers[name]
        
        if name not in self.timing_data:
            self.timing_data[name] = []
            
        self.timing_data[name].append(elapsed)
        del self.active_timers[name]
        
    def record_memory_usage(self, name: str):
        """Record current memory usage"""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            if name not in self.memory_data:
                self.memory_data[name] = []
                
            self.memory_data[name].append(memory_mb)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary"""
        summary = {}
        
        # Timing analysis
        for operation, times in self.timing_data.items():
            summary[f"{operation}_avg_time"] = np.mean(times)
            summary[f"{operation}_max_time"] = np.max(times)
            summary[f"{operation}_min_time"] = np.min(times)
            summary[f"{operation}_std_time"] = np.std(times)
            
        # Memory analysis
        for operation, memories in self.memory_data.items():
            summary[f"{operation}_avg_memory"] = np.mean(memories)
            summary[f"{operation}_max_memory"] = np.max(memories)
            summary[f"{operation}_min_memory"] = np.min(memories)
            
        return summary 