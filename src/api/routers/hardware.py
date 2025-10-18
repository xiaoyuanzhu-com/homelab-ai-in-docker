"""Hardware stats API endpoints."""

import logging
from typing import Optional

import psutil
import torch
from fastapi import APIRouter

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["hardware"])


@router.get("/hardware")
async def get_hardware_stats():
    """Get current hardware statistics including CPU, memory, and GPU."""
    try:
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Memory stats
        memory = psutil.virtual_memory()

        # GPU stats using PyTorch
        gpu_stats = []
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)

                # Get memory info
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                mem_total = props.total_memory / (1024**3)  # GB

                gpu_info = {
                    "id": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(mem_total, 2),
                    "allocated_memory_gb": round(mem_allocated, 2),
                    "reserved_memory_gb": round(mem_reserved, 2),
                    "free_memory_gb": round(mem_total - mem_reserved, 2),
                    "memory_usage_percent": round((mem_reserved / mem_total) * 100, 1),
                }

                # Try to get utilization from GPUtil if available
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if i < len(gpus):
                            gpu_info["utilization_percent"] = round(gpus[i].load * 100, 1)
                            gpu_info["temperature_c"] = gpus[i].temperature
                    except Exception as e:
                        logger.debug(f"Could not get GPU utilization from GPUtil: {e}")

                gpu_stats.append(gpu_info)

        # Determine which device will be used for inference
        inference_device = "cuda" if cuda_available else "cpu"

        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "cores": cpu_count,
                "frequency_mhz": round(cpu_freq.current, 0) if cpu_freq else None,
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent,
            },
            "gpu": {
                "available": cuda_available,
                "count": len(gpu_stats),
                "devices": gpu_stats,
            },
            "inference": {
                "device": inference_device,
                "description": f"AI models will use {inference_device.upper()} for inference",
            },
        }
    except Exception as e:
        logger.error(f"Error getting hardware stats: {e}")
        return {
            "error": "Failed to retrieve hardware stats",
            "detail": str(e),
        }
