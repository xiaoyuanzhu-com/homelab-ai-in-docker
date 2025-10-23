"""Hardware stats API endpoints."""

import logging
from typing import Optional

import psutil
import pynvml
import torch
from fastapi import APIRouter

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

        # Get CPU model name
        cpu_model = None
        try:
            # Try to read from /proc/cpuinfo (Linux)
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.strip().startswith('model name'):
                        cpu_model = line.split(':', 1)[1].strip()
                        break
        except Exception as e:
            logger.debug(f"Could not get CPU model from /proc/cpuinfo: {e}")

        # Get CPU temperature
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Try common sensor names in order of preference
                for sensor_name in ['coretemp', 'k10temp', 'zenpower', 'cpu_thermal']:
                    if sensor_name in temps:
                        entries = temps[sensor_name]
                        if entries:
                            # Get the first entry or look for 'Package id 0'/'Tdie'
                            for entry in entries:
                                if entry.label in ['Package id 0', 'Tdie', 'Tctl'] or not entry.label:
                                    cpu_temp = entry.current
                                    break
                            if cpu_temp is None and entries:
                                # Fallback to first entry
                                cpu_temp = entries[0].current
                        break
        except Exception as e:
            logger.debug(f"Could not get CPU temperature: {e}")

        # Memory stats
        memory = psutil.virtual_memory()

        # GPU stats using NVML (nvidia-smi equivalent)
        gpu_stats = []
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            pynvml.nvmlInit()

            # Get driver and CUDA versions (system-wide, not per-GPU)
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion()
            except Exception as e:
                logger.debug(f"Could not get driver version: {e}")
                driver_version = None

            try:
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                # Convert to readable format (e.g., 12060 -> "12.6")
                if cuda_version:
                    cuda_major = cuda_version // 1000
                    cuda_minor = (cuda_version % 1000) // 10
                    cuda_version_str = f"{cuda_major}.{cuda_minor}"
                else:
                    cuda_version_str = None
            except Exception as e:
                logger.debug(f"Could not get CUDA version: {e}")
                cuda_version_str = None

            gpu_count = pynvml.nvmlDeviceGetCount()

            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get device info
                name = pynvml.nvmlDeviceGetName(handle)

                # Get memory info (matches nvidia-smi exactly)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_total_gb = mem_info.total / (1024**3)
                mem_used_gb = mem_info.used / (1024**3)
                mem_free_gb = mem_info.free / (1024**3)

                # Get utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    mem_util = utilization.memory
                except Exception as e:
                    logger.debug(f"Could not get utilization for GPU {i}: {e}")
                    gpu_util = None
                    mem_util = None

                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception as e:
                    logger.debug(f"Could not get temperature for GPU {i}: {e}")
                    temperature = None

                # Get PyTorch memory stats for additional context
                try:
                    torch_mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    torch_mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                except Exception:
                    torch_mem_allocated = None
                    torch_mem_reserved = None

                gpu_info = {
                    "id": i,
                    "name": name,
                    "driver_version": driver_version,
                    "cuda_version": cuda_version_str,
                    # NVML stats (matches nvidia-smi)
                    "total_memory_gb": round(mem_total_gb, 2),
                    "used_memory_gb": round(mem_used_gb, 2),
                    "free_memory_gb": round(mem_free_gb, 2),
                    "memory_usage_percent": round((mem_used_gb / mem_total_gb) * 100, 1),
                    "utilization_percent": gpu_util,
                    "memory_utilization_percent": mem_util,
                    "temperature_c": temperature,
                    # PyTorch stats for context
                    "pytorch_allocated_gb": round(torch_mem_allocated, 2) if torch_mem_allocated is not None else None,
                    "pytorch_reserved_gb": round(torch_mem_reserved, 2) if torch_mem_reserved is not None else None,
                }

                gpu_stats.append(gpu_info)

            pynvml.nvmlShutdown()

        # Determine which device will be used for inference
        inference_device = "cuda" if cuda_available else "cpu"

        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "cores": cpu_count,
                "frequency_mhz": round(cpu_freq.current, 0) if cpu_freq else None,
                "model": cpu_model,
                "temperature_c": round(cpu_temp, 1) if cpu_temp is not None else None,
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
