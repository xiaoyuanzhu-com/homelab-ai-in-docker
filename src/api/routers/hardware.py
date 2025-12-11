"""Hardware stats API endpoints."""

import logging
from typing import Optional

import psutil
import pynvml
from fastapi import APIRouter

# Lazy import torch - not in main env anymore
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["hardware"])


def _cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


@router.get("/hardware/gpu/memory")
async def get_gpu_memory_details():
    """Get detailed GPU memory breakdown showing what's using memory.

    This provides a breakdown similar to nvidia-smi but with PyTorch-specific
    details about allocated vs cached memory, and which models are loaded.
    """
    # Use pynvml to check for GPU, not torch
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return {"error": "No GPU devices found"}
    except Exception as e:
        return {"error": f"NVML initialization failed: {e}"}

    try:
        devices = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Get NVML memory info (matches nvidia-smi)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            nvml_total_mb = mem_info.total / (1024**2)
            nvml_used_mb = mem_info.used / (1024**2)
            nvml_free_mb = mem_info.free / (1024**2)

            # Get PyTorch memory info (if torch available)
            torch_allocated_mb = 0.0
            torch_reserved_mb = 0.0
            torch_cached_mb = 0.0
            memory_summary = None

            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    torch_allocated_mb = torch.cuda.memory_allocated(i) / (1024**2)
                    torch_reserved_mb = torch.cuda.memory_reserved(i) / (1024**2)
                    torch_cached_mb = torch_reserved_mb - torch_allocated_mb
                except Exception:
                    pass

                # Get memory summary from PyTorch
                try:
                    memory_summary = torch.cuda.memory_summary(i)
                except Exception:
                    pass

            device_info = {
                "device_id": i,
                "name": pynvml.nvmlDeviceGetName(handle),
                "nvml_stats": {
                    "total_mb": round(nvml_total_mb, 2),
                    "used_mb": round(nvml_used_mb, 2),
                    "free_mb": round(nvml_free_mb, 2),
                    "usage_percent": round((nvml_used_mb / nvml_total_mb) * 100, 1),
                },
                "pytorch_stats": {
                    "allocated_mb": round(torch_allocated_mb, 2),
                    "reserved_mb": round(torch_reserved_mb, 2),
                    "cached_mb": round(torch_cached_mb, 2),
                    "explanation": {
                        "allocated": "Memory actively used by tensors/models",
                        "reserved": "Memory reserved by PyTorch from GPU",
                        "cached": "Reserved but unused memory (can be freed with torch.cuda.empty_cache())",
                    }
                },
                "memory_summary": memory_summary,
            }

            devices.append(device_info)

        pynvml.nvmlShutdown()

        # Get loaded models from coordinator
        loaded_models = []
        try:
            from ...services.model_coordinator import get_coordinator
            coordinator = get_coordinator()
            memory_stats = await coordinator.get_memory_stats()
            loaded_models = memory_stats.get("model_details", [])
        except Exception as e:
            logger.debug(f"Could not get loaded models: {e}")

        return {
            "devices": devices,
            "loaded_models": loaded_models,
            "tip": "If cached_mb is high with no models loaded, PyTorch is holding onto freed memory. This is normal and the cache will be reused for future allocations.",
        }

    except Exception as e:
        logger.error(f"Error getting GPU memory details: {e}")
        return {
            "error": "Failed to retrieve GPU memory details",
            "detail": str(e),
        }


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
        cuda_available = _cuda_available()

        # Also check via pynvml if torch is not available
        nvml_available = False
        try:
            pynvml.nvmlInit()
            nvml_available = pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            pass

        if cuda_available or nvml_available:
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

                # Get PyTorch memory stats for additional context (if torch available)
                torch_mem_allocated = None
                torch_mem_reserved = None
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        torch_mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        torch_mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    except Exception:
                        pass

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

        # Get model coordinator stats
        model_coordinator_stats = None
        try:
            from ...services.model_coordinator import get_coordinator
            from ...db.settings import get_setting_int, get_setting_bool, get_setting_float

            coordinator = get_coordinator()
            memory_stats = await coordinator.get_memory_stats()

            # Get coordinator configuration from database
            max_models = get_setting_int("max_models_in_memory", 1)
            enable_preemptive = get_setting_bool("enable_preemptive_unload", True)
            max_memory = get_setting_float("max_memory_mb", None)
            idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

            model_coordinator_stats = {
                "config": {
                    "max_models_in_memory": max_models,
                    "enable_preemptive_unload": enable_preemptive,
                    "max_memory_mb": max_memory,
                    "idle_timeout_seconds": idle_timeout,
                },
                "models_loaded": memory_stats.get("models_loaded", 0),
                "loaded_models": memory_stats.get("model_details", []),
                "gpu_memory": {
                    "available": memory_stats.get("gpu_available", False),
                    "total_mb": round(memory_stats.get("gpu_total_mb", 0), 2) if memory_stats.get("gpu_total_mb") else None,
                    "used_mb": round(memory_stats.get("gpu_used_mb", 0), 2) if memory_stats.get("gpu_used_mb") else None,
                    "cached_mb": round(memory_stats.get("gpu_cached_mb", 0), 2) if memory_stats.get("gpu_cached_mb") else None,
                    "free_mb": round(memory_stats.get("gpu_free_mb", 0), 2) if memory_stats.get("gpu_free_mb") else None,
                },
            }
        except Exception as e:
            logger.debug(f"Could not get model coordinator stats: {e}")

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
            "model_coordinator": model_coordinator_stats,
        }
    except Exception as e:
        logger.error(f"Error getting hardware stats: {e}")
        return {
            "error": "Failed to retrieve hardware stats",
            "detail": str(e),
        }
