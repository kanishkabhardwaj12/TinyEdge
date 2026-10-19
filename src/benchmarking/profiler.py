import torch
import torch.nn as nn
import time
import numpy as np
import os
import psutil

class TinyMLProfiler:
    """
    Comprehensive profiler for TinyML models on laptop hardware.
    Measures latency, throughput, memory, and parameters.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config['evaluation']['device'])
        self.model.to(self.device).eval()

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params
        }

    def get_model_size(self, artifact_path=None):
        if artifact_path and os.path.exists(artifact_path):
            size_bytes = os.path.getsize(artifact_path)
        else:
            # Approximation if artifact not yet saved
            size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        return {
            "size_mb": size_bytes / (1024 * 1024),
            "size_bytes": size_bytes
        }

    def measure_latency_throughput(self, input_shape=(1, 3, 32, 32),
                                   iterations=None, warmup=None):
        iterations = iterations or self.config["evaluation"].get("latency_iterations", 100)
        warmup = warmup or self.config["evaluation"].get("latency_warmup", 10)
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        self._synchronize()

        # Measurement
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                self._synchronize()
                latencies.append(time.perf_counter() - start)

        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        std_latency = np.std(latencies)
        throughput = 1.0 / mean_latency
        p95_latency = np.percentile(latencies, 95)

        return {
            "mean_latency_s": mean_latency,
            "median_latency_s": median_latency,
            "std_latency_s": std_latency,
            "p95_latency_s": p95_latency,
            "throughput_fps": throughput,
            "throughput_batch_size": input_shape[0],
            "throughput_calculation": "batch_size / mean_batch_latency_s",
            "input_shape": input_shape,
            "iterations": iterations
        }

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def measure_memory(self, input_shape=(1, 3, 32, 32)):
        # Measure peak process memory during a single forward pass
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        dummy_input = torch.randn(input_shape).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)

        mem_after = process.memory_info().rss

        return {
            "peak_memory_mb": mem_after / (1024 * 1024),
            "memory_delta_mb": (mem_after - mem_before) / (1024 * 1024),
            "method": "process_rss_before_after_forward",
            "note": "RSS is a process-level estimate, not allocator-level peak activation memory."
        }

    def profile_flops(self, input_shape=(1, 3, 32, 32)):
        """Count MACs for Conv2d and Linear layers using forward hooks."""
        macs = 0
        handles = []

        def count(module, inputs, output):
            nonlocal macs
            if isinstance(module, nn.Conv2d):
                output_elements = output.shape[0] * output.shape[2] * output.shape[3]
                macs += output_elements * module.in_channels * module.kernel_size[0] * module.kernel_size[1] // module.groups
            elif isinstance(module, nn.Linear):
                macs += output.numel() * module.in_features

        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(count))
        with torch.no_grad():
            self.model(torch.randn(input_shape).to(self.device))
        for handle in handles:
            handle.remove()
        return {
            "macs": int(macs),
            "flops": int(macs * 2),
            "status": "estimated_from_conv_linear_hooks",
            "note": "MACs count multiply-accumulate operations; FLOPs are reported as 2 per MAC."
        }

    def run_full_profile(self, artifact_path=None):
        params = self.get_parameter_count()
        size = self.get_model_size(artifact_path)
        perf = self.measure_latency_throughput()
        mem = self.measure_memory()
        flops = self.profile_flops()

        return {
            "parameters": params,
            "size": size,
            "performance": perf,
            "memory": mem,
            "complexity": flops,
            "device": str(self.device),
            "timestamp": time.time()
        }
