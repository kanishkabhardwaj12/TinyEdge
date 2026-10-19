import torch
import time
import numpy as np
import json
import os
import argparse
from src.models.model_factory import get_model
from src.evaluation.evaluator import evaluate_model

def benchmark(model, input_shape=(1, 3, 32, 32), iterations=100, warmup=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.time() - start)

    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    throughput = 1.0 / mean_latency

    return {
        "mean_latency": mean_latency,
        "median_latency": median_latency,
        "std_latency": std_latency,
        "throughput": throughput
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    print(f"Benchmarking model: {args.model} from {args.path}")
    # Try loading as a whole object first (for quantized models)
    try:
        model = torch.load(args.path, weights_only=False)
    except Exception:
        # Fallback to state_dict
        model = get_model(args.model)
        model.load_state_dict(torch.load(args.path, weights_only=False))

    results = benchmark(model)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output}")
    print(f"Mean Latency: {results['mean_latency']:.4f}s")
    print(f"Throughput: {results['throughput']:.2f} FPS")

if __name__ == "__main__":
    main()
