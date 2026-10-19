import yaml
import torch
import os
import json
from src.models.model_factory import get_model
from src.benchmarking.profiler import TinyMLProfiler

def main():
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    models_to_profile = [config['models']['baseline'], config['models']['lightweight']]
    results = {}

    for name in models_to_profile:
        print(f"Profiling {name}...")
        model = get_model(name)
        # For the baseline, we should load trained weights if available
        best_path = os.path.join(config['paths']['artifacts_dir'], f"{name}_best.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=False))

        profiler = TinyMLProfiler(model, config)

        # Use the final artifact path for size measurement if it exists
        final_path = os.path.join(config['paths']['artifacts_dir'], f"{name}_final.pth")

        results[name] = profiler.run_full_profile(artifact_path=final_path)

    # Save results
    with open("results/laptop_profiling.json", 'w') as f:
        json.dump(results, f, indent=4)

    print("Profiling complete. Results saved to results/laptop_profiling.json")
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Params: {data['parameters']['total_params']}")
        print(f"  Size: {data['size']['size_mb']:.2f} MB")
        print(f"  Latency: {data['performance']['mean_latency_s']:.4f}s")
        print(f"  Throughput: {data['performance']['throughput_fps']:.2f} FPS")

if __name__ == "__main__":
    main()
