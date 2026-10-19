import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

def load_metrics(results_dir, model_names):
    data = []
    for name in model_names:
        metric_file = os.path.join(results_dir, f"{name}_metrics.json")
        info_file = os.path.join(results_dir, f"{name}_info.json")

        if os.path.exists(metric_file) and os.path.exists(info_file):
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
            with open(info_file, 'r') as f:
                info = json.load(f)

            data.append({
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Macro-F1": metrics["macro_f1"],
                "Size (MB)": info.get("model_size_mb", info.get("quantized_size_mb", 0)),
                "Params": info.get("params", 0)
            })
    return pd.DataFrame(data)

def generate_report(df, results_dir):
    if df.empty:
        print("No data to analyze.")
        return

    # Calculate relative changes vs baseline (first row)
    baseline = df.iloc[0]
    df['Acc Change (%)'] = 100 * (df['Accuracy'] - baseline['Accuracy']) / (baseline['Accuracy'] + 1e-6)
    df['Size Reduction (%)'] = 100 * (baseline['Size (MB)'] - df['Size (MB)']) / (baseline['Size (MB)'] + 1e-6)

    # Save as CSV
    df.to_csv(os.path.join(results_dir, "comparative_analysis.csv"), index=False)

    # Plots
    plt.figure(figsize=(12, 5))

    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(x="Model", y="Accuracy", data=df)
    plt.title("Accuracy across Variants")
    plt.ylim(0, 1)

    # Plot 2: Size
    plt.subplot(1, 2, 2)
    sns.barplot(x="Model", y="Size (MB)", data=df)
    plt.title("Model Size (MB)")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "performance_comparison.png"))
    print(f"Analysis report and plots saved to {results_dir}")

if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    results_dir = config['paths']['results_dir']
    model_names = [f"{config['models']['baseline']}_baseline", config['models']['lightweight'], f"{config['models']['baseline']}_structured_pruned"]

    df = load_metrics(results_dir, model_names)
    print("Analysis DataFrame:")
    print(df)
    generate_report(df, results_dir)
