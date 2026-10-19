import torch
from torchvision import datasets
import os

def download_cifar10():
    print("Starting CIFAR-10 download...")
    try:
        datasets.CIFAR10(root='./data', train=True, download=True)
        print("Train set downloaded.")
        datasets.CIFAR10(root='./data', train=False, download=True)
        print("Test set downloaded.")
        print("CIFAR-10 download successful!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_cifar10()
