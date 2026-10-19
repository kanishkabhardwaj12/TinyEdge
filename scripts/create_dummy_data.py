import torch
import os

def create_dummy_data():
    os.makedirs("data/dummy", exist_ok=True)

    # CIFAR-10 shapes: 3x32x32 images, 10 classes
    # Create small subsets for verification
    train_x = torch.randn(1000, 3, 32, 32)
    train_y = torch.randint(0, 10, (1000,))

    val_x = torch.randn(200, 3, 32, 32)
    val_y = torch.randint(0, 10, (200,))

    test_x = torch.randn(200, 3, 32, 32)
    test_y = torch.randint(0, 10, (200,))

    torch.save({'x': train_x, 'y': train_y}, "data/dummy/train.pt")
    torch.save({'x': val_x, 'y': val_y}, "data/dummy/val.pt")
    torch.save({'x': test_x, 'y': test_y}, "data/dummy/test.pt")
    print("Dummy data created successfully in data/dummy/")

if __name__ == "__main__":
    create_dummy_data()
