import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(config):
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    seed = config['training']['seed']
    subset_fraction = config['dataset'].get('subset_fraction', 1.0)

    # Check if we should use dummy data
    use_dummy = os.environ.get("USE_DUMMY_DATA", "false").lower() == "true"

    if use_dummy:
        print("Using dummy data for verification...")
        try:
            train_data = torch.load("data/dummy/train.pt")
            val_data = torch.load("data/dummy/val.pt")
            test_data = torch.load("data/dummy/test.pt")
        except FileNotFoundError:
            raise FileNotFoundError("Dummy data not found. Run 'python3 scripts/create_dummy_data.py' first.")

        train_loader = DataLoader(
            torch.utils.data.TensorDataset(train_data['x'], train_data['y']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(val_data['x'], val_data['y']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(test_data['x'], test_data['y']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, val_loader, test_loader

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        train_dataset_raw = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        val_dataset_raw = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    except Exception as e:
        print(f"Critical error loading CIFAR-10: {e}")
        raise RuntimeError("CIFAR-10 dataset could not be loaded. Please ensure data is in ./data.")

    if not 0 < subset_fraction <= 1.0:
        raise ValueError("dataset.subset_fraction must be in the interval (0, 1].")

    if subset_fraction < 1.0:
        print(f"Using subset of data: {subset_fraction*100}%")
        train_size = int(subset_fraction * len(train_dataset_raw))
        test_size = int(subset_fraction * len(test_dataset))
        indices_train = torch.randperm(
            len(train_dataset_raw),
            generator=torch.Generator().manual_seed(seed)
        )[:train_size]
        indices_test = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(seed))[:test_size]
        # Keep train and validation views aligned to the same sampled examples.
        train_dataset_raw = Subset(train_dataset_raw, indices_train.tolist())
        val_dataset_raw = Subset(val_dataset_raw, indices_train.tolist())
        test_dataset = Subset(test_dataset, indices_test)

    val_size = int(0.1 * len(train_dataset_raw))
    indices = torch.randperm(len(train_dataset_raw), generator=torch.Generator().manual_seed(seed))
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    train_loader = DataLoader(Subset(train_dataset_raw, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(val_dataset_raw, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
