import torch
import unittest
import os
import yaml
from src.data.dataloader import get_dataloaders
from src.models.model_factory import get_model
from src.models.custom_cnn import CustomCNN
from src.benchmarking.profiler import TinyMLProfiler
from src.compression.pruner import TinyMLPruner
from src.utils.runtime import available_devices

class TestTinyMLPipeline(unittest.TestCase):
    def setUp(self):
        with open("configs/config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        os.environ["USE_DUMMY_DATA"] = "true"

    def test_dataloader(self):
        train_loader, val_loader, test_loader = get_dataloaders(self.config)
        self.assertIsNotNone(train_loader)
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape[1:], (3, 32, 32))

    def test_model_output(self):
        model = get_model(self.config['models']['baseline'])
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 10))

    def test_custom_cnn_variants_and_parameter_scaling(self):
        names = ["custom_cnn", "custom_cnn_w075", "custom_cnn_w050", "custom_cnn_w025"]
        counts = []
        for name in names:
            model = get_model(name)
            self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertIsInstance(get_model("custom_cnn"), CustomCNN)

    def test_profiler_reports_mac_estimate(self):
        model = get_model("custom_cnn_w025")
        profile = TinyMLProfiler(model, self.config).profile_flops()
        self.assertGreater(profile["macs"], 0)
        self.assertEqual(profile["flops"], 2 * profile["macs"])

    def test_checkpoint_save_and_load(self):
        model = get_model("custom_cnn_w025")
        path = "data/dummy/test_checkpoint.pth"
        torch.save(model.state_dict(), path)
        restored = get_model("custom_cnn_w025")
        restored.load_state_dict(torch.load(path, weights_only=False))
        self.assertTrue(torch.equal(model(torch.ones(1, 3, 32, 32)),
                                    restored(torch.ones(1, 3, 32, 32))))
        os.remove(path)

    def test_device_list_contains_cpu(self):
        self.assertIn("cpu", available_devices())

    def test_experiment_configuration(self):
        self.assertEqual(
            self.config["models"]["comparison"],
            ["resnet18", "custom_cnn", "custom_cnn_w075",
             "custom_cnn_w050", "custom_cnn_w025"],
        )
        self.assertEqual(self.config["experiment"]["output_root"],
                         "results/experiments")

    def test_progressive_pruning_reduces_physical_channels_and_parameters(self):
        parent = get_model("custom_cnn")
        previous_params = sum(p.numel() for p in parent.parameters())
        expected_channels = [(24, 48, 96), (16, 32, 64), (8, 16, 32)]
        for retention, channels in zip((0.75, 2 / 3, 0.5), expected_channels):
            child, metadata = TinyMLPruner(parent, self.config).prune_custom_cnn(
                retention, parent
            )
            self.assertEqual(child.channels, channels)
            self.assertEqual(child(torch.randn(2, 3, 32, 32)).shape, (2, 10))
            current_params = sum(p.numel() for p in child.parameters())
            self.assertLess(current_params, previous_params)
            self.assertEqual(metadata["selected_channels"], list(channels))
            parent, previous_params = child, current_params

    def test_cifar_split_sizes_are_disjoint(self):
        cifar_root = "data/cifar-10-batches-py"
        if not os.path.exists(os.path.join(cifar_root, "data_batch_1")):
            self.skipTest("Full CIFAR-10 download is not present")
        os.environ.pop("USE_DUMMY_DATA", None)
        train, validation, test = get_dataloaders(self.config)
        self.assertEqual((len(train.dataset), len(validation.dataset), len(test.dataset)),
                         (45000, 5000, 10000))
        train_indices = set(train.dataset.indices)
        validation_indices = set(validation.dataset.indices)
        self.assertTrue(train_indices.isdisjoint(validation_indices))

if __name__ == "__main__":
    unittest.main()
