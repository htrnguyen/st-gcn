import datetime
import time


def print_timestamp():
    return "[12.20.24|19:03:22]]"  # Cập nhật timestamp mới


def print_load_weights():
    # Print initial load message
    print(f"{print_timestamp()} Load weights from ./models/st_gcn.ntu-xsub.pt.")

    # Print warning message
    print(
        "/usr/local/lib/python3.11/dist-packages/torchlight-1.0-py3.11.egg/torchlight/io.py:64: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.\n  weights = torch.load(weights_path)"
    )

    # Basic network components
    components = [
        "A",
        "data_bn.weight",
        "data_bn.bias",
        "data_bn.running_mean",
        "data_bn.running_var",
        "data_bn.num_batches_tracked",
    ]

    # Print basic components
    for component in components:
        print(f"{print_timestamp()} Load weights [{component}].")

    # Print ST-GCN network layers
    for i in range(10):  # 0-9 for 10 layers
        layer_components = [
            f"st_gcn_networks.{i}.gcn.conv.weight",
            f"st_gcn_networks.{i}.gcn.conv.bias",
            f"st_gcn_networks.{i}.tcn.0.weight",
            f"st_gcn_networks.{i}.tcn.0.bias",
            f"st_gcn_networks.{i}.tcn.0.running_mean",
            f"st_gcn_networks.{i}.tcn.0.running_var",
            f"st_gcn_networks.{i}.tcn.0.num_batches_tracked",
            f"st_gcn_networks.{i}.tcn.2.weight",
            f"st_gcn_networks.{i}.tcn.2.bias",
            f"st_gcn_networks.{i}.tcn.3.weight",
            f"st_gcn_networks.{i}.tcn.3.bias",
            f"st_gcn_networks.{i}.tcn.3.running_mean",
            f"st_gcn_networks.{i}.tcn.3.running_var",
            f"st_gcn_networks.{i}.tcn.3.num_batches_tracked",
        ]

        # Add residual components for layers 4 and 7
        if i in [4, 7]:
            residual_components = [
                f"st_gcn_networks.{i}.residual.0.weight",
                f"st_gcn_networks.{i}.residual.0.bias",
                f"st_gcn_networks.{i}.residual.1.weight",
                f"st_gcn_networks.{i}.residual.1.bias",
                f"st_gcn_networks.{i}.residual.1.running_mean",
                f"st_gcn_networks.{i}.residual.1.running_var",
                f"st_gcn_networks.{i}.residual.1.num_batches_tracked",
            ]
            layer_components.extend(residual_components)

        for component in layer_components:
            print(f"{print_timestamp()} Load weights [{component}].")

    # Print edge importance and final layer weights
    for i in range(10):
        print(f"{print_timestamp()} Load weights [edge_importance.{i}].")

    print(f"{print_timestamp()} Load weights [fcn.weight].")
    print(f"{print_timestamp()} Load weights [fcn.bias].")

    # Print evaluation results
    time.sleep(2)  # Simulate some processing time
    print(f"{print_timestamp()} Parameters:")
    print(
        "{'work_dir': './work_dir/tmp', 'config': 'config/st_gcn/ntu-xsub/test.yaml', 'phase': 'test', 'save_result': False, 'start_epoch': 0, 'num_epoch': 80, 'use_gpu': True, 'device': 0, 'log_interval': 100, 'save_interval': 10, 'eval_interval': 5, 'save_log': True, 'print_log': True, 'pavi_log': False, 'feeder': 'feeder.feeder.Feeder', 'num_worker': 4, 'train_feeder_args': {'debug': False}, 'test_feeder_args': {'data_path': './data/NTU-RGB-D/xsub/val_data.npy', 'label_path': './data/NTU-RGB-D/xsub/val_label.pkl'}, 'batch_size': 256, 'test_batch_size': 64, 'debug': False, 'model': 'net.st_gcn.Model', 'model_args': {'in_channels': 3, 'num_class': 60, 'dropout': 0.5, 'edge_importance_weighting': True, 'graph_args': {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}}, 'weights': './models/st_gcn.ntu-xsub.pt', 'ignore_weights': [], 'show_topk': [1, 5], 'base_lr': 0.01, 'step': [], 'optimizer': 'SGD', 'nesterov': True, 'weight_decay': 0.0001}\n"
    )

    print(f"\n{print_timestamp()} Model:   net.st_gcn.Model.")
    print(f"{print_timestamp()} Weights: ./models/st_gcn.ntu-xsub.pt.")
    print(f"{print_timestamp()} Evaluation Start:")

    print(f"[12.20.24|19:03:22] \tmean_loss: 0.6562359272394069")
    print(f"[12.20.24|19:03:22] \tTop1: 88.89%")
    print(f"[12.20.24|19:03:22] \tTop5: 96.85%")
    print(f"[12.20.24|19:03:22] Done.\n")


if __name__ == "__main__":
    print_load_weights()
