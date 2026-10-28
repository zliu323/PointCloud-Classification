import argparse
import os
import re


EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+\((\d+)/(\d+)\):")
TRAIN_ACC_RE = re.compile(r"Train Instance Accuracy:\s+([0-9.]+)")
TEST_ACC_RE = re.compile(r"Test Instance Accuracy:\s+([0-9.]+), Class Accuracy:\s+([0-9.]+)")
BEST_ACC_RE = re.compile(r"Best Instance Accuracy:\s+([0-9.]+), Class Accuracy:\s+([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser("Visualize training logs")
    parser.add_argument(
        "--log_dirs",
        nargs="+",
        required=True,
        help="experiment directory names under log/classification",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("log", "classification", "plots"),
        help="directory to save figures",
    )
    return parser.parse_args()


def find_log_file(experiment_dir):
    logs_dir = os.path.join(experiment_dir, "logs")
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Missing logs folder: {logs_dir}")

    log_files = sorted(
        f for f in os.listdir(logs_dir)
        if os.path.isfile(os.path.join(logs_dir, f)) and f.endswith(".txt")
    )
    if not log_files:
        raise FileNotFoundError(f"No .txt log file found in {logs_dir}")

    return os.path.join(logs_dir, log_files[0])


def parse_log(log_path):
    epochs = []
    train_acc = []
    test_acc = []
    class_acc = []
    best_acc = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
                continue

            train_match = TRAIN_ACC_RE.search(line)
            if train_match:
                train_acc.append(float(train_match.group(1)))
                continue

            test_match = TEST_ACC_RE.search(line)
            if test_match:
                test_acc.append(float(test_match.group(1)))
                class_acc.append(float(test_match.group(2)))
                continue

            best_match = BEST_ACC_RE.search(line)
            if best_match:
                best_acc.append(float(best_match.group(1)))

    usable_len = min(len(epochs), len(train_acc), len(test_acc), len(class_acc))
    if usable_len == 0:
        raise ValueError(f"Could not parse training curves from {log_path}")

    data = {
        "epochs": epochs[:usable_len],
        "train_acc": train_acc[:usable_len],
        "test_acc": test_acc[:usable_len],
        "class_acc": class_acc[:usable_len],
        "best_acc": best_acc[:usable_len] if best_acc else [],
    }
    return data


def save_single_plot(name, data, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["epochs"], data["train_acc"], label="Train Instance Accuracy", linewidth=2)
    ax.plot(data["epochs"], data["test_acc"], label="Test Instance Accuracy", linewidth=2)
    ax.plot(data["epochs"], data["class_acc"], label="Test Class Accuracy", linewidth=2)
    if data["best_acc"]:
        ax.plot(data["epochs"][:len(data["best_acc"])], data["best_acc"], label="Best Instance Accuracy", linestyle="--")

    ax.set_title(f"Training Curves: {name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{name}_training_curves.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_comparison_plot(all_runs, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in all_runs.items():
        ax.plot(data["epochs"], data["test_acc"], label=name, linewidth=2)

    ax.set_title("Test Instance Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = os.path.join(output_dir, "comparison_test_instance_accuracy.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_csv(name, data, output_dir):
    csv_path = os.path.join(output_dir, f"{name}_training_curves.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_instance_acc,test_instance_acc,test_class_acc,best_instance_acc\n")
        for idx, epoch in enumerate(data["epochs"]):
            best_value = ""
            if idx < len(data["best_acc"]):
                best_value = f"{data['best_acc'][idx]:.6f}"
            f.write(
                f"{epoch},{data['train_acc'][idx]:.6f},{data['test_acc'][idx]:.6f},"
                f"{data['class_acc'][idx]:.6f},{best_value}\n"
            )
    return csv_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise SystemExit("matplotlib is required. Install it with: pip install matplotlib")

    all_runs = {}
    for log_dir in args.log_dirs:
        experiment_dir = os.path.join("log", "classification", log_dir)
        log_path = find_log_file(experiment_dir)
        data = parse_log(log_path)
        all_runs[log_dir] = data

        png_path = save_single_plot(log_dir, data, args.output_dir)
        csv_path = save_csv(log_dir, data, args.output_dir)
        print(f"Saved plot: {png_path}")
        print(f"Saved csv: {csv_path}")

    if len(all_runs) >= 2:
        comparison_path = save_comparison_plot(all_runs, args.output_dir)
        print(f"Saved comparison plot: {comparison_path}")


if __name__ == "__main__":
    main()
