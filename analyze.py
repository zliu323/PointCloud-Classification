import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.ModelNetDataLoader import ModelNetDataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.append(MODELS_DIR)


def parse_args():
    parser = argparse.ArgumentParser("Analyze classification failures")
    parser.add_argument("--use_cpu", action="store_true", default=False, help="use cpu mode")
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--num_category", default=40, type=int, choices=[10, 40], help="ModelNet10/40")
    parser.add_argument("--num_point", type=int, default=1024, help="point number")
    parser.add_argument("--log_dir", type=str, required=True, help="experiment directory name under log/classification")
    parser.add_argument("--model", type=str, default=None, help="override model name if auto-detect fails")
    parser.add_argument("--use_normals", action="store_true", default=False, help="use normals")
    parser.add_argument("--use_uniform_sample", action="store_true", default=False, help="use FPS sampling")
    parser.add_argument("--num_votes", type=int, default=3, help="voting count")
    parser.add_argument("--top_k", type=int, default=20, help="number of failure cases to save")
    parser.add_argument("--data_path", type=str, default="data/modelnet40_normal_resampled", help="ModelNet dataset directory")
    parser.add_argument("--num_workers", type=int, default=0, help="number of data loading workers")
    return parser.parse_args()


def detect_model_name(experiment_dir, override=None):
    if override:
        return override

    logs_dir = os.path.join(experiment_dir, "logs")
    if os.path.isdir(logs_dir):
        log_files = sorted(
            f for f in os.listdir(logs_dir)
            if os.path.isfile(os.path.join(logs_dir, f)) and f.endswith(".txt")
        )
        if log_files:
            return os.path.splitext(log_files[0])[0]

    raise ValueError(
        "Could not detect model name. Pass it explicitly with --model, "
        "for example --model pointnet2_cls_ssg."
    )


def find_checkpoint(experiment_dir):
    candidates = [
        os.path.join(experiment_dir, "checkpoints", "best_model.pth"),
        os.path.join(experiment_dir, "best_model.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Could not find best_model.pth in experiment directory.")


def write_confusion_matrix(path, confusion, class_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for idx, class_name in enumerate(class_names):
            writer.writerow([class_name] + confusion[idx].tolist())


def write_per_class(path, confusion, class_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "correct", "total", "accuracy"])
        for idx, class_name in enumerate(class_names):
            total = int(confusion[idx].sum())
            correct = int(confusion[idx, idx])
            accuracy = (correct / total) if total else 0.0
            writer.writerow([class_name, correct, total, f"{accuracy:.6f}"])


def write_failures(path, failures):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "shape_name", "file_path", "true_label", "pred_label"])
        for row in failures:
            writer.writerow(row)


def write_top_confusions(path, confusion, class_names, top_n=10):
    pairs = []
    num_classes = len(class_names)
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx])
            if count > 0:
                pairs.append((count, class_names[true_idx], class_names[pred_idx]))
    pairs.sort(reverse=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "true_label", "pred_label"])
        for count, true_label, pred_label in pairs[:top_n]:
            writer.writerow([count, true_label, pred_label])


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    experiment_dir = os.path.join("log", "classification", args.log_dir)
    checkpoint_path = find_checkpoint(experiment_dir)
    model_name = detect_model_name(experiment_dir, args.model)

    model_lib = __import__(model_name)
    classifier = model_lib.get_model(args.num_category, normal_channel=args.use_normals)
    classifier = classifier.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    dataset = ModelNetDataLoader(root=args.data_path, args=args, split="test", process_data=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_names = dataset.cat
    confusion = np.zeros((args.num_category, args.num_category), dtype=np.int64)
    failures = []
    mean_correct = []

    with torch.no_grad():
        sample_offset = 0
        for points, target in tqdm(loader, total=len(loader)):
            points = points.to(device)
            target = target.to(device)
            points = points.transpose(2, 1)

            vote_pool = torch.zeros(target.size(0), args.num_category, device=device)
            for _ in range(args.num_votes):
                pred, _ = classifier(points)
                vote_pool += pred
            pred_choice = (vote_pool / args.num_votes).data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size(0)))

            pred_np = pred_choice.cpu().numpy()
            target_np = target.cpu().numpy()
            batch_size = target_np.shape[0]

            for i in range(batch_size):
                true_idx = int(target_np[i])
                pred_idx = int(pred_np[i])
                confusion[true_idx, pred_idx] += 1

                if true_idx != pred_idx:
                    data_index = sample_offset + i
                    shape_name, file_path = dataset.datapath[data_index]
                    failures.append([
                        data_index,
                        shape_name,
                        file_path,
                        class_names[true_idx],
                        class_names[pred_idx],
                    ])

            sample_offset += batch_size

    class_accs = []
    for idx in range(args.num_category):
        total = confusion[idx].sum()
        class_accs.append((confusion[idx, idx] / total) if total else 0.0)

    instance_acc = float(np.mean(mean_correct))
    class_acc = float(np.mean(class_accs))

    analysis_dir = os.path.join(experiment_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    write_confusion_matrix(os.path.join(analysis_dir, "confusion_matrix.csv"), confusion, class_names)
    write_per_class(os.path.join(analysis_dir, "per_class_accuracy.csv"), confusion, class_names)
    write_failures(os.path.join(analysis_dir, "misclassified_cases.csv"), failures[:args.top_k])
    write_top_confusions(os.path.join(analysis_dir, "top_confusions.csv"), confusion, class_names)

    summary_path = os.path.join(analysis_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Instance Accuracy: {instance_acc:.6f}\n")
        f.write(f"Class Accuracy: {class_acc:.6f}\n")
        f.write(f"Total Misclassified Samples: {len(failures)}\n")

    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Instance Accuracy: {instance_acc:.6f}")
    print(f"Class Accuracy: {class_acc:.6f}")
    print(f"Saved analysis to: {analysis_dir}")


if __name__ == "__main__":
    main(parse_args())
