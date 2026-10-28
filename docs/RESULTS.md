# Experimental Results

## Setup

- Dataset: ModelNet40
- Input points: 1,024
- Sampling: farthest point sampling (FPS)
- Evaluation: 3-vote aggregation
- Inputs: XYZ coordinates without normals

## Test results

| Experiment | Model | Instance Accuracy | Class Accuracy |
| --- | --- | ---: | ---: |
| `pointnet_cls_fps` | PointNet | 0.900566 | 0.863945 |
| `pointnet2_cls_ssg_fps` | PointNet++ SSG | 0.923544 | 0.885346 |
| `dgcnn_cls_fps` | DGCNN | **0.928479** | **0.897359** |
| `repo_cls_ssg_fps` | PointNet++ SSG (comparison run) | 0.924743 | 0.890435 |

The table reports the final values recorded in each experiment's `eval.txt`. Analysis runs may differ slightly because evaluation and aggregation settings can change between runs.

## Generated analysis

`analyze.py` writes the following files to `log/classification/<experiment>/analysis/`:

- `summary.txt`
- `confusion_matrix.csv`
- `per_class_accuracy.csv`
- `top_confusions.csv`
- `misclassified_cases.csv`

Full logs, checkpoints, and generated analysis files are intentionally excluded from Git. The comparison figure committed under `docs/assets/` is a lightweight snapshot for the project page.
