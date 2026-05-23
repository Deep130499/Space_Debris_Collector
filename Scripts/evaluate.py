"""
Evaluate a trained YOLO model on the test set.

This script:
- Loads the best model from a given experiment folder.
- Runs evaluation on the test split (as defined in the dataset YAML).
- Prints test metrics (mAP50, mAP50-95, precision, recall, per‑class AP).
- Generates training curves from results.csv.
- Displays (or saves) the confusion matrix.
- Produces a grid of sample predictions from the test set.
- Saves all generated plots in the experiment folder.

Usage:
    python evaluate.py --exp_path /path/to/experiment
    python evaluate.py --exp_path runs/detect/my_exp --yaml data.yaml --test_img_dir /path/to/test/images
"""

import os
import argparse
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on test set.")
    parser.add_argument("--exp_path", required=True, help="Path to experiment folder (e.g., runs/detect/exp).")
    parser.add_argument("--yaml", default=None, help="Path to dataset YAML file (if not in exp_path).")
    parser.add_argument("--test_img_dir", default=None, help="Directory containing test images (if not derived from YAML).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for visualizations.")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of test images to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Construct paths
    best_model = os.path.join(args.exp_path, "weights", "best.pt")
    if not os.path.exists(best_model):
        raise FileNotFoundError(f"Best model not found at {best_model}")

    # Determine dataset YAML
    if args.yaml is None:
        # Try common locations
        possible_yaml = [
            os.path.join(args.exp_path, "data.yaml"),
            "/content/balanced_dataset.yaml",
        ]
        yaml_path = None
        for p in possible_yaml:
            if os.path.exists(p):
                yaml_path = p
                break
        if yaml_path is None:
            raise ValueError("Could not locate dataset YAML. Please specify with --yaml.")
    else:
        yaml_path = args.yaml
    print(f"Using dataset YAML: {yaml_path}")

    # Determine test image directory
    if args.test_img_dir is None:
        # Attempt to derive from YAML (if it contains a 'test' path)
        import yaml
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        base_path = data_yaml.get('path', '.')
        test_rel = data_yaml.get('test', 'images/test')
        test_img_dir = os.path.join(base_path, test_rel)
        if not os.path.exists(test_img_dir):

            test_img_dir = "/content/yolo_balanced/images/test"
    else:
        test_img_dir = args.test_img_dir

    print(f"Test images directory: {test_img_dir}")

    # Load model
    model = YOLO(best_model)
    print("Model loaded.")

    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    # Use plots=True to save confusion matrix and PR curve automatically
    test_metrics = model.val(data=yaml_path, split='test', plots=True)

    # Print metrics
    print(f"Test mAP50: {test_metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {test_metrics.box.map:.4f}")
    print(f"Test Precision: {test_metrics.box.mp:.4f}")
    print(f"Test Recall: {test_metrics.box.mr:.4f}")
    print("\nPer‑class Test AP:")
    for i, class_name in enumerate(['non_debris', 'debris']):
        print(f"  {class_name}: AP50={test_metrics.box.ap50[i]:.4f}, AP50-95={test_metrics.box.ap[i]:.4f}")

    # Confusion matrix (already saved by YOLO)
    cm_path = os.path.join(args.exp_path, "confusion_matrix.png")
    if os.path.exists(cm_path):
        print(f"Confusion matrix saved at: {cm_path}")
    else:
        print("Confusion matrix not found (YOLO may not have saved it).")

    # Plot training curves from results.csv
    results_csv = os.path.join(args.exp_path, "results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        metrics_to_plot = [
            ('train/box_loss', 'Box Loss'),
            ('train/cls_loss', 'Classification Loss'),
            ('train/dfl_loss', 'DFL Loss'),
            ('metrics/precision(B)', 'Precision'),
            ('metrics/recall(B)', 'Recall'),
            ('metrics/mAP50(B)', 'mAP50')
        ]
        for i, (col, title) in enumerate(metrics_to_plot):
            if col in df.columns:
                axes[i].plot(df['epoch'], df[col], marker='.', linestyle='-')
                axes[i].set_xlabel('Epoch')
                axes[i].set_title(title)
                axes[i].grid(True)
        plt.tight_layout()
        curves_path = os.path.join(args.exp_path, "training_curves.png")
        plt.savefig(curves_path, dpi=150)
        print(f"Training curves saved to: {curves_path}")
        plt.close()
    else:
        print("results.csv not found; training curves not generated.")

    # Visualize sample test predictions
    if not os.path.exists(test_img_dir):
        print(f"Test image directory {test_img_dir} not found. Skipping prediction visualizations.")
        return

    test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        print("No test images found. Skipping prediction visualizations.")
        return

    # Sample a few images
    sample_images = random.sample(test_images, min(args.num_samples, len(test_images)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    for i, img_path in enumerate(sample_images):
        results = model(img_path, conf=args.conf)
        annotated = results[0].plot()   # returns BGR numpy array
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[i].imshow(annotated_rgb)
        axes[i].set_title(os.path.basename(img_path))
        axes[i].axis('off')
    plt.tight_layout()
    pred_path = os.path.join(args.exp_path, "test_predictions.png")
    plt.savefig(pred_path, dpi=150)
    print(f"Sample predictions saved to: {pred_path}")
    plt.close()

    print("Evaluation completed.")

if __name__ == "__main__":
    main()
