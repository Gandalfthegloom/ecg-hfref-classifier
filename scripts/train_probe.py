# ML Packages
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# script imports & utilities
import sys
ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))
from src.utils import seed_everything

def load_and_scale_features(model_name: str, features_dir: Path):
    """
    Loads train, val, and test features for a given model and applies 
    StandardScaler (Z-score) normalization fitted ONLY on the training data.
    Returns NumPy arrays instead of PyTorch Tensors for scikit-learn compatibility.
    """
    # 1. Load dictionaries
    train_data = torch.load(features_dir / f"{model_name}_train_features.pt", weights_only=True)
    val_data = torch.load(features_dir / f"{model_name}_val_features.pt", weights_only=True)
    test_data = torch.load(features_dir / f"{model_name}_test_features.pt", weights_only=True)

    X_train, y_train = train_data["embeddings"], train_data["labels"]
    X_val, y_val = val_data["embeddings"], val_data["labels"]
    X_test, y_test = test_data["embeddings"], test_data["labels"]
    
    dim = train_data["dim"]
    
    # 2. Normalize Features (Fit on train, apply to val/test)
    train_mean = X_train.mean(dim=0, keepdim=True)
    train_std = X_train.std(dim=0, keepdim=True) + 1e-7 # add small epsilon
    
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    # 3. Convert to NumPy for Sklearn
    return (
        (X_train.numpy(), y_train.numpy()), 
        (X_val.numpy(), y_val.numpy()), 
        (X_test.numpy(), y_test.numpy())
    )

def train_and_evaluate(model_name: str, features_dir: Path):
    """
    Trains a deterministic scikit-learn Logistic Regression probe.
    Performs grid search for regularization, locks thresholds on Validation,
    and cleanly evaluates on Test.
    """
    print(f"\n{'='*40}\n Evaluating Pipeline: {model_name.upper()}\n{'='*40}")
    
    train_set, val_set, test_set = load_and_scale_features(model_name, features_dir)
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    # 1. Grid Search for Regularization Strength (C is inverse of weight_decay)
    C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_val_auc = 0
    best_c = None
    best_probe = None

    for c in C_values:
        # L-BFGS is the standard deterministic solver for linear evaluation
        probe = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, random_state=42)
        probe.fit(X_train, y_train)
        
        # Predict on validation
        val_probs = probe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c
            best_probe = probe

    print(f" Best Val AUC: {best_val_auc:.4f} (C={best_c})")

    # 2. Threshold Tuning on Validation (Fixing Data Leakage)
    val_probs = best_probe.predict_proba(X_val)[:, 1]
    fpr_val, tpr_val, thresh_val = roc_curve(y_val, val_probs)
    
    # Find threshold for ~90% Specificity (FPR <= 0.10)
    idx_90spec = np.where(fpr_val <= 0.10)[0][-1] 
    locked_thresh_90spec = thresh_val[idx_90spec]
    
    # Find threshold that maximizes F1 on validation
    best_val_f1 = 0
    locked_thresh_f1 = 0.5
    for t in thresh_val:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_val_f1:
            best_val_f1 = f1
            locked_thresh_f1 = t
            
    print(f" Locked Val Thresholds -> F1: {locked_thresh_f1:.4f}, Sens@90%Spec: {locked_thresh_90spec:.4f}")

    # 3. Test Evaluation using Locked Thresholds
    test_probs = best_probe.predict_proba(X_test)[:, 1]
    
    # AUC
    test_auc = roc_auc_score(y_test, test_probs)
    
    # F1 Score & Accuracy
    test_preds_f1 = (test_probs >= locked_thresh_f1).astype(int)
    test_f1 = f1_score(y_test, test_preds_f1)
    test_acc = accuracy_score(y_test, test_preds_f1)
    
    # Sens @ 90% Spec Target
    test_preds_90spec = (test_probs >= locked_thresh_90spec).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_90spec).ravel()
    test_sens = tp / (tp + fn)
    test_spec = tn / (tn + fp) # Actual Specificity achieved on test
    
    # ROC arrays for plotting
    fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
    
    # Calculate accuracy across a range of thresholds for plotting
    plot_thresholds = np.linspace(0, 1, 100)
    test_accuracies = [accuracy_score(y_test, (test_probs >= t).astype(int)) for t in plot_thresholds]

    print("\n Final Test Set Metrics:")
    print(f"  ROC-AUC:       {test_auc:.4f}")
    print(f"  F1-Score:      {test_f1:.4f} (at threshold {locked_thresh_f1:.2f})")
    print(f"  Accuracy:      {test_acc:.4f} (at threshold {locked_thresh_f1:.2f})")
    print(f"  Target 90%Spec Realized -> Sens: {test_sens:.4f}, Spec: {test_spec:.4f} (at threshold {locked_thresh_90spec:.2f})")

    return {
        "AUC": test_auc,
        "F1": test_f1,
        "Accuracy": test_acc,
        "Sens": test_sens,
        "Spec": test_spec,
        "fpr": fpr_test,
        "tpr": tpr_test,
        "probs": test_probs,
        "labels": y_test,
        "thresholds": plot_thresholds,
        "accuracies": test_accuracies
    }

def paired_bootstrap_auc(y_true, probs_a, probs_b, n_bootstraps=1000, seed=42):
    """
    Statistically robust test for the difference in AUCs on the same dataset.
    Bootstraps the differences (ΔAUC) rather than evaluating CIs independently.
    """
    rng = np.random.RandomState(seed)
    diffs = []
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, len(y_true), len(y_true))
        
        # Ensure bootstrap sample has both positive and negative classes
        if len(np.unique(y_true[indices])) < 2:
            continue 
            
        auc_a = roc_auc_score(y_true[indices], probs_a[indices])
        auc_b = roc_auc_score(y_true[indices], probs_b[indices])
        diffs.append(auc_a - auc_b)
        
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return np.mean(diffs), ci_lower, ci_upper

def main():
    seed_everything(42)
    features_dir = ROOT / "data" / "features"
    
    models_to_test = ["r2plus1d", "mvit"]
    results = {}

    # 1. Train and Evaluate Both Architectures
    for model_name in models_to_test:
        results[model_name] = train_and_evaluate(model_name, features_dir)

    # 2. Generate the "Money" Chart: Overlay ROC Curves
    print("\nGenerating ROC Curve Overlay...") 
    
    plt.figure(figsize=(8, 8))
    
    cnn = results["r2plus1d"]
    plt.plot(cnn["fpr"], cnn["tpr"], color='blue', lw=2, 
             label=f'R(2+1)D (AUC = {cnn["AUC"]:.3f})')
    
    mvit = results["mvit"]
    plt.plot(mvit["fpr"], mvit["tpr"], color='orange', lw=2, 
             label=f'MVIT v2 (AUC = {mvit["AUC"]:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Architectural Showdown: CNN vs. MVIT on Reduced EF', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    output_img_path = ROOT / "outputs" / "roc_curve_comparison.png"
    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    print(f"Saved artifact to: {output_img_path}")
    
    # 3. Proper Statistical Comparison (Paired Bootstrap)
    print("\n--- The Verdict ---")
    
    y_true = mvit["labels"] # Labels are identical for both models
    probs_mvit = mvit["probs"]
    probs_cnn = cnn["probs"]
    
    mean_diff, ci_lower, ci_upper = paired_bootstrap_auc(y_true, probs_mvit, probs_cnn)
    print(f"Mean Difference (MVIT - CNN): {mean_diff:.4f}")
    print(f"95% CI of Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if ci_lower > 0:
        print("Scenario A: The 'Transformer Triumph'. MVIT significantly outperformed the CNN.")
    elif ci_upper < 0:
        print("Scenario B: The 'CNN Stronghold'. R(2+1)D significantly outperformed the MVIT.")
    else:
        print("Statistical Tie: 0 is within the 95% CI of the difference. Neither architecture strictly dominated.")

if __name__ == "__main__":
    main()