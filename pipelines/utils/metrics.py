import numpy as np
import torch
import math
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd


def get_predictions_and_probs(model, data_loader, device):
    predictions = []
    true_labels = []
    all_probs = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for xi, yi in data_loader:
            xi = xi.to(device, dtype=torch.float32)
            yi = yi.to(device, dtype=torch.long)

            scores = model(xi)  # Raw scores (logits)
            probs = torch.nn.functional.softmax(scores, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(scores, 1)  # Get hard class labels

            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(yi.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())  # Store probabilities

    return predictions, true_labels, np.array(all_probs)


def plot_all_class_roc(y_true, y_prob, class_names=None, per_row=3):
    """
    Plots the ROC curve for all classes in a grid layout.

    Args:
        y_true (array-like): True class labels (integer values from 0 to num_classes-1).
        y_prob (array-like): Predicted probabilities of shape (n_samples, n_classes).
        class_names (dict, optional): Mapping of class indices to class names.
        per_row (int): Number of plots per row.

    Returns:
        dict: AUC values for each class.
    """
    num_classes = y_prob.shape[1]
    cols = per_row
    rows = math.ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    roc_auc = {}

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc[i] = auc(fpr, tpr)

        ax = axes[i]
        class_label = class_names[i] if class_names and i in class_names else f"Class {i}"

        ax.plot(fpr, tpr, lw=2, label=f"{class_label} (AUC = {roc_auc[i]:.2f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=2)  # Random chance line

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve: {class_label}")
        ax.legend(loc="lower right")

    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    return roc_auc


def plot_single_class_roc(y_true, y_prob, class_idx, class_names=None):
    """
    Plots the ROC curve for a single class.

    Args:
        y_true (array-like): True class labels (integer values from 0 to num_classes-1).
        y_prob (array-like): Predicted probabilities of shape (n_samples, n_classes).
        class_idx (int): The class index to highlight.
        class_names (dict, optional): Mapping of class indices to class names.
    """
    fpr, tpr, _ = roc_curve(y_true == class_idx, y_prob[:, class_idx])
    roc_auc = auc(fpr, tpr)

    class_label = class_names[class_idx] if class_names and class_idx in class_names else f"Class {class_idx}"

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"{class_label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=2)  # Random chance line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {class_label}")
    plt.legend(loc="lower right")
    plt.show()


def plot_macro_roc(y_true, y_prob, class_names=None):
    """
    Plots the macro-averaged ROC curve.

    Args:
        y_true (array-like): True class labels (integer values from 0 to num_classes-1).
        y_prob (array-like): Predicted probabilities of shape (n_samples, n_classes).
        class_names (dict, optional): Mapping of class indices to class names.
    """
    num_classes = y_prob.shape[1]

    all_fpr = np.unique(np.concatenate([roc_curve(y_true == i, y_prob[:, i])[0] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= num_classes  # Compute macro-average TPR
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(all_fpr, mean_tpr, lw=2, label=f"Macro-Average (AUC = {macro_auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=2)  # Random chance line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-Averaged ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    return macro_auc


def compute_classification_metrics(c_matrix, class_names=None):
    num_classes = c_matrix.shape[0]

    # Initialize storage
    metrics = {
        "Class": [],
        "Recall (Sensitivity)": [],
        "Miss Rate": [],
        "Specificity (Selectivity)": [],
        "Fallout": [],
        "Precision": [],
        "False Discovery Rate (FDR)": [],
        "Negative Predictive Value (NPV)": [],
        "False Omission Rate (FOR)": [],
        "F1 Score": []
    }
    TP_total = FP_total = FN_total = TN_total = 0

    for i in range(num_classes):
        TP = c_matrix[i, i]
        FN = np.sum(c_matrix[i, :]) - TP
        FP = np.sum(c_matrix[:, i]) - TP
        TN = np.sum(c_matrix) - (TP + FN + FP)

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        miss_rate = 1 - recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        fallout = 1 - specificity
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        fdr = 1 - precision
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        for_rate = 1 - npv

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # balanced_acc = (recall + specificity) / 2
        TP_total += TP
        FP_total += FP
        FN_total += FN
        TN_total += TN

        class_label = class_names[i] if class_names and i in class_names else f"Class {i}"
        metrics["Class"].append(class_label)
        metrics["Recall (Sensitivity)"].append(recall)
        metrics["Miss Rate"].append(miss_rate)
        metrics["Specificity (Selectivity)"].append(specificity)
        metrics["Fallout"].append(fallout)
        metrics["Precision"].append(precision)
        metrics["False Discovery Rate (FDR)"].append(fdr)
        metrics["Negative Predictive Value (NPV)"].append(npv)
        metrics["False Omission Rate (FOR)"].append(for_rate)
        metrics["F1 Score"].append(f1)

    #        metrics["Balanced Accuracy"].append(balanced_acc)

    df_per_class = pd.DataFrame(metrics).set_index("Class")

    # Compute macro-averaged values
    df_macro = df_per_class.mean(numeric_only=True).to_frame().T

    # Reset index to remove any unnecessary labels
    df_macro.reset_index(drop=True, inplace=True)
    df_macro = df_macro.T
    df_macro.columns = ["Valor"]  # Rename the single column to avoid "0" header
    # Compute Overall MCC
    mcc = ((TP_total * TN_total) - (FP_total * FN_total)) / np.sqrt(
        (TP_total + FP_total) * (TP_total + FN_total) * (TN_total + FP_total) * (TN_total + FN_total)
    ) if (TP_total + FP_total) * (TP_total + FN_total) * (TN_total + FP_total) * (TN_total + FN_total) > 0 else 0

    df_macro.loc["MCC"] = mcc
    return df_per_class, df_macro


def normal_cm(y_real, y_pred, class_labels=None):
    """
    Standard confusion matrix with counts and percentages for all classes.
    """
    cm = confusion_matrix(y_real, y_pred)
    num_classes = cm.shape[0]

    if class_labels is None:
        class_labels = [f"Clase {i}" for i in range(num_classes)]

    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # Normalize per class

    # Create cell labels
    cell_labels = [[f"{cm[i, j]}\n{cm_percent[i, j]:.1f}%" for j in range(num_classes)] for i in range(num_classes)]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_percent, annot=np.array(cell_labels), fmt='', cmap="Pastel1", xticklabels=class_labels,
                     yticklabels=class_labels, cbar=False)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix with Counts and Percentages")
    plt.show()

    return cm