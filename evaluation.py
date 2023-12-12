# Author: Luke Patterson
# Name: evaluation.py
# Description: This file contains code for evaluating the model.

import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import resize_and_normalize


# Calculate custom F1 scores
def calculate_custom_f1_scores(cm):
    """
    Calculate the F1 score for each class and the average F1 score (the final challenge score).
    :param cm: The confusion matrix.
    :return F1_scores, avg_F1_score: F1_scores, the F1 scores for each class. avg_F1_score, the average F1 score.
    """
    F1_scores = {}
    for i in range(len(cm)):
        F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i])) if np.sum(cm[i, :]) + np.sum(cm[:, i]) > 0 else 0
        F1_scores[f'F1_{i}'] = F1
    avg_F1_score = np.mean(list(F1_scores.values()))
    return F1_scores, avg_F1_score


# Plot and save ROC curve
def plot_roc_curve(y_test_categorical, y_pred_raw, class_names):
    """
    Plot and save a ROC curve.
    :param y_test_categorical: The true labels.
    :param y_pred_raw: The predicted labels.
    :param class_names: The class names.
    :return: None, saves the ROC curve to a file.
    """
    n_classes = y_test_categorical.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_categorical[:, i], y_pred_raw[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))

    # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=4)

    # Plot ROC curve for each class
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('model_evaluation/multi_class_roc.png')


def plot_confusion_matrix(cm, class_names):
    """
    Plot and save a confusion matrix.
    :param cm: The confusion matrix.
    :param class_names: The class names.
    :return: None, saves the confusion matrix to a file.
    """
    # Calculate sums for rows and columns
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    total = np.sum(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.yticks(rotation=0)
    plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, left=False, labeltop=True)
    ax.set_xlabel('Predicted Classification', labelpad=30, fontsize=20)
    ax.set_ylabel('Reference Classification', labelpad=30, fontsize=20)
    plt.title('Confusion Matrix', fontsize=20)

    # Add row totals to the right of the heatmap
    for i, value in enumerate(row_sums):
        plt.text(len(cm) + 0.1, i + 0.5, value, va='center')

    # Add column totals below the heatmap
    for i, value in enumerate(col_sums):
        plt.text(i + 0.5, len(cm) + 0.1, value, ha='center')

    # Add total to the bottom right corner
    plt.text(len(cm) + 0.1, len(cm) + 0.1, str(total), ha='center', va='center')

    plt.savefig('model_evaluation/test_data_confusion_matrix.png')


# Create and save a table
def save_metrics_table(data, filename, title):
    """
    Create and save a table with the given data.
    :param data: Text to place into table cells.
    :param filename: The filename to save the table to.
    :param title: The title of the table.
    :return: None, saves the table to a file.
    """
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data, colLabels=["Metric", "Value"], loc="center", cellLoc='center')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'model_evaluation/{filename}.png')


if __name__ == '__main__':
    # Load model and test data
    model_path = 'models/model_20231210-194623_F1_0_5953.h5'
    model = tf.keras.models.load_model(model_path)
    with open('split_data/test.pkl', 'rb') as f:
        test = pickle.load(f)
    X_test, y_test = test

    # Preprocess test data
    X_test_processed = resize_and_normalize(X_test)
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded)

    # Predict test data
    y_pred_raw = model.predict(X_test_processed)
    y_pred = np.argmax(y_pred_raw, axis=1)

    # Calculate metrics
    cm = confusion_matrix(y_test_encoded, y_pred)
    F1_scores, avg_F1_score = calculate_custom_f1_scores(cm)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision_micro = precision_score(y_test_encoded, y_pred, average='macro')
    recall_micro = recall_score(y_test_encoded, y_pred, average='macro')
    auc_micro = roc_auc_score(y_test_categorical, y_pred_raw, multi_class='ovo', average='macro')

    # Save overall metrics table
    overall_metrics = [
        ["F1 Score", f"{avg_F1_score:.2f}"],
        ["Accuracy", f"{accuracy:.2f}"],
        ["Precision (macro)", f"{precision_micro:.2f}"],
        ["Recall (macro)", f"{recall_micro:.2f}"],
        ["AUC (macro)", f"{auc_micro:.2f}"]
    ]
    save_metrics_table(overall_metrics, "overall_metrics", "Overall Metrics")

    # Save class names
    class_names = le.classes_.tolist()

    # Save F1 scores for each class table
    class_f1_scores = [[class_names[int(k.split('_')[1])], f"{v:.2f}"] for k, v in F1_scores.items()]
    save_metrics_table(class_f1_scores, "class_f1_scores", "F1 Scores for Each Class")

    # Save confusion matrix
    plot_confusion_matrix(cm, class_names)

    # Save ROC curve
    class_names = le.classes_.tolist()
    plot_roc_curve(y_test_categorical, y_pred_raw, class_names)
