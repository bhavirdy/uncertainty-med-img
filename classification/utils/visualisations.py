import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def plot_metrics(train_losses, val_losses, train_accs, val_accs,
                 train_prec, val_prec,
                 train_rec, val_rec,
                 train_f1, val_f1,
                 dataset_name='dataset',
                 model_path='model.pth',
                 all_labels=None,
                 all_preds=None,
                 class_names=None):
    """
    Plots Loss, Accuracy, Precision, Recall, F1-score and confusion matrix.
    If all_labels and all_preds are provided, also plots the confusion matrix.
    """
    epochs = range(1, len(train_losses)+1)
    n_subplots = 4  # Loss, Accuracy, Precision, Recall & F1

    plt.figure(figsize=(12, 20))

    # Extract model filename for main title
    model_name = os.path.basename(model_path)

    # Main figure title
    plt.suptitle(f'Dataset: {dataset_name} | Model: {model_name}', fontsize=16, y=0.95)

    # Loss
    plt.subplot(n_subplots, 1, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    # Accuracy
    plt.subplot(n_subplots, 1, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    # Precision
    plt.subplot(n_subplots, 1, 3)
    plt.plot(epochs, train_prec, label='Train Precision')
    plt.plot(epochs, val_prec, label='Val Precision')
    plt.legend()
    plt.title('Precision')

    # Recall & F1
    plt.subplot(n_subplots, 1, 4)
    plt.plot(epochs, train_rec, label='Train Recall', linestyle='--')
    plt.plot(epochs, val_rec, label='Val Recall', linestyle='--')
    plt.plot(epochs, train_f1, label='Train F1', linestyle=':')
    plt.plot(epochs, val_f1, label='Val F1', linestyle=':')
    plt.legend()
    plt.title('Recall & F1-score')

    # Ensure output directory exists
    save_dir = './outputs/visualisations/'
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics plot
    metrics_path = os.path.join(save_dir, f'metrics_{dataset_name}_{model_name}.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Confusion matrix ---
    if all_labels is not None and all_preds is not None:
        cm = confusion_matrix(all_labels, all_preds)
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Dataset: {dataset_name} | Model: {model_name} - Confusion Matrix')

        cm_path = os.path.join(save_dir, f'confusion_matrix_{dataset_name}_{model_name}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
