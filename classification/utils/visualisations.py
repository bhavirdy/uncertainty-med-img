import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(all_labels, all_preds,
                          model_path='model.pth',
                          class_names=None,
                          output_dir=None):
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Extract model filename for title
    model_name = os.path.basename(model_path)

    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    cm_path = os.path.join(output_dir, f'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()