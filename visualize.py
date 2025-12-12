import matplotlib.pyplot as plt
import torch
import numpy as np

CLASSES = [
    'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
    'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
]

def plot_detection(img, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores=None, thresh=0.2):
    img = img.permute(1,2,0).cpu().numpy().copy()
    img = np.clip(img*255, 0, 255).astype(np.uint8)
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    ax = plt.gca()
    # GT
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='lime', linewidth=2))
        ax.text(x1, y1-2, CLASSES[label], fontsize=9, color='lime', bbox=dict(facecolor='black', alpha=0.3))
    # Predictions
    if pred_boxes is not None:
        for i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
            x1, y1, x2, y2 = box
            if pred_scores is not None:
                score = pred_scores[i]
                if score < thresh: continue
                labelstr = f'{CLASSES[label]}: {score:.2f}'
            else:
                labelstr = CLASSES[label]
            ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='yellow', linestyle='--', linewidth=2))
            ax.text(x1, y2+8, labelstr, fontsize=9, color='yellow', bbox=dict(facecolor='black', alpha=0.3))
    plt.axis('off')
    plt.show()
