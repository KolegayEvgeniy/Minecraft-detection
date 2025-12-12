import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.notebook import tqdm
import torch.nn.functional as F
from losses import sigmoid_focal_loss, compute_bbox_loss
from models.fcos import FCOSDetector
from data.dataset import VOCDataset, collate_fn
from config import CONFIG
import matplotlib.pyplot as plt
import numpy as np

# Функция генерации anchor/grid по feature map

def generate_grid(feat_h, feat_w, stride, device):
    yv, xv = torch.meshgrid(
        [torch.arange(feat_h, device=device), torch.arange(feat_w, device=device)], indexing='ij')
    grid = torch.stack([xv, yv], dim=-1).reshape(-1, 2) * stride
    return grid # [[x, y], ...]

def assign_targets(boxes, labels, feat_size, img_size, stride, num_classes):
    # boxes - [N, 4] (xyxy), labels - [N]
    # Возвращает:
    #  cls_target: [H*W, num_classes], bbox_target: [H*W, 4], mask: [H*W]
    h, w = feat_size
    grid = generate_grid(h, w, stride, boxes.device)  # [[x, y]] координаты центров
    num_anc = grid.shape[0]
    cls_target = torch.zeros((num_anc, num_classes), device=boxes.device)
    bbox_target = torch.zeros((num_anc, 4), device=boxes.device)
    mask = torch.zeros((num_anc,), dtype=torch.bool, device=boxes.device)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box  # (xmin, ymin, xmax, ymax)
        inside = (grid[:, 0] >= x1) & (grid[:, 0] <= x2) & (grid[:, 1] >= y1) & (grid[:, 1] <= y2)
        inds = torch.where(inside)[0]
        if len(inds) > 0:
            cls_target[inds, label] = 1.0
            # regression: для всех индексов - от центра anchor до границ bbox (в пикселях)
            bbox_target[inds, 0] = grid[inds, 0] - x1  # left
            bbox_target[inds, 1] = grid[inds, 1] - y1  # top
            bbox_target[inds, 2] = x2 - grid[inds, 0]  # right
            bbox_target[inds, 3] = y2 - grid[inds, 1]  # bottom
            mask[inds] = 1
    return cls_target, bbox_target, mask

def plot_metrics(loss_history, bbox_loss_history, cls_loss_history):
    plt.figure(figsize=(10,6))
    plt.plot(loss_history, label='Total Loss')
    plt.plot(bbox_loss_history, label='BBox Loss')
    plt.plot(cls_loss_history, label='Cls Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FCOS Losses by Epoch')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    model = FCOSDetector(CONFIG['num_classes']).to(device)

    train_dataset = VOCDataset(CONFIG['data_dir'], 'train', CONFIG['img_size'])
    val_dataset = VOCDataset(CONFIG['data_dir'], 'valid', CONFIG['img_size'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    stride = 64  # output map: (img_size/stride, img_size/stride)
    loss_history, bbox_loss_history, cls_loss_history = [], [], []
    map_history, map50_history, prec_history, rec_history, f1_history = [], [], [], [], []
    for epoch in range(CONFIG['epochs']):
        model.train()
        losses = []
        losses_bbox = []
        losses_cls = []
        for imgs, boxes_batch, labels_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            imgs = imgs.to(device)
            boxes = boxes_batch[0].to(device)
            labels = labels_batch[0].to(device)
            logits, bbox_pred = model(imgs)  # [B, C, H, W], [B, 4, H, W]
            B, C, H, W = logits.shape
            feat_size = (H, W)
            logits = logits.permute(0,2,3,1).reshape(-1, CONFIG['num_classes'])
            bbox_pred = bbox_pred.permute(0,2,3,1).reshape(-1, 4)
            # target assignment:
            cls_target, bbox_target, mask = assign_targets(boxes, labels, feat_size, CONFIG['img_size'], stride, CONFIG['num_classes'])
            loss_cls = sigmoid_focal_loss(logits, cls_target)
            loss_bbox = compute_bbox_loss(bbox_pred, bbox_target, mask)
            loss = loss_cls + loss_bbox
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            losses_bbox.append(loss_bbox.item())
            losses_cls.append(loss_cls.item())
        mean_loss = sum(losses)/max(1,len(losses))
        mean_bbox = sum(losses_bbox)/max(1,len(losses_bbox))
        mean_cls = sum(losses_cls)/max(1,len(losses_cls))
        print(f'Epoch: {epoch+1}, Loss: {mean_loss:.4f}, Cls: {mean_cls:.4f}, BBox: {mean_bbox:.4f}')
        loss_history.append(mean_loss)
        bbox_loss_history.append(mean_bbox)
        cls_loss_history.append(mean_cls)

        # --- Валидация ---
        model.eval()
        pred_all, gt_all = [], []
        for imgs, boxes_batch, labels_batch in DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn):
            img = imgs[0].to(device)
            gt_boxes = boxes_batch[0]
            gt_labels = labels_batch[0]
            with torch.no_grad():
                pred_boxes, pred_scores, pred_labels = model.detect(img, score_thr=0.05, iou_thr=0.5, stride=stride)
            # Перевод в numpy
            pred_boxes = pred_boxes.cpu().numpy() if len(pred_boxes) > 0 else np.zeros((0,4))
            pred_labels = pred_labels.cpu().numpy() if len(pred_labels) > 0 else np.zeros(0, dtype=int)
            pred_scores = pred_scores.cpu().numpy() if len(pred_scores) > 0 else np.zeros(0)
            gt_boxes = np.array(gt_boxes) if len(gt_boxes) > 0 else np.zeros((0,4))
            gt_labels = np.array(gt_labels) if len(gt_labels) > 0 else np.zeros(0, dtype=int)
            pred_all.append((pred_boxes, pred_labels, pred_scores))
            gt_all.append((gt_boxes, gt_labels))
        # --- Подсчёт метрик ---
        def calc_map_metrics(pred_all, gt_all, num_classes=CONFIG['num_classes'], iou_thr=0.5):
            TP, FP, npos = 0, 0, 0
            scores_all = []
            matches = []
            for preds, gts in zip(pred_all, gt_all):
                pred_boxes, pred_labels, pred_scores = preds
                gt_boxes, gt_labels = gts
                detected = set()
                npos += len(gt_boxes)
                for i, (pb, pl, ps) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                    ious = [bb_iou(pb, gb) for gb in gt_boxes] if len(gt_boxes) else []
                    match_gt = -1
                    best_iou = 0
                    for j, (iou, gl) in enumerate(zip(ious, gt_labels)):
                        if pl == gl and iou > iou_thr and j not in detected and iou > best_iou:
                            match_gt = j; best_iou = iou
                    if match_gt >= 0:
                        TP += 1
                        detected.add(match_gt)
                        matches.append(1)
                    else:
                        FP += 1
                        matches.append(0)
                    scores_all.append(ps)
            scores_all = np.array(scores_all)
            matches = np.array(matches)
            if len(matches) > 0:
                sort_ind = np.argsort(-scores_all)
                matches = matches[sort_ind]
                scores_all = scores_all[sort_ind]
                tp_cum = np.cumsum(matches)
                fp_cum = np.cumsum(1 - matches)
                rec = tp_cum / (npos + 1e-8)
                prec = tp_cum / (tp_cum + fp_cum + 1e-8)
                AP = compute_ap_simple(rec, prec)
            else:
                rec = np.array([])
                prec = np.array([])
                AP = 0.0
            Recall = rec[-1] if len(rec) > 0 else 0.0
            Precision = prec[-1] if len(prec) > 0 else 0.0
            F1 = 2 * Precision * Recall / (Precision + Recall + 1e-8)
            return AP, Precision, Recall, F1
        def bb_iou(boxA, boxB):
            # [xmin, ymin, xmax, ymax]
            xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
            return iou
        def compute_ap_simple(rec, prec):
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            return ap
        # mAP@0.5
        map50, prec, rec, f1 = calc_map_metrics(pred_all, gt_all, iou_thr=0.5)
        # mAP@[.5:.95:0.05]
        ap_all = []
        for t in np.arange(0.5, 1.0, 0.05):
            ap, _, _, _ = calc_map_metrics(pred_all, gt_all, iou_thr=t)
            ap_all.append(ap)
        mean_ap = np.mean(ap_all)
        map_history.append(mean_ap)
        map50_history.append(map50)
        prec_history.append(prec)
        rec_history.append(rec)
        f1_history.append(f1)
        print(f'Валидация: mAP: {mean_ap:.4f}, mAP@50: {map50:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'fcos_minecraft.pt'))
    print('Модель сохранена!')

    # Сохранение истории метрик в CSV
    import pandas as pd
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(loss_history)+1)),
        'total_loss': loss_history,
        'bbox_loss': bbox_loss_history,
        'cls_loss': cls_loss_history,
        'mAP': map_history,
        'mAP_50': map50_history,
        'Precision': prec_history,
        'Recall': rec_history,
        'F1-score': f1_history
    })
    metrics_df.to_csv(os.path.join(CONFIG['output_dir'], 'metrics.csv'), index=False)
    print('Метрики сохранены в metrics.csv')

    plot_metrics(loss_history, bbox_loss_history, cls_loss_history)

