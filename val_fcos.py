import torch
import numpy as np
from models.fcos import FCOSDetector
from data.dataset import VOCDataset, collate_fn
from config import CONFIG
import pandas as pd

def validate_fcos(val_split='valid', ckpt_path='artifacts/fcos_scratch/fcos_minecraft.pt',
                  iou_thr=0.5, score_thr=0.05, device=None):
    device = torch.device(device or (CONFIG['device'] if torch.cuda.is_available() else 'cpu'))
    model = FCOSDetector(CONFIG['num_classes']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    val_dataset = VOCDataset(CONFIG['data_dir'], split=val_split, img_size=CONFIG['img_size'])

    def bb_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

    def calc_map_metrics(pred_all, gt_all, iou_thr=0.5):
        TP, FP, npos = 0, 0, 0
        scores_all, matches = [], []
        for preds, gts in zip(pred_all, gt_all):
            pred_boxes, pred_labels, pred_scores = preds
            gt_boxes, gt_labels = gts
            detected = set()
            npos += len(gt_boxes)
            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
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
            def compute_ap_simple(rec, prec):
                mrec = np.concatenate(([0.], rec, [1.]))
                mpre = np.concatenate(([0.], prec, [0.]))
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                i = np.where(mrec[1:] != mrec[:-1])[0]
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
                return ap
            AP = compute_ap_simple(rec, prec)
        else:
            rec = np.array([0.0])
            prec = np.array([0.0])
            AP = 0.0
        Recall = rec[-1] if len(rec) > 0 else 0.0
        Precision = prec[-1] if len(prec) > 0 else 0.0
        F1 = 2 * Precision * Recall / (Precision + Recall + 1e-8)
        return AP, Precision, Recall, F1

    pred_all, gt_all = [], []
    for i in range(len(val_dataset)):
        img, boxes, labels = val_dataset[i]
        img = img.to(device)
        boxes = np.array(boxes)
        labels = np.array(labels)
        with torch.no_grad():
            pred_boxes, pred_scores, pred_labels = model.detect(img, score_thr=score_thr, iou_thr=iou_thr)
        pred_boxes = pred_boxes.cpu().numpy() if len(pred_boxes) > 0 else np.zeros((0,4))
        pred_scores = pred_scores.cpu().numpy() if len(pred_scores) > 0 else np.zeros(0)
        pred_labels = pred_labels.cpu().numpy() if len(pred_labels) > 0 else np.zeros(0, dtype=int)
        pred_all.append((pred_boxes, pred_labels, pred_scores))
        gt_all.append((boxes, labels))

    # mAP@[.5:.95]
    ap_all = []
    for thr in np.arange(0.5, 1.0, 0.05):
        ap, _, _, _ = calc_map_metrics(pred_all, gt_all, iou_thr=thr)
        ap_all.append(ap)
    mean_ap = np.mean(ap_all)
    ap50, precision, recall, f1 = calc_map_metrics(pred_all, gt_all, iou_thr=0.5)

    print(f"mAP50-95: {mean_ap:.4f}")
    print(f"mAP50: {ap50:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Возвращаем метрики
    return {
        'mean_ap': mean_ap,
        'ap50': ap50,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap_all': ap_all  # опционально
    }

if __name__ == '__main__':
    # Получаем метрики из функции
    metrics_dict = validate_fcos()
    
    # Собираем метрики для CSV
    metrics = {
        'model': 'fcos',
        'mAP50-95': round(metrics_dict['mean_ap'], 4),
        'mAP50': round(metrics_dict['ap50'], 4),
        'precision': round(metrics_dict['precision'], 4),
        'recall': round(metrics_dict['recall'], 4),
        'f1_score': round(metrics_dict['f1'], 4),
    }
    
    # Сохраняем в CSV
    df_fcos = pd.DataFrame([metrics])
    df_fcos.to_csv('fcos_metrics.csv', mode='a', index=False, header=True)
    print("Метрики сохранены в fcos_metrics.csv")
    print(df_fcos)
