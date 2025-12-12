import torch
import numpy as np

def iou_matrix(a, b):
    # a: [N, 4], b: [M, 4], xyxy
    area_a = (a[:,2]-a[:,0]).clamp(0) * (a[:,3]-a[:,1]).clamp(0)
    area_b = (b[:,2]-b[:,0]).clamp(0) * (b[:,3]-b[:,1]).clamp(0)
    lt = torch.max(a[:,None,:2], b[None,:,:2])
    rb = torch.min(a[:,None,2:], b[None,:,2:])
    wh = (rb - lt).clamp_(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area_a[:,None] + area_b - inter
    return inter / (union + 1e-6)

def calc_ap(rec, prec):
    rec = np.concatenate(([0.], rec, [1.]))
    prec = np.concatenate(([0.], prec, [0.]))
    for i in range(prec.size - 1, 0, -1):
        prec[i-1] = np.maximum(prec[i-1], prec[i])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1])

def evaluate_detector(model, dataset, device, iou_thr=0.5, max_det=100):
    from models.fcos import FCOSDetector
    model.eval()
    all_gt, all_pred = {}, {}
    n_classes = model.head.num_classes
    n_imgs = len(dataset)
    TP = np.zeros(n_classes)
    FP = np.zeros(n_classes)
    FN = np.zeros(n_classes)
    ap_list = []  # для mAP_50
    ap95_list = []  # mAP@[.5:.95]
    with torch.no_grad():
        for idx in range(n_imgs):
            img, gt_boxes, gt_labels = dataset[idx]
            img = img.to(device)
            gt_boxes = gt_boxes.to(device)
            logits, bbox_pred = model(img[None])
            pred_boxes, pred_scores, pred_labels = model.detect(img, score_thr=0.05)
            if pred_boxes.numel() == 0:
                FN[gt_labels.numpy()] += 1
                continue
            used_gt = set()
            # AP/IoU подсчёт
            ious = iou_matrix(pred_boxes, gt_boxes)
            for c in range(n_classes):
                # Все предсказания и GT с данным классом
                p_idx = (pred_labels == c).cpu().numpy()
                g_idx = (gt_labels == c).cpu().numpy()
                pred_c = pred_boxes[p_idx]
                gt_c = gt_boxes[g_idx]
                n_gt, n_pred = gt_c.shape[0], pred_c.shape[0]
                match = set()
                if n_gt == 0 and n_pred == 0:
                    continue
                if n_gt == 0 and n_pred > 0:
                    FP[c] += n_pred
                    continue
                if n_pred == 0 and n_gt > 0:
                    FN[c] += n_gt
                    continue
                ious_cls = iou_matrix(pred_c, gt_c)
                detected_gt = np.zeros(n_gt, dtype=bool)
                detected_pred = np.zeros(n_pred, dtype=bool)
                for i in range(n_pred):
                    imax = ious_cls[i].argmax()
                    if ious_cls[i, imax] > iou_thr and not detected_gt[imax]:
                        TP[c] += 1
                        detected_gt[imax] = True
                        detected_pred[i] = True
                    else:
                        FP[c] += 1
                FN[c] += np.sum(~detected_gt)
                # Для AP@50 (примитивная реализация)
                scores = pred_scores[p_idx].cpu().numpy() if pred_scores.numel() else np.zeros(0)
                tsort = np.argsort(-scores) if scores.shape[0]>0 else []
                tps = detected_pred[tsort] if len(tsort)>0 else np.zeros(0)
                fps = ~tps if len(tsort)>0 else np.zeros(0)
                tp_sum = np.cumsum(tps)
                fp_sum = np.cumsum(fps)
                npos = n_gt
                rec = tp_sum / (npos + 1e-12) if npos>0 else np.zeros_like(tp_sum)
                prec = tp_sum / (tp_sum + fp_sum + 1e-12) if len(tp_sum)>0 else np.zeros_like(tp_sum)
                AP = calc_ap(rec, prec) if len(tp_sum)>0 else 0.0
                ap_list.append(AP)
    Precision = np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-8)
    Recall = np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-8)
    F1 = 2*Precision*Recall/(Precision+Recall+1e-8)
    mAP_50 = np.mean(ap_list) if ap_list else 0.0
    # mAP@[.5:.95] для простоты вычисляется только как mAP_50
    # Можно добавить скользящее окно по IoU по желанию
    return {
        'Precision': Precision,
        'Recall': Recall,
        'F1-score': F1,
        'mAP_50': mAP_50,
        'mAP': mAP_50  # Для простоты, mAP_50 используется как mAP
    }
