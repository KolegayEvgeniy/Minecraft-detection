import torch
import torchvision.transforms as T
from ultralytics import YOLO
from models.fcos import FCOSDetector
from config import CONFIG
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

# Классы Minecraft
MINECRAFT_CLASSES = [
    'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
    'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
]

# Создаём фиксированную палитру цветов
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(MINECRAFT_CLASSES), 3), dtype=np.uint8)

def draw_boxes_minecraft(image, boxes, labels, scores, threshold=0.5, show_text=True):
    """
    Отрисовка bounding boxes для Minecraft классов.
    
    Args:
        image: PIL Image или numpy array
        boxes: tensor или numpy array [N, 4] в формате [xmin, ymin, xmax, ymax]
        labels: tensor или numpy array [N] с индексами классов (0-16)
        scores: tensor или numpy array [N] с уверенностью
        threshold: порог для фильтрации по уверенности
        show_text: показывать ли текст с классом и уверенностью
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.copy())
    else:
        image_np = image.copy()
    
    # Конвертируем в numpy если нужно
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    
    # Обработка пустых детекций
    if len(boxes) == 0:
        return Image.fromarray(image_np) if isinstance(image, Image.Image) else image_np
    
    for i, box in enumerate(boxes):
        if len(scores) > i and scores[i] > threshold:
            class_id = int(labels[i])
            
            # Проверяем границы индекса
            if class_id < 0 or class_id >= len(MINECRAFT_CLASSES):
                continue
                
            label_text = MINECRAFT_CLASSES[class_id]
            color = [int(c) for c in COLORS[class_id]]
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Ограничиваем координаты размерами изображения
            h, w = image_np.shape[:2]
            x_min = max(0, min(x_min, w))
            y_min = max(0, min(y_min, h))
            x_max = max(0, min(x_max, w))
            y_max = max(0, min(y_max, h))
            
            # Адаптивный размер шрифта
            box_width = x_max - x_min
            font_scale = max(0.4, min(1.0, box_width / 200.0))
            thickness = max(1, int(font_scale * 2))
            
            # Полный текст подписи
            display_text = f"{label_text}: {scores[i]:.2f}"
            
            # Расчёт размера текстового блока
            (text_width, text_height), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Позиция текста
            text_x = x_min
            text_y = y_min - 5
            
            # Сдвигаем текст, если он выходит за пределы изображения
            if text_y < text_height:
                text_y = y_min + text_height + 5
            
            # Отрисовка фона для текста
            if show_text:
                cv2.rectangle(
                    image_np, 
                    (text_x, text_y - text_height - baseline), 
                    (text_x + text_width, text_y + baseline), 
                    color, 
                    cv2.FILLED
                )
                cv2.putText(
                    image_np, 
                    display_text, 
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 255, 255),
                    thickness
                )
            
            # Отрисовка bounding box
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, 2)
            
    return Image.fromarray(image_np) if isinstance(image, Image.Image) else image_np

def load_fcos_model(ckpt_path, device, num_classes=17, img_size=512):
    """Загрузка FCOS модели из чекпоинта"""
    model = FCOSDetector(num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def fcos_inference(model, img_pil, device, img_size=512, score_thr=0.2, iou_thr=0.5):
    """
    Инференс для FCOS модели.
    
    Args:
        model: FCOSDetector модель
        img_pil: PIL Image
        device: torch device
        img_size: размер изображения для ресайза
        score_thr: порог уверенности
        iou_thr: порог IoU для NMS
    
    Returns:
        boxes, scores, labels в формате numpy arrays
    """
    # Трансформация изображения
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    
    img_tensor = transform(img_pil).to(device)
    
    # Инференс
    with torch.no_grad():
        pred_boxes, pred_scores, pred_labels = model.detect(
            img_tensor, 
            score_thr=score_thr, 
            iou_thr=iou_thr, 
            stride=64
        )
    
    # Масштабируем координаты обратно к исходному размеру
    orig_w, orig_h = img_pil.size
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size
    
    if len(pred_boxes) > 0:
        pred_boxes = pred_boxes.cpu().numpy()
        pred_boxes[:, [0, 2]] *= scale_x
        pred_boxes[:, [1, 3]] *= scale_y
        pred_scores = pred_scores.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
    else:
        pred_boxes = np.zeros((0, 4))
        pred_scores = np.zeros(0)
        pred_labels = np.zeros(0, dtype=int)
    
    return pred_boxes, pred_scores, pred_labels

def yolo_inference(model, img_pil, conf_threshold=0.25):
    """
    Инференс для YOLO модели.
    
    Args:
        model: YOLO модель
        img_pil: PIL Image
        conf_threshold: порог уверенности
    
    Returns:
        boxes, scores, labels в формате numpy arrays
    """
    results = model(img_pil, conf=conf_threshold, verbose=False)
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy().astype(int)
    else:
        boxes = np.zeros((0, 4))
        scores = np.zeros(0)
        labels = np.zeros(0, dtype=int)
    
    return boxes, scores, labels

def run_and_visualize_minecraft(image_path, yolo_model, fcos_model, device, 
                                 threshold=0.25, img_size=512, show_text=True):
    """
    Запуск инференса на изображении и визуализация результатов для обеих моделей.
    
    Args:
        image_path: путь к изображению
        yolo_model: YOLO модель
        fcos_model: FCOS модель
        device: torch device
        threshold: порог уверенности
        img_size: размер изображения для FCOS
        show_text: показывать ли текст
    
    Returns:
        PIL Images с отрисованными результатами
    """
    # Загрузка изображения
    img_pil = Image.open(image_path).convert('RGB')
    
    # Инференс YOLO
    yolo_boxes, yolo_scores, yolo_labels = yolo_inference(yolo_model, img_pil, threshold)
    
    # Инференс FCOS
    fcos_boxes, fcos_scores, fcos_labels = fcos_inference(
        fcos_model, img_pil, device, img_size, score_thr=threshold
    )
    
    # Отрисовка результатов
    yolo_img_drawn = draw_boxes_minecraft(
        img_pil.copy(), yolo_boxes, yolo_labels, yolo_scores, 
        threshold=threshold, show_text=show_text
    )
    fcos_img_drawn = draw_boxes_minecraft(
        img_pil.copy(), fcos_boxes, fcos_labels, fcos_scores, 
        threshold=threshold, show_text=show_text
    )
    
    return yolo_img_drawn, fcos_img_drawn, img_pil

def measure_fps_minecraft(models_dict, image_paths, num_runs=100, device='cpu', img_size=512):
    """
    Универсальная функция для измерения FPS для обеих моделей.
    
    Args:
        models_dict: dict с ключами 'yolo' и 'fcos' и соответствующими моделями
        image_paths: список путей к изображениям для тестирования
        num_runs: количество запусков для усреднения
        device: torch device
        img_size: размер изображения для FCOS
    
    Returns:
        dict с результатами FPS
    """
    results = {}
    
    # Загружаем изображения
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    # Подготовка трансформаций для FCOS
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    
    # Разогрев
    print("Прогрев моделей...")
    if 'yolo' in models_dict:
        for img in images:
            models_dict['yolo'].predict(img, verbose=False)
    
    if 'fcos' in models_dict:
        for img in images:
            img_tensor = transform(img).to(device)
            with torch.no_grad():
                models_dict['fcos'].detect(img_tensor, score_thr=0.2, iou_thr=0.5, stride=64)
    
    # Измерение FPS для YOLO (включает весь pipeline: preprocessing + inference + postprocessing)
    if 'yolo' in models_dict:
        print("Измерение FPS для YOLOv8...")
        # Используем тот же размер, что и FCOS для справедливого сравнения
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.perf_counter()
        for _ in range(num_runs):
            for img in images:
                models_dict['yolo'].predict(img, verbose=False, imgsz=img_size)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()
        total_time = end_time - start_time
        fps = (num_runs * len(images)) / total_time
        results['YOLOv8s'] = fps
    
    # Измерение FPS для FCOS (включает preprocessing + inference + postprocessing)
    if 'fcos' in models_dict:
        print("Измерение FPS для FCOS...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.perf_counter()
        for _ in range(num_runs):
            for img in images:
                img_tensor = transform(img).to(device)
                with torch.no_grad():
                    models_dict['fcos'].detect(img_tensor, score_thr=0.2, iou_thr=0.5, stride=64)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()
        total_time = end_time - start_time
        fps = (num_runs * len(images)) / total_time
        results['FCOS'] = fps
    
    # Вывод результатов
    print("\n" + "="*50 + " Результаты измерения FPS " + "="*50)
    for name, fps in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name:<15}: {fps:.2f} FPS")
    
    return results

def main():
    """Основная функция для тестирования"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Пути к моделям
    yolo_path = 'artifacts/yolo_training/yolov8s_minecraft_v1/weights/best.pt'
    fcos_path = 'artifacts/fcos_scratch/fcos_minecraft.pt'
    
    # Проверка существования моделей
    if not os.path.exists(yolo_path):
        print(f"Ошибка: модель YOLO не найдена по пути {yolo_path}")
        return
    if not os.path.exists(fcos_path):
        print(f"Ошибка: модель FCOS не найдена по пути {fcos_path}")
        return
    
    # Загрузка моделей
    print("Загрузка моделей...")
    yolo_model = YOLO(yolo_path).to(device)
    fcos_model = load_fcos_model(fcos_path, device, num_classes=17, img_size=CONFIG['img_size'])
    print("Модели загружены!")
    
    # Выбор 3 тестовых изображений
    test_dir = Path('mmdetection/datasets/minecraft/test')
    test_images = sorted(list(test_dir.glob('*.jpg')))[:3]
    
    if len(test_images) < 3:
        print(f"Ошибка: найдено только {len(test_images)} изображений в test директории")
        return
    
    print(f"\nИспользуемые тестовые изображения:")
    for img_path in test_images:
        print(f"  - {img_path.name}")
    
    # Визуализация результатов для каждого изображения
    print("\n" + "="*50)
    print("Визуализация результатов детекции")
    print("="*50)
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    
    for idx, img_path in enumerate(test_images):
        yolo_img, fcos_img, orig_img = run_and_visualize_minecraft(
            str(img_path), yolo_model, fcos_model, device, 
            threshold=0.25, img_size=CONFIG['img_size'], show_text=True
        )
        
        # Отображение оригинального изображения
        axs[idx, 0].imshow(orig_img)
        axs[idx, 0].set_title(f'Оригинал: {img_path.name}', fontsize=10)
        axs[idx, 0].axis('off')
        
        # YOLO результаты
        axs[idx, 1].imshow(yolo_img)
        axs[idx, 1].set_title('YOLOv8s', fontsize=12, fontweight='bold')
        axs[idx, 1].axis('off')
        
        # FCOS результаты
        axs[idx, 2].imshow(fcos_img)
        axs[idx, 2].set_title('FCOS', fontsize=12, fontweight='bold')
        axs[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Измерение FPS
    print("\n" + "="*50)
    print("Измерение FPS")
    print("="*50)
    
    models_dict = {
        'yolo': yolo_model,
        'fcos': fcos_model
    }
    
    fps_results = measure_fps_minecraft(
        models_dict, 
        [str(img) for img in test_images], 
        num_runs=100, 
        device=device,
        img_size=CONFIG['img_size']
    )
    
    return fps_results

if __name__ == '__main__':
    main()
