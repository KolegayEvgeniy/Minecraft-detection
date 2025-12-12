# train_yolo.py
import os
import sys
import torch

# КРИТИЧЕСКИ ВАЖНО для Jupyter
os.environ['PYTHONUNBUFFERED'] = '1'  # Отключаем буферизацию вывода
os.environ['TQDM_DISABLE'] = '0'      # Включаем tqdm

# Перенаправляем stderr в stdout (для прогресс-баров)
sys.stderr = sys.stdout

# Функция для принудительного вывода
def log(message):
    print(message, flush=True)

log("🚀 Начинаем обучение YOLO...")

from ultralytics import YOLO

# Проверяем доступность CUDA
if torch.cuda.is_available():
    device = 0
    log(f"✅ CUDA доступна, используется GPU")
else:
    device = 'cpu'
    log(f"⚠️  CUDA недоступна, используется CPU")

# Создаем директории для артефактов
os.makedirs('artifacts/yolo_training', exist_ok=True)

# Загружаем предобученную модель YOLOv8s
log("Загрузка модели YOLOv8s...")
model = YOLO('yolov8s.pt')

# Настройка параметров обучения - ВАЖНО: verbose должен быть True!
train_args = {
    'data': 'mmdetection/datasets/minecraft_yolo/data_voc.yaml',
    'epochs': 12,
    'imgsz': 512,
    'batch': 2,
    'workers': 2,
    'amp': True if torch.cuda.is_available() else False,
    'save_period': 1,
    'project': 'artifacts/yolo_training',
    'name': 'yolov8s_minecraft_v1',
    'exist_ok': True,
    'patience': 5,
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'verbose': True,  # ⚠️ ВАЖНО: должен быть True для отображения прогресса!
    'plots': True,    # Включаем графики
    'save': True,     # Сохраняем результаты
}

if device == 'cuda':
    train_args['device'] = 0
else:
    train_args['device'] = 'cpu'

log(f"\n📋 Параметры обучения:")
for key, value in train_args.items():
    if key != 'data':
        log(f"  {key}: {value}")

log("\n" + "="*50)
log("🎯 Запуск обучения...")
log("="*50)

# Дообучаем на наших данных
log("⏳ Начало процесса обучения...")
results = model.train(**train_args)

log("\n" + "="*50)
log("✅ Обучение YOLO завершено!")
log("="*50)

# Валидация после обучения
log("\n📊 Запуск валидации...")
metrics = model.val()
log(f"📈 mAP50-95: {metrics.box.map:.4f}")
log(f"📈 mAP50: {metrics.box.map50:.4f}")