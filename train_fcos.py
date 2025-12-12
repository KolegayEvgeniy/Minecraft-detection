"""
Скрипт для обучения модели FCOS на датасете Minecraft

Использование:
    python train_fcos.py
    
    или с дополнительными параметрами:
    python train_fcos.py --work-dir artifacts/fcos_training --amp
"""
import os
import sys
import argparse

# Добавляем mmdetection в путь
sys.path.insert(0, 'mmdetection')

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo

def parse_args():
    parser = argparse.ArgumentParser(description='Train FCOS detector on Minecraft dataset')
    parser.add_argument(
        '--config',
        default='mmdetection/configs/fcos/fcos_minecraft.py',
        help='train config file path')
    parser.add_argument(
        '--work-dir',
        default='artifacts/fcos_training',
        help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Загружаем конфигурацию
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    
    # Переопределяем work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
        os.makedirs(args.work_dir, exist_ok=True)
    
    # FP16 - уже настроено в конфиге, но можно включить через аргументы
    if args.amp and not hasattr(cfg, 'fp16'):
        cfg.fp16 = dict(loss_scale='dynamic')
    
    # Resume обучение
    if args.resume == 'auto':
        cfg.resume = True
    elif args.resume is not None:
        cfg.resume = args.resume
    
    # Переопределяем параметры через cfg-options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # Убираем ограничение кэша
    setup_cache_size_limit_of_dynamo()
    
    # Выводим информацию о конфигурации
    print("=" * 60)
    print("Конфигурация обучения FCOS")
    print("=" * 60)
    print(f"Конфигурация: {args.config}")
    print(f"Рабочая директория: {cfg.work_dir}")
    print(f"Максимум эпох: {cfg.train_cfg.max_epochs}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"Workers: {cfg.train_dataloader.num_workers}")
    print(f"Размер изображения: {cfg.train_dataloader.dataset.pipeline[2].scale}")
    print(f"FP16: {hasattr(cfg, 'fp16')}")
    print(f"Resume: {cfg.get('resume', False)}")
    print("=" * 60)
    
    # Создаем Runner и запускаем обучение
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    
    print("\nНачинаем обучение...\n")
    runner.train()

if __name__ == '__main__':
    main()

