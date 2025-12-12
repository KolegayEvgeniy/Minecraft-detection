# Конфиг для обучения FCOS на датасете Minecraft

CONFIG = {
    'data_dir': 'mmdetection/datasets/minecraft', # папка train/valid/test, формат VOC
    'output_dir': 'artifacts/fcos_scratch',
    'num_classes': 17,
    'batch_size': 4,
    'num_workers': 2,
    'epochs': 10,
    'img_size': 512,
    'lr': 0.0005,
    'device': 'cuda',
}

