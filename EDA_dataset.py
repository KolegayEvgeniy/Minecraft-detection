import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
import numpy as np
from pathlib import Path

class MinecraftDatasetAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.folder_dir = self.data_dir / 'train'
        
        
        self.classes = [
            'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
            'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
        ]
        

    def check_dataset_folders(self):
        """
        Проверяем на соответствие полноты разметки датасета

        Arg: 
            base_path: путь к папке с датасетом где расположены папки train/val/test
            folder: данные которые хотим проверить

        Return:
            Информация о найденных проблемах и их количестве

        """

        print("=" * 60)
        print("1. ПРОВЕРКА ПОЛНОТЫ РАЗМЕТКИ ДАТАСЕТА")
        print("=" * 60)
    
        folders = ['train', 'valid', 'test']

        for folder in folders:
            # Проверяем наличие папки
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.exists(folder_path):
                print(f"⚠️  Папка '{folder}' не найдена по пути: {folder_path}")

            # Получаем все файлы в папке
            all_files = os.listdir(folder_path)

            # Разделяем на изображения и XML
            img_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
            xml_files = [f for f in all_files if f.lower().endswith('.xml')]

            # Получаем имена без расширений
            if img_files and xml_files:
                img_names = {os.path.splitext(f)[0] for f in img_files}
                xml_names = {os.path.splitext(f)[0] for f in xml_files}

            # Находим имена файлов, которые есть в одном множестве, но отсутствуют в другом
            img_without_xml = img_names - xml_names
            xml_without_img = xml_names - img_names

            if not img_without_xml and not xml_without_img:
                print(f'\nВ папке: {folder}')
                print('Все изображения имеют разметку и отсутствует разметка без изображений.')
                print(f'Всего изображений: {len(img_files)}')
                print(f'Всего XML: {len(xml_files)}')
                print(f'ВЫВОД: C {folder} все норм! Менять ничего не нужно!')
            else:
                if img_without_xml:
                    all_img_without_xml = []
                    for img in img_without_xml:
                        all_img_without_xml.append(img)
                    print(f'\nВ папке: {folder}')
                    print(f'Без разметки: {len(all_img_without_xml)}')
                    print(f'Изображения: {all_img_without_xml}')
                    print(f'Всего изображений: {len(img_files)}')
                    print(f'Всего XML: {len(xml_files)}')
                    print(f'ВЫВОД: Необходимо проверить {all_img_without_xml}')

                if xml_without_img:
                    all_xml_without_img = []
                    for img in img_without_xml:
                        all_xml_without_img.append(img)
                    print(f'\nВ папке: {folder}')
                    print(f'Без изображений: {len(all_xml_without_img)}')
                    print(f'XML: {all_xml_without_img}')
                    print(f'Всего изображений: {len(img_files)}')
                    print(f'Всего XML: {len(xml_files)}')
                    print(f'ВЫВОД: Необходимо проверить {all_img_without_xml}')


    def parse_annotation(self, xml_file):
        """Парсинг XML аннотации"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotation_data = {
            'filename': root.find('filename').text,
            'size': {
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'depth': int(root.find('size/depth').text)
            },
            'objects': []
        }
        
        for obj in root.findall('object'):
            obj_data = {
                'name': obj.find('name').text,
                'bndbox': {
                    'xmin': int(obj.find('bndbox/xmin').text),
                    'ymin': int(obj.find('bndbox/ymin').text),
                    'xmax': int(obj.find('bndbox/xmax').text),
                    'ymax': int(obj.find('bndbox/ymax').text)
                }
            }
            annotation_data['objects'].append(obj_data)
        
        return annotation_data
    
    def analyze_dataset(self):
        """Анализ всего датасета"""
        xml_files = list(self.folder_dir.glob('*.xml'))
        
        print(f"Всего XML файлов: {len(xml_files)}")
        
        # Проверка соответствия изображений и аннотаций
        image_files = list(self.folder_dir.glob('*.jpg'))
        print(f"Всего изображений: {len(image_files)}")
        
        # Анализ распределения классов
        class_counter = Counter()
        bbox_sizes = []
        
        for xml_file in xml_files:
            annotation = self.parse_annotation(xml_file)
            for obj in annotation['objects']:
                class_counter[obj['name']] += 1
                
                # Расчет размера bbox
                bbox = obj['bndbox']
                width = bbox['xmax'] - bbox['xmin']
                height = bbox['ymax'] - bbox['ymin']
                area = width * height
                bbox_sizes.append(area)
        
        return class_counter, bbox_sizes
    
    def visualize_class_distribution(self, class_counter):
        """Визуализация распределения классов"""

        print("=" * 60)
        print("2. АНАЛИЗ РСПРЕДЕЛЕНИЯ КЛАССОВ")
        print("=" * 60)

        plt.figure(figsize=(12, 6))
        sorted_items = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
        classes = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        plt.bar(classes, counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Классы')
        plt.ylabel('Количество объектов')
        plt.title('Распределение классов в датасете')
        plt.tight_layout()
        plt.show()
        
        # Анализ дисбаланса
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Наибольшее количество: {max_count}")
        print(f"Наименьшее количество: {min_count}")
        print(f"Коэффициент дисбаланса: {imbalance_ratio:.2f}")
        
    def visualize_sample(self, image_name):
        """Визуализация примера с bounding boxes"""

        print("=" * 60)
        print("3. ВИЗУАЛИЗАЦИЯ ПРИМЕРА С bounding boxes")
        print("=" * 60)

        xml_file = self.folder_dir / f"{Path(image_name).stem}.xml"
        image_file = self.folder_dir / image_name
        
        if not xml_file.exists() or not image_file.exists():
            print(f"Изображение не найдено: {image_name}")
            return
        
        annotation = self.parse_annotation(xml_file)
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        
        for obj in annotation['objects']:
            bbox = obj['bndbox']
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

            # Рисуем прямоугольник на изображении
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 5)

            # ДобаДобавляем текст
            text = f"{obj['name']}"
            cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (10, 0, 200), 4)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Пример изображения: {image_name}")
        plt.tight_layout()
        plt.show()