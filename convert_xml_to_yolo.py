import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
import shutil

CLASSES = ['bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 
           'ghast', 'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 
           'turtle', 'wolf', 'zombie']

def convert_xml_to_yolo(xml_path, img_dir, label_dir, img_output_dir):
    """Конвертирует XML разметку в YOLO формат"""
    try:
        # Парсим XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Получаем имя файла из XML или из имени XML файла
        xml_basename = os.path.basename(xml_path)
        xml_name_without_ext = os.path.splitext(xml_basename)[0]
        
        filename_elem = root.find('filename')
        if filename_elem is not None and filename_elem.text:
            filename = filename_elem.text
        else:
            filename = xml_name_without_ext + '.jpg'
        
        # Находим изображение в исходной директории
        img_path = None
        # Сначала пробуем имя из XML
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(img_dir, os.path.splitext(filename)[0] + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        # Если не нашли, пробуем имя XML файла
        if not img_path:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_path = os.path.join(img_dir, xml_name_without_ext + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        if not img_path or not os.path.exists(img_path):
            print(f"Изображение не найдено для {xml_path}")
            return None
        
        # Читаем размеры изображения
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось прочитать изображение: {img_path}")
            return None
            
        img_height, img_width = img.shape[:2]
        
        # Создаем текстовый файл для YOLO
        txt_filename = xml_name_without_ext + '.txt'
        txt_path = os.path.join(label_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in CLASSES:
                    continue
                    
                cls_id = CLASSES.index(cls_name)
                
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Конвертация в YOLO формат (нормализованные координаты)
                x_center = (xmin + xmax) / (2 * img_width)
                y_center = (ymin + ymax) / (2 * img_height)
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Проверка корректности координат
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Копируем изображение в новую структуру
        os.makedirs(img_output_dir, exist_ok=True)
        img_output_path = os.path.join(img_output_dir, os.path.basename(img_path))
        shutil.copy(img_path, img_output_path)
        
        return img_path
        
    except Exception as e:
        print(f"Ошибка при обработке {xml_path}: {e}")
        return None

def process_split(split):
    """Обрабатывает все XML файлы для указанного сплита"""
    base_dir = f'mmdetection/datasets/minecraft/{split}'
    label_dir = f'mmdetection/datasets/minecraft_yolo/{split}/labels'
    img_dir = f'mmdetection/datasets/minecraft/{split}'
    img_output_dir = f'mmdetection/datasets/minecraft_yolo/{split}/images'
    
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Получаем список XML файлов
    xml_files = [f for f in os.listdir(base_dir) if f.endswith('.xml')]
    
    print(f"Обработка {split}: {len(xml_files)} файлов")
    
    for xml_file in tqdm(xml_files, desc=f"Processing {split}"):
        xml_path = os.path.join(base_dir, xml_file)
        convert_xml_to_yolo(xml_path, img_dir, label_dir, img_output_dir)
    
    print(f"Готово: {split}")

def main():
    """Основная функция"""
    print("Конвертация данных из XML в YOLO формат...")
    print("Исходный датасет: mmdetection/datasets/minecraft")
    print("Выходной датасет: mmdetection/datasets/minecraft_yolo")
    
    # Создаем директории
    os.makedirs('mmdetection/datasets/minecraft_yolo', exist_ok=True)
    
    # Обрабатываем все сплиты
    for split in ['train', 'valid', 'test']:
        process_split(split)
    
    # Создаем data.yaml файл
    yaml_content = f"""# Minecraft Dataset for YOLO
path: mmdetection/datasets/minecraft_yolo
train: train/images
val: valid/images
test: test/images

# Классы (17 классов)
names:
{chr(10).join([f'  {i}: {cls}' for i, cls in enumerate(CLASSES)])}

# Количество классов
nc: {len(CLASSES)}
"""
    
    with open('mmdetection/datasets/minecraft_yolo/data_voc.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print("Конвертация завершена!")
    print("Файл конфигурации создан: mmdetection/datasets/minecraft_yolo/data_voc.yaml")
    print(f"Всего классов: {len(CLASSES)}")

if __name__ == '__main__':
    main()