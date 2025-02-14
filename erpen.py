import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Modeli yükleyin
model = YOLO('bestv2.pt')

# Resimlerin bulunduğu klasör yolu
image_folder = 'hatalieb'

# Klasördeki tüm resim dosyalarını al
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Tüm resimleri döngüye sok
for image_file in image_files:
    # Resim dosyasının tam yolunu alın
    image_path = os.path.join(image_folder, image_file)
    
    # Resmi yükleyin
    image = cv2.imread(image_path)
    
    # Modeli kullanarak resmi işleyin
    results = model(image)
    
    # Sonuçları alın
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            
            # Dikdörtgen çiz
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Görüntüyü matplotlib ile gösterin
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(image_file)
    plt.show()
