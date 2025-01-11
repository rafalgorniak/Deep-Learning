import cv2
import os

# Ścieżki do obrazów i adnotacji
images_dir = "../data/widerFace/images"
annotations_file = "../data/widerFace/wider_face_split/wider_face_val_bbx_gt.txt"
output_dir = "../data/widerFace/faces"
labels_file = "../data/widerFace/faces_gender_labels.txt"
os.makedirs(output_dir, exist_ok=True)

# Wczytaj adnotacje
with open(annotations_file, 'r') as f:
    lines = f.readlines()

image = None
current_image_path = None
face_count = 0

with open(labels_file, 'w') as label_file:
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(".jpg"):
            current_image_path = os.path.join(images_dir, line)
            image = cv2.imread(current_image_path)
            if image is None:
                print(f"Nie udało się wczytać obrazu: {current_image_path}")
                i += 1
                continue

            i += 1
            num_faces = int(lines[i].strip())
            i += 1

            for _ in range(num_faces):
                if i >= len(lines):
                    break
                face_line = lines[i].strip()
                i += 1

                try:
                    x, y, w, h = map(int, face_line.split()[:4])
                    face = image[y:y + h, x:x + w]

                    face_resized = cv2.resize(face, (64, 64))

                    face_count += 1
                    output_path = os.path.join(output_dir, f"{face_count}.jpg")
                    cv2.imwrite(output_path, face_resized)

                    label = 1
                    label_file.write(f"{os.path.basename(output_path)} {label}\n")

                except Exception as e:
                    print(f"Błąd podczas przetwarzania twarzy: {face_line} w obrazie {current_image_path}, błąd: {e}")
        else:
            i += 1

print("Przetwarzanie zakończone.")

