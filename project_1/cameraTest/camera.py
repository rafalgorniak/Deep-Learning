import cv2
import torch
from skimage.feature import Cascade
from torchvision import transforms


def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(100, 100), max_size=(200, 200))
    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']
        boxes.append((x, y, w, h))
    return boxes


def draw(frame, boxes, probabilities=None):
    for idx, (x, y, w, h) in enumerate(boxes):
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
        if probabilities:
            prob = probabilities[idx]
            label = f"Male: {prob*100:.1f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def preprocess_face(frame, box, transform):
    x, y, w, h = box
    cropped_face = frame[y:y + h, x:x + w]
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    cropped_face = cv2.resize(cropped_face, (64, 64))
    cropped_face = transform(cropped_face)
    return cropped_face


def test_with_camera(model):
    file = "./cameraTest/face.xml"
    detector = Cascade(file)

    cap = cv2.VideoCapture(0)
    skip = 5
    i = 0
    boxes = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % skip == 0:
            boxes = detect(frame, detector)

        if boxes:

            faces = [preprocess_face(frame, box, transform) for box in boxes]
            faces_tensor = torch.stack(faces).to(model.device)
            probabilities = model.probability(faces_tensor).cpu().numpy().flatten()

        else:
            probabilities = []

        draw(frame, boxes, probabilities)
        cv2.imshow('Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    # cap.release()
    # cv2.destroyAllWindows()
