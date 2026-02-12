import os
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
import numpy as np

MODEL_PATH = 'app/models/face_landmarker.task'
INPUT_IMAGE = 'app/input_images/eu.jpg'
BASE_OUTPUT_DIR = 'app/image_results'

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)               #heavy (create one time)

def create_dirs():
    subfolders = ['annotated', 'aligned']
    for folder in subfolders:
        path = os.path.join(BASE_OUTPUT_DIR, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"{path} folder created")

def get_landmarks_and_image(image_path, draw=True):
    try:
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            return None, None, None

        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        detection_result = detector.detect(mp_image)

        if len(detection_result.face_landmarks) == 0:
            return cv_img, None, None

        annotated = None
        if draw:
            annotated = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)        #sugestion
            #annotated = np.copy(rgb_img)                               #original version
            for face_landmarks in detection_result.face_landmarks:
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
                )
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
                )
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        return cv_img, detection_result.face_landmarks[0], annotated
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None

def align_face(image, landmarks, target_size=(160, 160)):
    h, w, _ = image.shape

    left_iris = landmarks[LEFT_IRIS_CENTER]
    right_iris = landmarks[RIGHT_IRIS_CENTER]

    lx, ly = int(left_iris.x * w), int(left_iris.y * h)
    rx, ry = int(right_iris.x * w), int(right_iris.y * h)

    dy = ry - ly
    dx = rx - lx
    angle = math.degrees(math.atan2(dy, dx))


    center_x = (lx + rx) // 2
    center_y = (ly + ry) // 2

    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))


    face_width = int(math.sqrt(dx**2 + dy**2) * 4.0)

    start_x = max(0, center_x - face_width // 2)
    start_y = max(0, center_y - face_width // 2)
    end_x = min(w, center_x + face_width // 2)
    end_y = min(h, center_y + face_width // 2)

    crop = rotated_img[start_y:end_y, start_x:end_x]

    # resize 160x160
    if crop.size == 0: return None
    aligned_face = cv2.resize(crop, target_size)

    return aligned_face

#starting whats missing
"""
def get_embedding(aligned_face):
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    aligned_face = np.ascontiguousarray(aligned_face, dtype=np.uint8)

    h, w, c = aligned_face.shape
    if c != 3:
        raise ValueError(f"Esperado 3 canais, obtido: {c}")

    face_tensor = torch.from_numpy(aligned_face)
    face_tensor = face_tensor.permute(2, 0, 1).contiguous()
    face_tensor = face_tensor.float().div(255.0)

    face_tensor = (face_tensor - 0.5) / 0.5
    face_tensor = face_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = resnet(face_tensor).detach().cpu()

    return embedding.flatten().tolist()

def recognize_face(image_path):
    print(f"Recognizing: {image_path}")
    img_bgr, landmarks, _ = get_landmarks_and_image(image_path)

    if landmarks is None:
        return "No face detected"

    aligned = align_face(img_bgr, landmarks)
    if aligned is None:
        return "Could not align face"

    emb = get_embedding(aligned)

    prediction_idx = classifier.predict([emb])[0]
    prediction_name = le.inverse_transform([prediction_idx])[0]

    distances, _ = classifier.kneighbors([emb])
    dist = distances[0][0]

    return f"Prediction: {prediction_name} (Distance: {dist:.4f})"
"""
#ending whats missing

def main(image_path):
    create_dirs()

    file_name = os.path.basename(image_path)

    original_img, landmarks, annotated_img = get_landmarks_and_image(image_path)

    if landmarks:
        path_annotated = os.path.join(BASE_OUTPUT_DIR, 'annotated', f'annotated_{file_name}')
        cv2.imwrite(path_annotated, annotated_img)
        print(f"Saved: {path_annotated}")

        aligned_img = align_face(original_img, landmarks)

        if align_face is not None:
            path_aligned = os.path.join(BASE_OUTPUT_DIR, 'aligned', f'aligned_{file_name}')
            cv2.imwrite(path_aligned, aligned_img)
            print(f"Saved: {path_aligned}")
    else:
        print("No face detected or error on processing.")

if __name__ == "__main__":
    main(INPUT_IMAGE)