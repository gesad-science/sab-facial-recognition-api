import os
import cv2
import math
import torch
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
import sklearn
import joblib 
import numpy as np

MODEL_PATH = 'src/models/face_landmarker.task'

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

MODELS = {}

def load_models():
    try:
        print("Loading models...")

        MODELS["classifier"] = joblib.load('src/models/classifier.joblib')
        MODELS["le"] = joblib.load('src/models/label_encoder.joblib')

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

        MODELS["DEVICE"] = DEVICE
        MODELS["resnet"] = resnet

        print("Models loaded!")
    except Exception as e:
        print(f"Error on loading models: {e}")

def get_landmarks_and_image(image, draw=True):
    if image is None:
        return None, None, None
    
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    detection_result = detector.detect(mp_image)

    if len(detection_result.face_landmarks) == 0:
        return image, None, None

    annotated = None
    if draw:
        annotated = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
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

    return image, detection_result.face_landmarks[0], annotated
    
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
    face_tensor = face_tensor.unsqueeze(0).to(MODELS["DEVICE"])

    with torch.no_grad():
        embedding = MODELS["resnet"](face_tensor).detach().cpu()

    return embedding.flatten().tolist()

def classify_face(image):
    original_img, landmarks, annotated_img = get_landmarks_and_image(image)

    if landmarks:
        aligned_img = align_face(original_img, landmarks)

        if aligned_img is not None:
            emb = get_embedding(aligned_img)

            prediction_idx = MODELS["classifier"].predict([emb])[0]
            prediction_name = MODELS["le"].inverse_transform([prediction_idx])[0]

            distances, _ = MODELS["classifier"].kneighbors([emb])
            dist = distances[0][0]
    else:
        raise print("No face detected or error on processing.")
    
    return annotated_img, prediction_name, dist