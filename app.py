import pickle
import numpy as np
import cv2
import gradio as gr
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

MODELO_PATH = "modelo_senas.pkl"
MODELO_MANO_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Cargar modelos una sola vez
with open(MODELO_PATH, "rb") as f:
    modelo = pickle.load(f)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELO_MANO_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
landmarker = HandLandmarker.create_from_options(options)


def normalizar_landmarks(hand_landmarks):
    """Normaliza landmarks relativo a la muñeca y escala por tamaño de mano."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = coords[0].copy()
    coords = coords - wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords.flatten()


def dibujar_landmarks(img, hand_landmarks):
    h, w, _ = img.shape
    puntos = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        puntos.append((cx, cy))
        cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
    for i, j in HAND_CONNECTIONS:
        if i < len(puntos) and j < len(puntos):
            cv2.line(img, puntos[i], puntos[j], (255, 255, 255), 2)
    return img


def predecir(imagen):
    if imagen is None:
        return imagen

    img = imagen.copy()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen)
    resultado = landmarker.detect(mp_image)

    if not resultado.hand_landmarks:
        cv2.putText(img, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    for hand_landmarks in resultado.hand_landmarks:
        img = dibujar_landmarks(img, hand_landmarks)

        datos = normalizar_landmarks(hand_landmarks).reshape(1, -1)
        letra = modelo.predict(datos)[0]
        probabilidades = modelo.predict_proba(datos)[0]
        confianza = max(probabilidades)

        h, w, _ = img.shape
        texto = f"{letra.upper()} ({confianza:.0%})"
        cv2.putText(img, texto, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 6)
        cv2.putText(img, texto, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

    return img


# --- Gradio Interface ---
with gr.Blocks(title="ASL Sign Language Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ASL Sign Language Recognition")
    gr.Markdown("Show a sign from the **ASL alphabet** in front of the camera.")

    entrada = gr.Image(sources=["webcam"], type="numpy", show_label=False,
                       streaming=True)
    entrada.stream(fn=predecir, inputs=entrada, outputs=entrada)

demo.launch()
