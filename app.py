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
        return None, "No se recibio imagen"

    img = imagen.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen)
    resultado = landmarker.detect(mp_image)

    if not resultado.hand_landmarks:
        return imagen, "No se detecto una mano. Intenta de nuevo."

    for hand_landmarks in resultado.hand_landmarks:
        img = dibujar_landmarks(img, hand_landmarks)

        fila = []
        for lm in hand_landmarks:
            fila.extend([lm.x, lm.y, lm.z])
        datos = np.array(fila).reshape(1, -1)

        letra = modelo.predict(datos)[0]
        probabilidades = modelo.predict_proba(datos)[0]
        confianza = max(probabilidades)

    resultado_texto = f"## Letra: {letra.upper()}\nConfianza: {confianza:.0%}"
    return img, resultado_texto


# --- Interfaz Gradio ---
with gr.Blocks(title="Reconocimiento de Lenguaje de Senas", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤟 Reconocimiento de Lenguaje de Senas")
    gr.Markdown("Toma una foto haciendo una sena del **alfabeto ASL** y el sistema la identificara.")

    with gr.Row():
        with gr.Column(scale=2):
            entrada = gr.Image(sources=["webcam"], type="numpy", label="Camara")
            boton = gr.Button("Identificar Sena", variant="primary", size="lg")

        with gr.Column(scale=1):
            imagen_resultado = gr.Image(label="Resultado", type="numpy")
            texto_resultado = gr.Markdown("Toma una foto y haz clic en **Identificar Sena**")

    boton.click(fn=predecir, inputs=entrada, outputs=[imagen_resultado, texto_resultado])

    gr.Markdown("### Referencia - Alfabeto ASL")
    gr.Image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/American_Sign_Language_ASL.svg/800px-American_Sign_Language_ASL.svg.png",
        label="Alfabeto ASL",
        show_download_button=False,
    )

demo.launch()
