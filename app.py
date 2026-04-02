import pickle
import numpy as np
import cv2
import streamlit as st
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


@st.cache_resource
def cargar_modelo():
    with open(MODELO_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def crear_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_MANO_PATH),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


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


def procesar_imagen(img_array):
    landmarker = crear_landmarker()
    modelo = cargar_modelo()

    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = landmarker.detect(mp_image)

    letra = None
    confianza = 0.0
    img_resultado = img_array.copy()

    if resultado.hand_landmarks:
        for hand_landmarks in resultado.hand_landmarks:
            img_resultado = dibujar_landmarks(img_resultado, hand_landmarks)

            fila = []
            for lm in hand_landmarks:
                fila.extend([lm.x, lm.y, lm.z])
            datos = np.array(fila).reshape(1, -1)

            letra = modelo.predict(datos)[0]
            probabilidades = modelo.predict_proba(datos)[0]
            confianza = max(probabilidades)

    return img_resultado, letra, confianza


# --- UI ---
st.set_page_config(page_title="Reconocimiento de Lenguaje de Senas", layout="wide")

st.title("Reconocimiento de Lenguaje de Senas")
st.markdown("Toma una foto haciendo una sena del **alfabeto ASL** y el sistema la identificara.")

col1, col2 = st.columns([2, 1])

with col1:
    foto = st.camera_input("Haz una sena y toma la foto")

    if foto is not None:
        # Convertir la foto a array de OpenCV
        file_bytes = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_resultado, letra, confianza = procesar_imagen(img)

        # Mostrar imagen con landmarks
        st.image(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB), width="stretch")

        if letra:
            st.markdown(f"""
            <div style="background-color: #1a1a2e; padding: 30px; border-radius: 15px; text-align: center; margin-top: 10px;">
                <h1 style="font-size: 100px; margin: 0; color: #00ff88;">{letra.upper()}</h1>
                <p style="font-size: 24px; color: #aaa; margin: 0;">Confianza: {confianza:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No se detecto una mano en la imagen. Intenta de nuevo con la mano mas visible.")

with col2:
    st.markdown("### Instrucciones")
    st.markdown("""
    1. Permite el acceso a la camara
    2. Haz una sena del alfabeto ASL frente a la camara
    3. Haz clic en **Toma una foto** para capturar
    4. El sistema detectara la letra
    5. Toma otra foto para probar otra sena
    """)

    st.markdown("### Alfabeto ASL")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/American_Sign_Language_ASL.svg/800px-American_Sign_Language_ASL.svg.png",
        caption="Referencia del alfabeto ASL",
        width="stretch",
    )

    st.markdown("---")
    st.markdown(f"**Letras disponibles:** {', '.join(sorted(cargar_modelo().classes_)).upper()}")
