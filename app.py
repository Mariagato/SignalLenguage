import pickle
import numpy as np
import cv2
import av
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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.modelo = cargar_modelo()
        self.landmarker = crear_landmarker()
        self.historial = []
        self.TAMANO_HISTORIAL = 10
        self.letra_actual = ""
        self.confianza_actual = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = self.landmarker.detect(mp_image)

        if resultado.hand_landmarks:
            for hand_landmarks in resultado.hand_landmarks:
                # Dibujar landmarks
                h, w, _ = img.shape
                puntos = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    puntos.append((cx, cy))
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                for i, j in HAND_CONNECTIONS:
                    if i < len(puntos) and j < len(puntos):
                        cv2.line(img, puntos[i], puntos[j], (255, 255, 255), 2)

                # Predecir letra
                fila = []
                for lm in hand_landmarks:
                    fila.extend([lm.x, lm.y, lm.z])
                datos = np.array(fila).reshape(1, -1)

                prediccion = self.modelo.predict(datos)[0]
                probabilidades = self.modelo.predict_proba(datos)[0]
                confianza = max(probabilidades)

                self.historial.append(prediccion)
                if len(self.historial) > self.TAMANO_HISTORIAL:
                    self.historial.pop(0)

                self.letra_actual = max(set(self.historial), key=self.historial.count)
                self.confianza_actual = confianza
        else:
            self.historial.clear()
            self.letra_actual = ""
            self.confianza_actual = 0.0

        # Dibujar resultado
        if self.letra_actual:
            cv2.rectangle(img, (0, 0), (350, 100), (0, 0, 0), -1)
            color = (0, 255, 0) if self.confianza_actual > 0.6 else (0, 255, 255)
            cv2.putText(
                img, self.letra_actual.upper(),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5,
            )
            cv2.putText(
                img, f"{self.confianza_actual:.0%}",
                (140, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2,
            )
        else:
            cv2.rectangle(img, (0, 0), (350, 45), (0, 0, 0), -1)
            cv2.putText(
                img, "Muestra una sena...",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- UI ---
st.set_page_config(page_title="Reconocimiento de Lenguaje de Señas", page_icon="🤟", layout="wide")

st.title(" Reconocimiento de Lenguaje de Señas")
st.markdown("Haz una seña del **alfabeto ASL** frente a la cámara y el sistema la identificará en tiempo real.")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

with col2:
    st.markdown("### Instrucciones")
    st.markdown("""
    1. Haz clic en **START** para activar la cámara
    2. Haz una seña del alfabeto ASL frente a la cámara
    3. La letra detectada aparecerá en la esquina superior
    4. Haz clic en **STOP** para detener
    """)

    st.markdown("### Alfabeto ASL")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/American_Sign_Language_ASL.svg/800px-American_Sign_Language_ASL.svg.png",
        caption="Referencia del alfabeto ASL",
        use_container_width=True,
    )
