import os
import pickle
import time
import numpy as np
import cv2
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

# Conexiones de la mano para dibujar
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def dibujar_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    puntos = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        puntos.append((cx, cy))
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    for i, j in HAND_CONNECTIONS:
        if i < len(puntos) and j < len(puntos):
            cv2.line(frame, puntos[i], puntos[j], (255, 255, 255), 2)


def main():
    if not os.path.exists(MODELO_PATH):
        print(f"Error: No se encontró '{MODELO_PATH}'.")
        print("Primero entrena el modelo con entrenar_modelo.py")
        return

    if not os.path.exists(MODELO_MANO_PATH):
        print(f"Error: No se encontró '{MODELO_MANO_PATH}'.")
        return

    # Cargar modelo de clasificación
    with open(MODELO_PATH, "rb") as f:
        modelo = pickle.load(f)

    print("Modelo cargado.")
    print(f"Letras que reconoce: {sorted(modelo.classes_)}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Crear detector de manos con confianza más baja para mejor detección
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_MANO_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Para suavizar predicciones
    historial = []
    TAMANO_HISTORIAL = 10
    tiempo_inicio = time.time()

    print("=" * 50)
    print("RECONOCIMIENTO DE SEÑAS EN TIEMPO REAL")
    print("=" * 50)
    print("Haz una seña frente a la cámara")
    print("Presiona 'q' para salir")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Usar timestamp real en milisegundos
        timestamp_ms = int((time.time() - tiempo_inicio) * 1000)

        resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

        letra_detectada = ""
        confianza = 0.0
        mano_detectada = False

        if resultado.hand_landmarks:
            mano_detectada = True
            for hand_landmarks in resultado.hand_landmarks:
                dibujar_landmarks(frame, hand_landmarks)

                # Extraer landmarks
                fila = []
                for lm in hand_landmarks:
                    fila.extend([lm.x, lm.y, lm.z])

                datos = np.array(fila).reshape(1, -1)

                # Predecir
                prediccion = modelo.predict(datos)[0]
                probabilidades = modelo.predict_proba(datos)[0]
                confianza = max(probabilidades)

                # Suavizar con historial
                historial.append(prediccion)
                if len(historial) > TAMANO_HISTORIAL:
                    historial.pop(0)

                # La letra más frecuente en el historial
                letra_detectada = max(set(historial), key=historial.count)

                # Debug: imprimir en consola
                print(f"Predicción: {prediccion.upper()} | Confianza: {confianza:.0%} | Suavizada: {letra_detectada.upper()}", end="\r")
        else:
            historial.clear()

        # Mostrar resultado
        if letra_detectada:
            # Fondo negro para el texto
            cv2.rectangle(frame, (0, 0), (350, 100), (0, 0, 0), -1)

            # Letra grande
            color_letra = (0, 255, 0) if confianza > 0.5 else (0, 255, 255)
            cv2.putText(
                frame,
                letra_detectada.upper(),
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                color_letra,
                4,
            )

            # Confianza
            cv2.putText(
                frame,
                f"{confianza:.0%}",
                (130, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2,
            )

            # Predicción directa (sin suavizar)
            cv2.putText(
                frame,
                f"Directa: {prediccion.upper()}",
                (220, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
        elif mano_detectada:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "Mano detectada...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "Muestra tu mano...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Reconocimiento de Senas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
