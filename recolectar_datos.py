
import csv
import os
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

DATOS_PATH = "datos_senas.csv"
MODELO_MANO_PATH = "hand_landmarker.task"

# Conexiones de la mano para dibujar
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),       # Índice
    (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
    (0, 13), (13, 14), (14, 15), (15, 16),# Anular
    (0, 17), (17, 18), (18, 19), (19, 20),# Meñique
    (5, 9), (9, 13), (13, 17),            # Palma
]


def dibujar_landmarks(frame, hand_landmarks):
    """Dibuja los puntos y conexiones de la mano en el frame."""
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
    if not os.path.exists(MODELO_MANO_PATH):
        print(f"Error: No se encontró '{MODELO_MANO_PATH}'.")
        print("Descárgalo con:")
        print("  curl -sLO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Crear el detector de manos
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_MANO_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    letra_actual = None
    muestras_totales = 0
    muestras_letra = 0
    timestamp_ms = 0

    # Cargar datos previos si existen
    if os.path.exists(DATOS_PATH):
        with open(DATOS_PATH, "r") as f:
            muestras_totales = sum(1 for _ in f) - 1
            if muestras_totales < 0:
                muestras_totales = 0
        print(f"Datos previos encontrados: {muestras_totales} muestras")

    # Preparar archivo CSV
    archivo_existe = os.path.exists(DATOS_PATH) and muestras_totales > 0
    archivo = open(DATOS_PATH, "a", newline="")
    writer = csv.writer(archivo)

    if not archivo_existe:
        # Encabezado: 21 landmarks x 3 coordenadas (x, y, z) = 63 features + label
        encabezado = []
        for i in range(21):
            encabezado.extend([f"x{i}", f"y{i}", f"z{i}"])
        encabezado.append("letra")
        writer.writerow(encabezado)

    print("=" * 50)
    print("RECOLECTOR DE DATOS - LENGUAJE DE SEÑAS")
    print("=" * 50)
    print("Presiona una letra (a-z) para seleccionar la seña")
    print("Haz la seña frente a la cámara")
    print("Presiona 'q' para guardar y salir")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Espejo
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        resultado = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33  # ~30 fps

        # Dibujar landmarks si hay mano detectada
        if resultado.hand_landmarks:
            for hand_landmarks in resultado.hand_landmarks:
                dibujar_landmarks(frame, hand_landmarks)

                # Si hay una letra seleccionada, guardar datos
                if letra_actual is not None:
                    fila = []
                    for lm in hand_landmarks:
                        fila.extend([lm.x, lm.y, lm.z])
                    fila.append(letra_actual)
                    writer.writerow(fila)
                    muestras_totales += 1
                    muestras_letra += 1

        # Mostrar info en pantalla
        color = (0, 255, 0) if letra_actual else (0, 0, 255)
        texto_letra = letra_actual.upper() if letra_actual else "Ninguna"
        cv2.putText(
            frame,
            f"Letra: {texto_letra}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Muestras letra: {muestras_letra}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Total: {muestras_totales}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if not resultado.hand_landmarks:
            cv2.putText(
                frame,
                "No se detecta mano",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Recolectar Datos - Senas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif ord("a") <= key <= ord("z"):
            letra_actual = chr(key)
            muestras_letra = 0
            print(f"Capturando letra: {letra_actual.upper()}")

    archivo.close()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDatos guardados en '{DATOS_PATH}' ({muestras_totales} muestras)")


if __name__ == "__main__":
    main()
