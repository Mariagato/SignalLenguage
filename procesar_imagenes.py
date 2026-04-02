import csv
import os
import sys
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


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    max_por_letra = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    if not os.path.exists(carpeta):
        print(f"Error: No se encontró la carpeta '{carpeta}'.")
        print()
        print("Crea la carpeta con subcarpetas por letra:")
        print(f"  {carpeta}/A/  (fotos de la seña A)")
        print(f"  {carpeta}/B/  (fotos de la seña B)")
        print("  ...")
        print()
        print("Puedes descargar un dataset como 'ASL Alphabet' de Kaggle:")
        print("  https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        return

    if not os.path.exists(MODELO_MANO_PATH):
        print(f"Error: No se encontró '{MODELO_MANO_PATH}'.")
        print("Descárgalo con:")
        print("  curl -sLO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    # Detectar las letras (subcarpetas)
    letras = sorted([
        d for d in os.listdir(carpeta)
        if os.path.isdir(os.path.join(carpeta, d)) and len(d) == 1
    ])

    if not letras:
        print(f"No se encontraron subcarpetas de letras en '{carpeta}'.")
        print("Cada subcarpeta debe tener el nombre de una letra (A, B, C...)")
        return

    print("=" * 50)
    print("PROCESADOR DE IMÁGENES - LENGUAJE DE SEÑAS")
    print("=" * 50)
    print(f"Carpeta: {carpeta}")
    print(f"Letras encontradas: {', '.join(letras)}")
    print(f"Máximo por letra: {max_por_letra}")
    print("=" * 50)

    # Crear detector de manos (modo IMAGE para fotos)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_MANO_PATH),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Preparar CSV
    archivo = open(DATOS_PATH, "w", newline="")
    writer = csv.writer(archivo)

    encabezado = []
    for i in range(21):
        encabezado.extend([f"x{i}", f"y{i}", f"z{i}"])
    encabezado.append("letra")
    writer.writerow(encabezado)

    total_procesadas = 0
    total_detectadas = 0
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for letra in letras:
        ruta_letra = os.path.join(carpeta, letra)
        archivos = [
            f for f in os.listdir(ruta_letra)
            if os.path.splitext(f)[1].lower() in extensiones
        ]

        archivos = archivos[:max_por_letra]  # Limitar cantidad

        detectadas = 0
        for i, nombre_archivo in enumerate(archivos):
            ruta_img = os.path.join(ruta_letra, nombre_archivo)

            try:
                img = cv2.imread(ruta_img)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                resultado = landmarker.detect(mp_image)

                if resultado.hand_landmarks:
                    for hand_landmarks in resultado.hand_landmarks:
                        fila = []
                        for lm in hand_landmarks:
                            fila.extend([lm.x, lm.y, lm.z])
                        fila.append(letra.lower())
                        writer.writerow(fila)
                        detectadas += 1
            except Exception as e:
                print(f"  Error procesando {ruta_img}: {e}")
                continue

            total_procesadas += 1

            # Progreso cada 100 imágenes
            if (i + 1) % 100 == 0:
                print(f"  {letra}: {i + 1}/{len(archivos)} procesadas...")

        total_detectadas += detectadas
        porcentaje = (detectadas / len(archivos) * 100) if archivos else 0
        print(f"  {letra}: {detectadas}/{len(archivos)} manos detectadas ({porcentaje:.0f}%)")

    archivo.close()
    landmarker.close()

    print()
    print("=" * 50)
    print(f"Total imágenes procesadas: {total_procesadas}")
    print(f"Total manos detectadas:    {total_detectadas}")
    print(f"Datos guardados en '{DATOS_PATH}'")
    print("=" * 50)
    print()
    print("Siguiente paso: python entrenar_modelo.py")


if __name__ == "__main__":
    main()
