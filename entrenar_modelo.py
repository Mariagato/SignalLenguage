

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATOS_PATH = "datos_senas.csv"
MODELO_PATH = "modelo_senas.pkl"


def main():
    if not os.path.exists(DATOS_PATH):
        print(f"Error: No se encontró '{DATOS_PATH}'.")
        print("Primero ejecuta recolectar_datos.py para capturar datos.")
        return

    # Cargar datos
    df = pd.read_csv(DATOS_PATH)
    print(f"Datos cargados: {len(df)} muestras")
    print(f"Letras encontradas: {sorted(df['letra'].unique())}")
    print(f"Muestras por letra:")
    print(df["letra"].value_counts().sort_index().to_string())
    print()

    # Separar features y labels
    X = df.drop("letra", axis=1).values
    y = df["letra"].values

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Prueba: {len(X_test)} muestras")
    print()

    # Entrenar modelo
    print("Entrenando modelo...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    # Evaluar
    accuracy = modelo.score(X_test, y_test)
    print(f"Precisión en prueba: {accuracy:.1%}")
    print()

    y_pred = modelo.predict(X_test)
    print("Reporte detallado:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    with open(MODELO_PATH, "wb") as f:
        pickle.dump(modelo, f)

    print(f"Modelo guardado en '{MODELO_PATH}'")


if __name__ == "__main__":
    main()
