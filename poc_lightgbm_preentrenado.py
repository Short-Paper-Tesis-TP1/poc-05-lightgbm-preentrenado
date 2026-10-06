"""
PoC 05 | Objetivo unico: comprobar si un modelo LightGBM entrenado por terceros
acepta las variables de nuestro diccionario, para predecir (predict) o para
continuar su entrenamiento (init_model).

Reconstruccion del 6 de octubre de 2026: el modelo de terceros original no se
conservo, asi que se simula con un modelo guardado en archivo, entrenado con
20 columnas ajenas (estructura tipo German Credit). Nuestros datos usan las
9 variables del diccionario congelado. Ambos conjuntos son SINTETICOS.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

SEMILLA = 42
rng = np.random.default_rng(SEMILLA)
resultado = {"poc_id": "PoC-05-LightGBM-preentrenado", "pruebas": {}}

# ---------------------------------------------------------------- PASO 1
print("=" * 72)
print("PASO 1 | MODELO DE TERCEROS (20 columnas ajenas, guardado en archivo)")
cols_terceros = [f"atributo_{i:02d}" for i in range(1, 21)]
X_t = pd.DataFrame(rng.normal(size=(1000, 20)), columns=cols_terceros)
y_t = (X_t.iloc[:, 0] + rng.normal(0, 1, 1000) > 0).astype(int)
lgb.train({"objective": "binary", "verbose": -1, "seed": SEMILLA},
          lgb.Dataset(X_t, y_t), num_boost_round=50).save_model("modelo_terceros.txt")
ajeno = lgb.Booster(model_file="modelo_terceros.txt")
print(f"  Columnas esperadas por el modelo: {ajeno.num_feature()}")

# ---------------------------------------------------------------- PASO 2
VARIABLES = ["ratio_cuota_ingreso", "ratio_deuda_ingreso", "ratio_monto_ingreso",
             "n_otros_creditos", "atraso_historial", "edad", "nivel_educativo",
             "n_hijos", "anios_negocio"]
n = 500
X = pd.DataFrame({
    "ratio_cuota_ingreso": rng.gamma(2, 0.08, n), "ratio_deuda_ingreso": rng.gamma(2, 0.6, n),
    "ratio_monto_ingreso": rng.gamma(2, 1.0, n), "n_otros_creditos": rng.poisson(1.5, n),
    "atraso_historial": rng.binomial(1, 0.12, n), "edad": rng.integers(18, 70, n),
    "nivel_educativo": rng.integers(0, 4, n), "n_hijos": rng.poisson(1.0, n),
    "anios_negocio": rng.gamma(2, 3, n)})[VARIABLES]
y = rng.binomial(1, 0.07, n)
print(f"\nPASO 2 | NUESTROS DATOS: {X.shape[1]} variables del diccionario congelado")


def probar(nombre, funcion):
    try:
        funcion()
        print(f"  {nombre:<44} ACEPTADO")
        resultado["pruebas"][nombre] = {"aceptado": True}
    except lgb.basic.LightGBMError as e:
        msg = str(e).strip().splitlines()[0]
        print(f"  {nombre:<44} RECHAZADO -> LightGBMError: {msg}")
        resultado["pruebas"][nombre] = {"aceptado": False, "error": msg}


# ---------------------------------------------------------------- PASO 3
print("\nPASO 3 | USAR EL MODELO DE TERCEROS CON NUESTRAS VARIABLES")
probar("predict con 9 columnas", lambda: ajeno.predict(X))
probar("init_model (continuar entrenamiento)",
       lambda: lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y),
                         num_boost_round=10, init_model="modelo_terceros.txt"))

# ---------------------------------------------------------------- PASO 4
print("\nPASO 4 | RELLENAR HASTA 20 COLUMNAS PARA QUE EL MODELO ACEPTE LA ENTRADA")
X20 = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=[f"relleno_{i}" for i in range(11)])], axis=1)
p = ajeno.predict(X20.values)
print(f"  Con 9 columnas + 11 de relleno el modelo devuelve {len(p)} probabilidades")
print(f"  (min {p.min():.3f}, max {p.max():.3f}), pero cada columna se lee como el")
print("  atributo ajeno de su misma posicion: el resultado no tiene significado.")
resultado["forzado"] = {"filas": len(p), "min": round(float(p.min()), 4), "max": round(float(p.max()), 4),
                        "interpretacion": "sin significado: las columnas no corresponden a los atributos del modelo"}

resultado["conclusion"] = "Un LightGBM de terceros no puede reutilizarse: exige el mismo numero y significado de columnas."
json.dump(resultado, open("resultado_poc5.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nPASO 5 | Archivo generado: resultado_poc5.json")
print("=" * 72)
