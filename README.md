# PoC 05: reutilizar un modelo LightGBM preentrenado (resultado negativo)

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Comprobar si un modelo LightGBM entrenado por terceros acepta las variables de nuestro diccionario, ya sea para predecir o para continuar su entrenamiento (`init_model`).

## Resultado
Negativo. LightGBM rechaza la entrada con `LightGBMError`, porque el número de columnas debe ser idéntico al del entrenamiento, tanto en `predict` como en `init_model`.

## Decisión de diseño
El modelo se entrena desde cero con el dataset público elegido.

<!-- SCRIPT: pendiente de ubicar en la laptop; ver instrucciones de Claude Code -->
