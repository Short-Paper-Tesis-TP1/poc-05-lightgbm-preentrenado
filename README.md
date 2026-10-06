# PoC 05: reutilizar un modelo LightGBM preentrenado (resultado negativo)

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Comprobar si un modelo LightGBM entrenado por terceros acepta las variables de nuestro diccionario, ya sea para predecir (`predict`) o para continuar su entrenamiento (`init_model`).

## Datos
Sintéticos. El modelo de terceros se simula con un modelo guardado en archivo, entrenado con 20 columnas ajenas. Nuestros datos usan las 9 variables del diccionario congelado.

## Cómo correrla
```
pip install -r requirements.txt
python poc_lightgbm_preentrenado.py
```

## Resultado
| Prueba | Resultado |
|---|---|
| `predict` con nuestras 9 columnas | Rechazado: `LightGBMError: The number of features in data (9) is not the same as it was in training data (20)` |
| `init_model` para continuar el entrenamiento | Rechazado con el mismo error |
| Rellenar hasta 20 columnas | El modelo devuelve 500 probabilidades (0.350 a 0.953), pero cada columna se interpreta como el atributo ajeno de su misma posición, así que el resultado no tiene significado |

## Decisión de diseño
El modelo se entrena desde cero con el dataset público elegido (Home Credit), porque un LightGBM de terceros exige el mismo número y el mismo significado de columnas.

## Archivos
- `poc_lightgbm_preentrenado.py`: script.
- `resultado_poc5.json`: salida estructurada.
- `salida_consola.txt`: salida completa de la consola.

## Corrida de referencia
Esta es una reconstrucción hecha el 6 de octubre de 2026, porque el script original no se conservó. Se corrió en Linux con Python 3.12.3 y LightGBM 4.7.0. El error obtenido coincide con el registrado originalmente.
