# Proyecto de Machine Learning

Este proyecto incluye dos implementaciones de aprendizaje automático:

- Regresión logística implementada manualmente para clasificación de sitios web de phishing.
- Random Forest con scikit-learn para regresión sobre el dataset Diamonds.

## Mejoras realizadas desde la primera revisión

A partir de la primera retroalimentación se realizaron varios ajustes al proyecto.

Se separaron los dos modelos en papers independientes para que cada uno tuviera un objetivo y análisis más claro. También se simplificó el EDA y se mantuvieron únicamente las revisiones necesarias para justificar la preparación de los datos.

En ambos modelos se revisó la división entre Train, Validation y Test. En particular, Test se mantuvo reservado hasta la evaluación final para evitar utilizarlo durante la selección de parámetros.

En el modelo de Random Forest se amplió el análisis de hiperparámetros. En lugar de comparar únicamente algunas configuraciones aisladas, se realizó una búsqueda progresiva estudiando por separado el número de árboles y la profundidad máxima, seguida de un refinamiento local.

La configuración final seleccionada fue de 200 árboles y profundidad máxima de 18, al obtener el menor MSE en Validation entre las configuraciones evaluadas.

También se agregaron nuevas tablas y gráficas para comparar el modelo base con el ajustado y analizar de forma más clara el bias, la varianza y el nivel de fitting.

En Test, el MSE disminuyó de 527,360.1483 a 345,534.1000 y el R² aumentó de 0.9670 a 0.9783.

Estas modificaciones permitieron mejorar tanto la metodología del experimento como la interpretación y presentación de los resultados.
