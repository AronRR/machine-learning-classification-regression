"""Random Forest para Diamonds: búsqueda progresiva y resultados para el paper.

Coloca diamonds.csv junto al script y ejecuta este archivo.
Dependencias: pandas, matplotlib y scikit-learn.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def evaluar_modelo(modelo, conjuntos):
    resultados = []
    for nombre, X, y in conjuntos:
        pred = modelo.predict(X)
        resultados.append({"Conjunto": nombre,
                           "MSE": mean_squared_error(y, pred),
                           "R2": r2_score(y, pred)})
    return pd.DataFrame(resultados)


def guardar_tabla(tabla, carpeta, nombre):
    tabla.to_csv(carpeta / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    texto = tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}")
    (carpeta / f"{nombre}.txt").write_text(texto, encoding="utf-8")
    print(f"\n{nombre}\n{texto}", flush=True)


def guardar_figura(carpeta, nombre):
    plt.tight_layout()
    plt.savefig(carpeta / f"{nombre}.png", dpi=300, bbox_inches="tight")
    plt.close()


def elegir_mejor(tabla):
    # MSE de Validation decide; los empates exactos favorecen menor complejidad.
    return tabla.sort_values(["Validation MSE", "max_depth", "n_estimators"]).iloc[0]


def graficar_predicciones(y, pred, titulo, limites, carpeta, nombre, color):
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y, pred, alpha=0.22, s=10, color=color, edgecolors="none")
    plt.plot(limites, limites, "--", color="#333333", label="Predicción ideal")
    plt.xlim(limites)
    plt.ylim(limites)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Precio real (USD)")
    plt.ylabel("Precio predicho (USD)")
    plt.title(titulo)
    plt.legend()
    guardar_figura(carpeta, nombre)


def main():
    carpeta = Path(__file__).resolve().parent
    archivo = carpeta / "diamonds.csv"
    if not archivo.exists():
        raise FileNotFoundError("Coloca diamonds.csv en la misma carpeta que este script.")
    salida = carpeta / "resultados_random_forest"
    tablas, figuras = salida / "tablas", salida / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    # 1. Misma limpieza del original: dimensiones en cero y duplicados.
    columnas = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
    datos = pd.read_csv(archivo)
    faltantes = [c for c in columnas if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en diamonds.csv: {faltantes}")
    datos = datos[columnas]  # Descarta también las columnas de índice del CSV.
    iniciales = len(datos)
    datos = datos[(datos["x"] != 0) & (datos["y"] != 0) & (datos["z"] != 0)]
    sin_ceros = len(datos)
    datos = datos.drop_duplicates()
    if datos.isna().any().any():
        raise ValueError("El CSV contiene valores faltantes. Revisa los datos.")

    # 2. Misma aleatorización y división manual 60/20/20, con semilla 42.
    datos = datos.sample(frac=1, random_state=42)
    fin_train, fin_val = int(len(datos) * 0.60), int(len(datos) * 0.80)
    train, val, test = datos.iloc[:fin_train], datos.iloc[fin_train:fin_val], datos.iloc[fin_val:]
    categoricas = ["cut", "color", "clarity"]
    X_train = pd.get_dummies(train.drop(columns="price"), columns=categoricas, dtype=int)
    # Las columnas de Train definen la codificación; categorías nuevas quedan en cero.
    X_val = pd.get_dummies(val.drop(columns="price"), columns=categoricas, dtype=int)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    y_train, y_val = train["price"], val["price"]
    conjuntos = [("Train", X_train, y_train), ("Validation", X_val, y_val)]
    # Test permanece separado: se codifica y predice sólo tras la selección final.
    ficha = pd.DataFrame({"Dato": ["Filas iniciales", "Filas con dimensiones cero",
        "Duplicados eliminados después de ceros", "Filas finales", "Variables originales",
        "Variables tras One-Hot", "Target", "Train", "Validation", "Test"],
        "Valor": [iniciales, iniciales - sin_ceros, sin_ceros - len(datos), len(datos),
                  9, X_train.shape[1], "price (USD)", len(train), len(val), len(test)]})
    guardar_tabla(ficha, tablas, "tabla_I_dataset")

    # 3. Entrenar cada configuración una vez; guardar sólo métricas Train/Validation.
    resultados = {}
    def probar(arboles, profundidad):
        clave = (arboles, profundidad)
        if clave not in resultados:
            print(f"Probando {arboles} árboles, profundidad {profundidad}...", flush=True)
            modelo = RandomForestRegressor(n_estimators=arboles, max_depth=profundidad,
                                           random_state=42)
            modelo.fit(X_train, y_train)
            metricas = evaluar_modelo(modelo, conjuntos).set_index("Conjunto")
            fila = {"n_estimators": arboles, "max_depth": profundidad}
            for conjunto in ["Train", "Validation"]:
                for metrica in ["MSE", "R2"]:
                    fila[f"{conjunto} {metrica}"] = metricas.loc[conjunto, metrica]
            resultados[clave] = fila
        return resultados[clave]

    base = probar(50, 10)
    tabla_base = pd.DataFrame([{"Conjunto": c, "MSE": base[f"{c} MSE"],
                               "R2": base[f"{c} R2"]} for c in ["Train", "Validation"]])
    guardar_tabla(tabla_base, tablas, "tabla_II_modelo_base")

    # 4. Experimentos separados: sólo cambia un hiperparámetro en cada uno.
    experimento_arboles = pd.DataFrame([probar(n, 10) for n in [50, 100, 200]])
    guardar_tabla(experimento_arboles, tablas, "tabla_III_arboles")
    arboles_fijos = 100
    experimento_depth = pd.DataFrame([probar(arboles_fijos, d) for d in [5, 10, 15, 20]])
    guardar_tabla(experimento_depth, tablas, "tabla_IV_profundidad")

    # 5. Refinar una sola vez: mejor profundidad y sus dos vecinos a distancia 2.
    centro = int(elegir_mejor(experimento_depth)["max_depth"])
    vecinos = sorted(set([max(1, centro - 2), centro, centro + 2]))
    refinamiento = pd.DataFrame([probar(arboles_fijos, d) for d in vecinos])
    profundidad_local = int(elegir_mejor(refinamiento)["max_depth"])
    # Comprobar de nuevo árboles, ahora fijando la mejor profundidad local.
    cierre = pd.DataFrame([probar(n, profundidad_local) for n in [50, 100, 200]])
    refinamiento = pd.concat([refinamiento, cierre]).drop_duplicates(
        subset=["n_estimators", "max_depth"]).reset_index(drop=True)
    guardar_tabla(refinamiento, tablas, "tabla_V_refinamiento")

    # 6. Elegir entre TODOS los candidatos, incluido el base, sólo con Validation.
    candidatos = pd.DataFrame(resultados.values())
    mejor = elegir_mejor(candidatos)
    n_final, d_final = int(mejor["n_estimators"]), int(mejor["max_depth"])
    seleccion = (f"Selección por menor MSE Validation: {n_final} árboles, profundidad {d_final}.\n"
                 f"MSE Validation: {mejor['Validation MSE']:.6f}.\n"
                 "Empates exactos: menor profundidad y después menos árboles.\n"
                 "Búsqueda progresiva limitada; no garantiza un óptimo global.\n")
    (salida / "seleccion.txt").write_text(seleccion, encoding="utf-8")
    print(seleccion, flush=True)

    # 7. Selección cerrada. Reentrenar sólo con Train reproduce los candidatos.
    # No unir Train y Validation: la comparación usa el mismo entrenamiento.
    modelo_base = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    modelo_ajustado = RandomForestRegressor(n_estimators=n_final, max_depth=d_final, random_state=42)
    modelo_base.fit(X_train, y_train)
    modelo_ajustado.fit(X_train, y_train)
    X_test = pd.get_dummies(test.drop(columns="price"), columns=categoricas, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = test["price"]
    finales = conjuntos + [("Test", X_test, y_test)]
    resumen = evaluar_modelo(modelo_base, finales).merge(
        evaluar_modelo(modelo_ajustado, finales), on="Conjunto", suffixes=(" base", " ajustado"))
    resumen["Reduccion MSE (%)"] = 100 * (resumen["MSE base"] - resumen["MSE ajustado"]) / resumen["MSE base"]
    guardar_tabla(resumen, tablas, "tabla_VI_base_vs_ajustado")

    # 8. Ocho figuras del paper. Incluso el histograma completo se genera al final.
    azul, naranja = "#2266A5", "#D97828"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18})
    plt.figure(figsize=(7, 4.5))
    plt.hist(datos["price"], bins=40, color=azul, edgecolor="white")
    plt.xlabel("Precio (USD)")
    plt.ylabel("Número de diamantes")
    plt.title("Distribución del precio | Dataset limpio")
    guardar_figura(figuras, "fig_01_distribucion_price")

    pred_base, pred_ajustado = modelo_base.predict(X_test), modelo_ajustado.predict(X_test)
    limites = (0, 1.03 * max(y_test.max(), pred_base.max(), pred_ajustado.max()))
    for numero, pred, nombre, color in [(2, pred_base, "base", azul), (6, pred_ajustado, "ajustado", naranja)]:
        graficar_predicciones(y_test, pred, f"Modelo {nombre} | Test", limites, figuras,
                              f"fig_{numero:02d}_real_vs_predicho_{nombre}", color)

    plt.figure(figsize=(7, 4.5))
    plt.plot(experimento_arboles["n_estimators"], experimento_arboles["Validation MSE"], "o-", color=azul)
    plt.xticks([50, 100, 200])
    plt.ylim(0, experimento_arboles["Validation MSE"].max() * 1.18)
    for n, mse in zip(experimento_arboles["n_estimators"], experimento_arboles["Validation MSE"]):
        plt.annotate(f"{mse:.0f}", (n, mse), xytext=(0, 9), textcoords="offset points", ha="center")
    plt.xlabel("Número de árboles")
    plt.ylabel("MSE Validation (USD²)")
    plt.title("Efecto del número de árboles | Profundidad 10")
    guardar_figura(figuras, "fig_03_efecto_arboles")

    plt.figure(figsize=(7, 4.5))
    for c, color in [("Train", azul), ("Validation", naranja)]:
        plt.plot(experimento_depth["max_depth"], experimento_depth[f"{c} R2"], "o-", label=c, color=color)
    plt.xticks([5, 10, 15, 20])
    plt.xlabel("Profundidad máxima")
    plt.ylabel("R²")
    plt.title(f"Efecto de la profundidad | {arboles_fijos} árboles")
    plt.legend()
    guardar_figura(figuras, "fig_04_efecto_profundidad")

    plt.figure(figsize=(8, 4.8))
    etiquetas = [f"{int(f.n_estimators)} árboles\nProf. {int(f.max_depth)}" for f in refinamiento.itertuples()]
    barras = plt.bar(etiquetas, refinamiento["Validation MSE"], color=azul)
    plt.bar_label(barras, fmt="%.0f", padding=3)
    plt.margins(y=0.18)
    plt.ylabel("MSE Validation (USD²)")
    plt.title("Candidatos del refinamiento local")
    guardar_figura(figuras, "fig_05_refinamiento")

    for numero, metrica, etiqueta in [(7, "MSE", "MSE (USD²)"), (8, "R2", "R²")]:
        plt.figure(figsize=(8, 5))
        for desplazamiento, nombre, color in [(-0.18, "base", azul), (0.18, "ajustado", naranja)]:
            barras = plt.bar([i + desplazamiento for i in range(3)], resumen[f"{metrica} {nombre}"],
                             width=0.36, label=f"Modelo {nombre}", color=color)
            plt.bar_label(barras, fmt="%.0f" if metrica == "MSE" else "%.4f", padding=3, fontsize=9)
        plt.xticks(range(3), resumen["Conjunto"])
        plt.ylabel(etiqueta)
        plt.title(f"{etiqueta}: base vs ajustado")
        plt.legend(loc="upper center", ncol=2)
        plt.margins(y=0.22)
        guardar_figura(figuras, f"fig_{numero:02d}_{metrica.lower()}_base_vs_ajustado")

    print(f"Resultados guardados en: {salida}")
    print("La brecha Train-Validation permite discutir posible sobreajuste; no mide formalmente bias/varianza.")


if __name__ == "__main__":
    main()
