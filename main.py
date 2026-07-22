"""
SVM lineal mediante optimización convexa.

Resuelve el problema primal de margen suave:

    minimizar    1/2 ||w||² + C * sum(xi_i)

    sujeto a     y_i (w^T x_i + b) >= 1 - xi_i
                 xi_i >= 0
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SVMConvexo:
    """SVM lineal de margen suave resuelto con CVXPY."""

    def __init__(self, C: float = 1.0):
        if C <= 0:
            raise ValueError("C debe ser mayor que cero.")

        self.C = float(C)
        self.w = None
        self.b = None
        self.xi = None
        self.estado = None
        self.valor_objetivo = None
        self.indices_vectores_soporte = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de muestras.")

        if not np.all(np.isin(np.unique(y), [-1, 1])):
            raise ValueError("Las etiquetas deben ser -1 y +1.")

        n_muestras, n_caracteristicas = X.shape

        w = cp.Variable(n_caracteristicas)
        b = cp.Variable()
        xi = cp.Variable(n_muestras, nonneg=True)

        margenes = cp.multiply(y, X @ w + b)

        objetivo = cp.Minimize(
            0.5 * cp.sum_squares(w) + self.C * cp.sum(xi)
        )

        restricciones = [margenes >= 1 - xi]

        problema = cp.Problem(objetivo, restricciones)

        ultimo_error = None

        for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
            try:
                problema.solve(solver=solver, verbose=False)
                if problema.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception as error:
                ultimo_error = error

        if problema.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                "No fue posible resolver el problema. "
                f"Estado: {problema.status}. Último error: {ultimo_error}"
            )

        self.w = np.asarray(w.value).reshape(-1)
        self.b = float(b.value)
        self.xi = np.asarray(xi.value).reshape(-1)
        self.estado = problema.status
        self.valor_objetivo = float(problema.value)

        margenes_numericos = y * (X @ self.w + self.b)

        self.indices_vectores_soporte = np.where(
            margenes_numericos <= 1 + 1e-4
        )[0]

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._verificar_entrenamiento()
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        valores = self.decision_function(X)
        return np.where(valores >= 0, 1, -1)

    def _verificar_entrenamiento(self):
        if self.w is None or self.b is None:
            raise RuntimeError("El modelo todavía no ha sido entrenado.")


def graficar_svm(modelo, X, y, nuevo_punto=None):
    """Grafica la frontera, los márgenes y los vectores de soporte."""

    if X.shape[1] != 2:
        raise ValueError("La gráfica requiere exactamente dos variables.")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )

    malla = np.c_[xx.ravel(), yy.ravel()]
    valores = modelo.decision_function(malla).reshape(xx.shape)

    plt.figure(figsize=(10, 7))

    plt.contourf(
        xx,
        yy,
        valores,
        levels=[-np.inf, 0, np.inf],
        alpha=0.2,
    )

    contornos = plt.contour(
        xx,
        yy,
        valores,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        linewidths=[1.5, 2.5, 1.5],
    )

    plt.clabel(
        contornos,
        fmt={-1: "Margen -1", 0: "Frontera", 1: "Margen +1"},
    )

    plt.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        marker="o",
        label="Clase -1",
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        marker="^",
        label="Clase +1",
    )

    indices = modelo.indices_vectores_soporte

    if indices is not None and len(indices) > 0:
        plt.scatter(
            X[indices, 0],
            X[indices, 1],
            s=190,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="Vectores de soporte",
        )

    if nuevo_punto is not None:
        plt.scatter(
            nuevo_punto[0, 0],
            nuevo_punto[0, 1],
            marker="*",
            s=300,
            edgecolors="black",
            label="Nuevo punto",
        )

    plt.title("SVM lineal mediante optimización convexa")
    plt.xlabel("Característica 1 normalizada")
    plt.ylabel("Característica 2 normalizada")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Crear datos de ejemplo.
    X, y_original = make_blobs(
        n_samples=150,
        centers=[(-2, -2), (2, 2)],
        cluster_std=1.6,
        random_state=10,
    )

    # Cambiar etiquetas 0 y 1 por -1 y +1.
    y = np.where(y_original == 0, -1, 1)

    # Separar entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=10,
        stratify=y,
    )

    # Normalizar los datos.
    escalador = StandardScaler()
    X_train_std = escalador.fit_transform(X_train)
    X_test_std = escalador.transform(X_test)

    # Entrenar el modelo.
    modelo = SVMConvexo(C=1.0)
    modelo.fit(X_train_std, y_train)

    # Evaluar.
    predicciones = modelo.predict(X_test_std)
    exactitud = accuracy_score(y_test, predicciones)
    matriz = confusion_matrix(y_test, predicciones)

    print("=" * 60)
    print("RESULTADOS DEL SVM")
    print("=" * 60)
    print(f"Estado: {modelo.estado}")
    print(f"Valor óptimo: {modelo.valor_objetivo:.6f}")
    print(f"Pesos óptimos w: {np.round(modelo.w, 6)}")
    print(f"Sesgo óptimo b: {modelo.b:.6f}")
    print(f"Norma de w: {np.linalg.norm(modelo.w):.6f}")
    print(f"Margen geométrico: {1 / np.linalg.norm(modelo.w):.6f}")
    print(f"Holgura total: {np.sum(modelo.xi):.6f}")
    print(
        "Vectores de soporte: "
        f"{len(modelo.indices_vectores_soporte)}"
    )
    print(f"Exactitud: {exactitud:.4f}")
    print("\nMatriz de confusión:")
    print(matriz)

    # Ejemplo de predicción.
    nuevo_punto_original = np.array([[1.5, 2.0]])
    nuevo_punto_std = escalador.transform(nuevo_punto_original)

    clase = modelo.predict(nuevo_punto_std)[0]
    valor = modelo.decision_function(nuevo_punto_std)[0]

    print("\n" + "=" * 60)
    print("EJEMPLO DE PREDICCIÓN")
    print("=" * 60)
    print(f"Punto original: {nuevo_punto_original[0]}")
    print(f"Valor de decisión: {valor:.6f}")
    print(f"Clase predicha: {clase}")

    graficar_svm(
        modelo,
        X_train_std,
        y_train,
        nuevo_punto_std,
    )


if __name__ == "__main__":
    main()
