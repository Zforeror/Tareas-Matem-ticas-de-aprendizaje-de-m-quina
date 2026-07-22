# SVM mediante optimización convexa

Este repositorio implementa un **Support Vector Machine lineal de margen suave**
resolviendo directamente su problema de optimización convexa con `CVXPY`.

## Modelo matemático

Se resuelve:

```text
minimizar    1/2 ||w||² + C sum(xi_i)

sujeto a     y_i (wᵀx_i + b) >= 1 - xi_i
             xi_i >= 0
```

Donde:

- `w` es el vector de pesos.
- `b` es el sesgo.
- `xi_i` son las variables de holgura.
- `C` controla la penalización de los errores.

## Archivos

```text
svm-optimizacion-convexa/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalación

```bash
git clone URL_DEL_REPOSITORIO
cd svm-optimizacion-convexa
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

El programa genera datos de ejemplo, entrena el SVM, imprime los pesos óptimos,
calcula la exactitud, clasifica un nuevo punto y muestra la frontera de decisión,
los márgenes y los vectores de soporte.

## Cambiar el parámetro C

En `main.py` puedes modificar:

```python
modelo = SVMConvexo(C=1.0)
```

Un valor grande de `C` penaliza más los errores. Un valor pequeño permite más
errores para intentar obtener un margen más amplio.
