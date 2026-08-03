# 🌟 **Aurum Market — Búsqueda Semántica y Control de Catálogo**  
### _Sistema vectorial completo con Qdrant + E5-small_

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" />
  <img src="https://img.shields.io/badge/Qdrant-VectorDB-purple?logo=qdrant" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" />
  <img src="https://img.shields.io/badge/Embeddings-E5--small-green" />
  <img src="https://img.shields.io/badge/Status-Complete-success" />
</p>

---

# 📘 **Descripción del proyecto**

Aurum Market implementa un sistema completo de:

- **Búsqueda semántica** sobre un catálogo de ~15.000 productos.  
- **Control de altas** mediante detección de duplicados con embeddings.  
- **Evaluación reproducible** con métricas de ranking, latencia y fidelidad ANN.  

El proyecto está compuesto por:

- **Notebooks 01–09** que implementan todo el pipeline vectorial.  
- **Código en `src/`** que encapsula funciones reutilizables.  
- **Artefactos en `results/`** generados durante la evaluación final.  

# 📌 **Nota importante para la evaluación**

Aunque el proyecto incluye todo el recorrido completo (notebooks 01–09),
el entregable principal a tener en consideración es el Notebook 09 ([09_evaluacion_final.ipynb](notebooks/09_evaluacion_final.ipynb)).

Este notebook:

- Ejecuta el pipeline completo sobre el catálogo final,
- Genera los artefactos obligatorios y solicitados en el archivo Enunciado.pdf:
    - Resultados_busqueda.csv
    - Resultados_duplicados.csv
    - Metricas_desarrollo.json
- Valida la ingesta de 15.000 productos,
- Calcula las métricas finales (nDCG, Recall, MRR, latencia, fidelidad ANN),
- y demuestra que el sistema funciona de principio a fin.

Los notebooks anteriores existen para mostrar el desarrollo, la experimentación y la construcción del sistema, pero *Notebook 09* es el punto de verificación final del proyecto.

---

# ⚙️ **1. Instalación del entorno**

### Crear entorno virtual

```bash
python3.11 -m venv env-aurum_market
source env-aurum_market/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Registrar kernel para Jupyter

```bash
python -m ipykernel install --user --name env-aurum_market --display-name "env-aurum_market"
jupyter kernelspec list
```

---

# 🗄️ **2. Levantar Qdrant con Docker**

### Ubicarse en la carpeta del docker-compose

```bash
cd docker/
```

### Levantar Qdrant

```bash
docker-compose up -d
```

### Verificar que Qdrant está corriendo

```bash
docker ps
```

Si ya existe el contenedor:

```bash
docker start qdrant
```

---

# 🔐 **3. Variables de entorno**

Crear el archivo `.env` a partir de la plantilla:

```bash
cp .env.template .env
```

---

# 📂 **4. Estructura del proyecto**

```
aurum_market/
│
├── data/
│   ├── ATRIBUCION_DATOS.md
│   ├── README_DATOS.md
│   ├── altas_desarrollo.csv
│   ├── altas_evaluacion.csv
│   ├── catalogo_muestra.csv
│   ├── catalogo_productos.csv.gz
│   ├── consultas_desarrollo.csv
│   ├── consultas_evaluacion.csv
│   ├── consultas_filtradas.csv
│   ├── eventos_catalogo.csv
│   ├── manifest.json
│   └── relevancias_desarrollo.csv
│
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_baseline_lexico.ipynb
│   ├── 03_embeddings_experimentos.ipynb
│   ├── 04_ingesta_e_indice_vectorial.ipynb
│   ├── 05_busqueda_y_filtros.ipynb
│   ├── 06_eventos_catalogo.ipynb
│   ├── 07_duplicados.ipynb
│   ├── 08_metricas_y_fidelidad_ann.ipynb
│   └── 09_evaluacion_final.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── duplicates.py
│   ├── embeddings.py
│   ├── events.py
│   ├── filters.py
│   ├── ingestion.py
│   ├── metrics.py
│   ├── search_engine.py
│   ├── utils.py
│   └── vector_ingestion.py
│
├── results/
│   ├── resultados_busqueda.csv
│   ├── resultados_duplicados.csv
│   └── metricas_desarrollo.json
│
├── db/
│   └── aurum.db
│
├── docker/
│   └── docker-compose.yml
│
├── .env
├── .env.template
├── .gitignore
└── requirements.txt

```

---

# 🚀 **5. Cómo ejecutar el proyecto**

> El flujo del proyecto se ha reproducido en Visual Studio Code usando notebooks de Jupyter.
> Si prefieres hacerlo en Jupyter Notebook clásico, el contenido es el mismo.

### 1. Levantar Qdrant  
(Sección anterior)

### 2. Abrir Visual Studio Code

Abrir la carpeta del proyecto en VS Code y usar la extensión de Jupyter para ejecutar los notebooks.

```bash
code .
```

Seleccionar el kernel:

```
env-aurum_market
```

### 3. Ejecutar los notebooks en orden

1. `01_exploracion_datos.ipynb`  
2. `02_baseline_lexico.ipynb`  
3. `03_embeddings_experimentos.ipynb`  
4. `04_ingesta_e_indice_vectorial.ipynb` → ingesta completa (15.000 productos)  
5. `05_busqueda_y_filtros.ipynb`  
6. `06_eventos_catalogo.ipynb`  
7. `07_duplicados.ipynb`  
8. `08_metricas_y_fidelidad_ann.ipynb`  
9. `09_evaluacion_final.ipynb` → genera los artefactos obligatorios

---

# 📊 **6. Artefactos generados (obligatorios según lo solicitado en el archivo Enunciado.pdf)**

### ✔ `results/resultados_busqueda.csv`
- 12 consultas ciegas  
- 10 resultados por consulta  
- 120 filas totales  
- Formato: `evaluation_id`, `rank`, `product_id`, `score`

### ✔ `results/metricas_desarrollo.json`
Con los valores reales:

```json
{
  "ndcg_at_10": 0.7413,
  "recall_at_10": 0.3213,
  "mrr_at_10": 0.71875,
  "latency_p50_ms": 94.8,
  "latency_p95_ms": 103.5
}
```

### ✔ `results/resultados_duplicados.csv`
- 14 predicciones  
- threshold ≈ 0.9103  
- F1 ≈ 1.0  

---

# 📈 **7. Resultados finales del sistema**

### 🔹 Métricas de ranking (catálogo completo)
- **nDCG@10 = 0.7413**  
- **Recall@10 = 0.3213**  
- **MRR@10 = 0.71875**

### 🔹 Latencia
- **p50 ≈ 94.8 ms**  
- **p95 ≈ 103.5 ms**

### 🔹 Fidelidad ANN
- **0.6875**

### 🔹 Duplicados
- threshold óptimo: **0.9103**  
- F1 ≈ **1.0**  
- duplicados reales: similitud **0.92–0.98**

---

# 🧠 **8. Decisiones técnicas: por qué Qdrant + E5-small**

El diseño del sistema Aurum Market responde a necesidades reales del proyecto y fue elegido tras evaluar alternativas como FAISS, Milvus, Weaviate y modelos de embeddings más grandes.

## 🔷 1. Por qué Qdrant

### ✔ Filtros estructurados nativos  
Aurum Market exige combinar similitud semántica + filtros por metadatos (`brand`, `color`, `active`, `catalog_version`).  
Qdrant permite aplicar filtros **directamente en el índice ANN**, sin postprocesado.

### ✔ Índice ANN HNSW rápido y estable  
Resultados reales:  
- p50 ≈ **94.8 ms**  
- p95 ≈ **103.5 ms**  
- Fidelidad ANN = **0.6875**

### ✔ Payload estructurado integrado  
Qdrant almacena vector + metadatos y permite filtrarlos en la misma operación.

### ✔ Idempotencia en eventos  
Permite upsert estable y sincronización del catálogo.

---

## 🔷 2. Por qué E5-small

### ✔ Latencia y coste computacional  
Embeddings de 384 dimensiones, rápidos y ligeros.  
Ingesta real: **15.000 productos → ~25 min en CPU**.

### ✔ Estabilidad en duplicados  
Resultados reales:  
- threshold ≈ **0.9103**  
- F1 ≈ **1.0**  
- duplicados reales: **0.92–0.98**

### ✔ Calidad suficiente  
Métricas reales:  
- nDCG@10 = **0.7413**  
- Recall@10 = **0.3213**  
- MRR@10 = **0.71875**

### ✔ Reproducibilidad  
La ejecución del proyecto se ha validado en CPU en este entorno, sin requerir GPU.

---

## 🔷 3. Conclusión técnica

Elegimos **Qdrant + E5-small** porque es la combinación que mejor satisface:

- rendimiento,  
- simplicidad,  
- estabilidad,  
- reproducibilidad,  
- coste computacional,  
- calidad de ranking,  
- fidelidad ANN,  
- control de duplicados,  
- filtros estructurados,  
- despliegue con Docker,  
- ejecución en CPU,  
- y facilidad para el profesor.

---


# 🛑 **9. Notas importantes**

### ❌ No se sube la base de datos de Qdrant al repositorio  
Se genera automáticamente al ejecutar los notebooks.

### ✔ El proyecto es completamente reproducible  
Como he comentado anteriormente, cualquier persona que se descargue el repositorio puede:

- levantar Qdrant,  
- ejecutar los notebooks,  
- regenerar el índice,  
- obtener los mismos artefactos.

---

# 🎉 **10. Estado final del proyecto**

✔ Búsqueda semántica  
✔ Filtros por metadatos  
✔ Control de duplicados  
✔ Métricas de ranking  
✔ Latencia  
✔ Fidelidad ANN  
✔ Artefactos obligatorios generados  
✔ Ingesta del catálogo completo  

---