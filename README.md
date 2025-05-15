# Evaluación de un Pipeline RAG en Español para Constituciones de Argentina

Este repositorio documenta una serie de experimentos realizados para evaluar y
mejorar un sistema **RAG (Retrieval-Augmented Generation)** aplicado sobre la
**Constitución Nacional Argentina** entre otras. Se exploraron múltiples
configuraciones en cuanto a embeddings, chunking, número de documentos
recuperados (`K`), y prompts.

---

## Experimentos

Cada experimento está organizado en una carpeta numerada (`1.` a `6.`) con sus
respectivos pipelines y resultados (`ragas_easy.json`, `ragas_hard.json`).

| Experimento | Embedding               | Chunking          | K   | Prompt     |
| ----------- | ----------------------- | ----------------- | --- | ---------- |
| Baseline    | `multilingual-e5-large` | Fijos con overlap | 5   | FAQ        |
| Exp 2       | `bge-m3-es-legal-tmp-6` | Fijos con overlap | 5   | FAQ        |
| Exp 3       | `bge-m3-es-legal-tmp-6` | Artículos         | 5   | FAQ        |
| Exp 4       | `bge-m3-es-legal-tmp-6` | Incisos           | 5   | FAQ        |
| Exp 5       | `bge-m3-es-legal-tmp-6` | Incisos           | 10  | FAQ        |
| Exp 6       | `bge-m3-es-legal-tmp-6` | Incisos           | 10  | Ingeniería |

---

## Métricas de Evaluación

### Usamos dos enfoques:

1. **[Ragas](https://github.com/explodinggradients/ragas)**:

   - Faithfulness
   - Context Precision
   - Context Recall

2. **Módulo personalizado (`rag_metrics`)** con métricas adicionales:
   - `faithfulness`
   - `context_precision`
   - `context_recall`
   - `answer_relevance`
   - `answer_correctness`
   - `retrieval_precision@k`
   - `retrieval_recall@k`
   - `rouge_l`

### Ejecución de métricas:

Los archivos `ragas_easy.json` y `ragas_hard.json` contienen los datos para
evaluar cada experimento, dentro de cada uno de los pipelines se ejecutó la
evaluación con la librería Ragas. El script `evaluate_custom.py` permite
ejecutar las métricas definidas en el módulo personalizado (`rag_metrics/`).

---

## Estructura del Proyecto

```
.
├── 1.chunk-generico/        ← Experimento base
├── 2.leg-embedding/         ← Cambio de embedding
├── 3.article-chunks/        ← Chunking por artículos
├── 4.incisos-chunks/        ← Chunking por incisos
├── 5.k/                     ← Incremento de K a 10
├── 6.prompt/                ← Cambio de prompt
├── rag_metrics/             ← Módulo con métricas personalizadas
├── rag_analysis/            ← Análisis y visualizaciones
│   ├── *.py                 ← Scripts de análisis
│   ├── results_custom_*.csv
│   └── combined_data.csv
├── evaluate_custom.py       ← Script principal para evaluar con métricas propias
├── easy_questions.json / hard_questions.json
└── README.md
```

---

## Visualización de Resultados

Los resultados se consolidan en `combined_data.csv` y otros archivos dentro de
`rag_analysis/`, que incluyen análisis como:

- Tendencias de evolución de métricas
- Comparación entre experimentos
- Correlación entre métricas
- Análisis de sensibilidad
- Gráficos individuales por métrica

---

## Modelo de Similitud

El archivo `rag_metrics/similarity.py` implementa la comparación semántica
usando `SentenceTransformers` y `cosine_similarity`.

```python
# Ejemplo de uso
model = RAGMetrics("bge-m3-es-legal-tmp-6")
score = model.context_precision(contexts, reference_answer)
```

---

## Cómo correr el proyecto

0. Si se desea correr los pipelines, se debe setear un archivo .env que contenga

   ```env
   OPENAI_API_KEY
   PINECONE_INDEX_NAME
   PINECONE_HOST
   PINECONE_API_KEY
   GEMINI_API_KEY
   K_RETRIEVE
   ```

   donde se debe setear la api key de OpenAI el nombre del indice en pinecone,
   la url del host, la api key de pinecone, la api key de gemini, y la cantidad
   de contextos que se quiere recuperar.

   > Recordar que esto es asi sólo si se quiere correr exactamente como lo
   > corrimos nosotros para experimentar, caso contrario se puede modificar para
   > cambiar de LLM o base de datos vectorial.

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Evaluar un experimento:

   ```bash
   python evaluate_custom.py <num-experimento>
   ```

3. Generar gráficos con los scripts de `rag_analysis/`.

---

## Resultados

Consideramos que los resultados de Ragas son de mejor calidad ya que hace
análisis más preciso y no como nuestra libreria de metricas que simplemente
compara un embedding con otro, sin embargo la evaluación con ragas es lenta y
costosa lo cual nos dio pie a generar nuestra propia implementación para testeos
rápidos.

### RAGAS

| experiment | context_precision | context_recall | faithfulness |
| :--------- | ----------------: | -------------: | -----------: |
| Exp 1      |          0.753803 |       0.768590 |     0.810095 |
| Exp 2      |          0.714566 |       0.778205 |     0.836070 |
| Exp 3      |          0.733206 |       0.737692 |     0.857552 |
| Exp 4      |          0.776501 |       0.796667 |     0.871826 |
| Exp 5      |          0.724312 |       0.807949 |     0.883329 |
| Exp 6      |          0.722281 |       0.817949 |     0.811880 |

## Custom metrics

| experiment | context_precision | context_recall | faithfulness | answer_relevance | answer_correctness |  rouge_l | retrieval_prec@5 (o @10 en 5 y 6) | retrieval_rec@5 (o @10 en 5 y 6) |
| :--------- | ----------------: | -------------: | -----------: | ---------------: | -----------------: | -------: | --------------------------------: | -------------------------------: |
| Exp 1      |          0.815495 |       0.825186 |     0.874991 |         0.892649 |           0.873165 | 0.178007 |                          0.706087 |                         0.860870 |
| Exp 2      |          0.792355 |       0.823878 |     0.874230 |         0.895383 |           0.874796 | 0.182626 |                          0.704348 |                         0.878261 |
| Exp 3      |          0.853333 |       0.824935 |     0.879692 |         0.894460 |           0.868526 | 0.180614 |                          0.775652 |                         0.921739 |
| Exp 4      |          0.854155 |       0.835007 |     0.879667 |         0.899572 |           0.872149 | 0.179218 |                          0.751304 |                         0.913043 |
| Exp 5      |          0.820641 |       0.831055 |     0.878715 |         0.899393 |           0.873809 | 0.178639 |                          0.721739 |                         0.956522 |
| Exp 6      |          0.820737 |       0.831252 |     0.877243 |         0.849247 |           0.895138 | 0.380906 |                          0.721739 |                         0.956522 |
