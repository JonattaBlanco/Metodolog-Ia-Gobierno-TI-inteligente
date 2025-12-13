# Gobierno Inteligente de TI - DIAN

## 📋 Descripción del Proyecto

Este repositorio contiene el código fuente y los datasets utilizados para el desarrollo de la metodología de **Gobierno Inteligente de TI** aplicada a la Dirección de Impuestos y Aduanas Nacionales (DIAN) de Colombia.

El proyecto implementa un sistema de evaluación dinámico basado en inteligencia artificial para el monitoreo continuo y mejora permanente de la gestión tecnológica, utilizando indicadores clave de desempeño (KPIs) basados en marcos internacionales (COBIT, ITIL, CMMI, CRISP-DM, SCRUM).

---

## 🎯 Objetivos del Proyecto

- **Objetivo Específico 2:** Diseñar un modelo de evaluación dinámico basado en inteligencia artificial
- **Objetivo Específico 4:** Proponer un sistema de indicadores inteligentes, fundamentado en analítica de datos

---

## 📁 Estructura del Repositorio

```
Metodolog-Ia-Gobierno-TI-inteligente/
├── src/                          # Scripts Python principales
│   ├── generar_datasets_gobierno_ti_inteligente_dian.py
│   ├── modelos_predictivos.py
│   └── eda_datasets_gobierno_ti_inteligente_dian.py
│
├── data/                         # Datos del proyecto
│   └── datasets/                # Datasets generados (10 KPIs)
│       └── datasets_gobierno_ti_inteligente_dian/
│
├── results/                      # Resultados generados
│   ├── modelos/                 # Resultados de modelos predictivos
│   ├── eda/                     # Resultados de análisis exploratorio
│   └── dashboard_ext/           # Archivos exportados para Power BI
│
├── requirements/                 # Archivos de dependencias
│   └── requirements.txt
│
└── README.md                     # Este archivo
```

---

## 🚀 Guía de Inicio Rápido

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Clonar el repositorio** (cuando esté disponible):
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd Metodolog-Ia-Gobierno-TI-inteligente
   ```

2. **Crear entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements/requirements.txt
   ```

---

## 📊 Proceso de Ejecución

### Paso 1: Generación de Datasets

Genera los datasets sintéticos (dummy) para los 10 KPIs definidos en la metodología.

```bash
cd src
python generar_datasets_gobierno_ti_inteligente_dian.py
```

**Resultado:**
- Se crean 10 archivos CSV en `data/datasets/datasets_gobierno_ti_inteligente_dian/`
- Cada dataset contiene datos históricos de 5 años (2020-2024)
- Total aproximado: ~28,000 registros distribuidos en 10 datasets

**KPIs generados:**
1. COBIT 1: Cumplimiento del Plan de Gobierno de TI
2. COBIT 2: Índice de Riesgo de TI
3. ITIL 1: Tiempo de Resolución de Incidentes
4. ITIL 2: Disponibilidad de Servicios Críticos
5. CMMI 1: Procesos Documentados
6. CMMI 2: Mejoras Implementadas
7. CRISP-DM 1: Cobertura de Datos Analizados
8. CRISP-DM 2: Tiempo de Respuesta en Análisis
9. SCRUM 1: Velocidad del Equipo
10. SCRUM 2: Cumplimiento de Retrospectivas

---

### Paso 2: Análisis Exploratorio de Datos (EDA)

Realiza un análisis exploratorio completo de cada dataset, generando estadísticas descriptivas y visualizaciones.

```bash
cd src
python eda_datasets_gobierno_ti_inteligente_dian.py
```

**Resultado:**
- Se generan gráficos de análisis en `results/eda/graficos/`
- Se crea un resumen consolidado en `results/eda/resumen_eda.txt`
- 4 visualizaciones por cada KPI (40 gráficos en total)

---

### Paso 3: Modelos Predictivos con IA

Entrena y evalúa modelos de machine learning para cada KPI, incluyendo:
- **Forecasting** (predicción temporal): 5 modelos
- **Clasificación de estados**: 4 modelos
- **Detección de anomalías**: 1 modelo

```bash
cd src
python modelos_predictivos.py
```

**Resultado:**
- Se entrenan 10 modelos predictivos (uno por KPI)
- Se generan gráficos de predicciones en `results/modelos/`
- Se calculan métricas de evaluación (MAE, RMSE, R², Accuracy, Precision, Recall, F1-Score)
- Se guardan predicciones futuras en archivos CSV

**Modelos implementados:**

**Forecasting (5 modelos):**
- COBIT 1: Predicción de cumplimiento futuro
- ITIL 2: Predicción de disponibilidad de servicios
- CMMI 1: Predicción de porcentaje de documentación
- CMMI 2: Predicción de mejoras implementadas
- CRISP-DM 2: Predicción de tiempo de respuesta

**Clasificación (4 modelos):**
- COBIT 2: Clasificación de estados de riesgo (Excelente/Aceptable/Riesgo Alto)
- CRISP-DM 1: Clasificación de cobertura de datos
- SCRUM 1: Clasificación de rendimiento del equipo
- SCRUM 2: Clasificación de cumplimiento de retrospectivas

**Detección de Anomalías (1 modelo):**
- ITIL 1: Detección de tiempos de resolución anómalos

---

## 📈 Resultados Esperados

### Datasets Generados

- **10 archivos CSV** con datos históricos de KPIs
- Período histórico: 2020-2024 (5 años)
- Volumen total: ~28,000 registros
- Todos los datasets tienen mínimo 1,000 registros

### Modelos Predictivos

- **10 modelos entrenados** y funcionando
- **Precisión alta** en modelos de clasificación (99-100% accuracy)
- **Capacidad predictiva** demostrada en modelos de forecasting
- **Detección automática** de anomalías operativa

### Visualizaciones

- **40 gráficos de EDA** (4 por KPI)
- **10 gráficos de modelos** (1 por modelo predictivo)
- **Resúmenes consolidados** en formato texto

---

## 🔧 Configuración y Personalización

### Modificar Volumen de Datos

Editar las variables en `generar_datasets_gobierno_ti_inteligente_dian.py`:

```python
AÑO_INICIO = 2020
AÑO_FIN = 2024
NUM_SERVICIOS = 8
```

### Ajustar Parámetros de Modelos

Editar los parámetros en `modelos_predictivos.py`:

```python
# Ejemplo: Ajustar número de árboles en Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
```

---

## 📦 Dependencias

Las dependencias están especificadas en `requirements/requirements.txt`:

```
pandas>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

Para instalar todas las dependencias:

```bash
pip install -r requirements/requirements.txt
```

---

## 📝 Notas Importantes

### Datos Sintéticos

⚠️ **Importante:** Los datasets generados son **datos sintéticos (dummy)** creados para demostración y validación de la metodología. No representan datos reales de producción de la DIAN debido a restricciones de confidencialidad.

### Reproducibilidad

Todos los scripts utilizan semillas aleatorias fijas (`random_state=42`, `np.random.seed(42)`) para garantizar resultados reproducibles.

### Limitaciones

- Los modelos utilizan técnicas simples de machine learning (Random Forest)
- Los datos son sintéticos y no reflejan la complejidad real de producción
- El prototipo está diseñado para demostración, no para uso productivo directo

---

## 📊 Dashboard de Visualización

El proyecto incluye exportación automática de resultados para crear un dashboard ejecutivo en Power BI. Los archivos se generan automáticamente al ejecutar los modelos predictivos.

**Archivos exportados:** `results/dashboard_ext/` contiene 15 archivos CSV estructurados listos para importar en Power BI.

---

## 🎓 Uso Académico

Este código fue desarrollado como parte del Trabajo de Fin de Máster (TFM):

**Título:** Metodología para un Gobierno Inteligente de TI: Un enfoque de IA para Evaluación y Mejora Continua

**Institución:** [Nombre de la Universidad]

**Año:** 2025

---

## 📄 Licencia

Este proyecto es de uso académico y está destinado exclusivamente para fines educativos y de investigación.

---

## 👥 Autores

- [Geanina Juliana Mendoza Numa]
- [Jonattan Andrez Blanco Barón]

---

## 🔗 Referencias

- COBIT 2019 Framework
- ITIL 4 Foundation
- CMMI-DEV v2.0
- CRISP-DM Methodology
- SCRUM Guide

---

## 📞 Soporte

Para preguntas o problemas relacionados con el código, por favor abrir un issue en el repositorio o contactar a los autores.

---

**Última actualización:** Diciembre 2025

