# Estructura del Repositorio

## 📁 Organización de Directorios

```
Metodolog-Ia-Gobierno-TI-inteligente/
│
├── src/                                    # Scripts Python principales
│   ├── generar_datasets_gobierno_ti_inteligente_dian.py
│   ├── modelos_predictivos.py
│   └── eda_datasets_gobierno_ti_inteligente_dian.py
│
├── data/                                   # Datos del proyecto
│   └── datasets/
│       └── datasets_gobierno_ti_inteligente_dian/
│           ├── cobit_1_cumplimiento_plan_gobierno_ti.csv
│           ├── cobit_2_indice_riesgo_ti.csv
│           ├── itil_1_tiempo_resolucion_incidentes.csv
│           ├── itil_2_disponibilidad_servicios.csv
│           ├── cmmi_1_procesos_documentados.csv
│           ├── cmmi_2_mejoras_implementadas.csv
│           ├── crisp_dm_1_cobertura_datos_analizados.csv
│           ├── crisp_dm_2_tiempo_respuesta_analisis.csv
│           ├── scrum_1_velocidad_equipo.csv
│           ├── scrum_2_cumplimiento_retrospectivas.csv
│           └── resumen_datasets.csv
│
├── results/                                # Resultados generados
│   ├── modelos/                           # Resultados de modelos predictivos
│   │   ├── forecasting_cobit1.png
│   │   ├── forecasting_itil2.png
│   │   ├── forecasting_cmmi1.png
│   │   ├── forecasting_cmmi2.png
│   │   ├── forecasting_crisp_dm2.png
│   │   ├── deteccion_anomalias_itil1.png
│   │   ├── clasificacion_estados_cobit2.png
│   │   ├── clasificacion_crisp_dm1.png
│   │   ├── clasificacion_scrum1.png
│   │   ├── clasificacion_scrum2.png
│   │   └── predicciones_cobit1.csv
│   │
│   └── eda/                               # Resultados de análisis exploratorio
│       ├── graficos/
│       │   ├── cobit_1_analisis.png
│       │   ├── cobit_2_analisis.png
│       │   ├── itil_1_analisis.png
│       │   ├── itil_2_analisis.png
│       │   ├── cmmi_1_analisis.png
│       │   ├── cmmi_2_analisis.png
│       │   ├── crisp_dm_1_analisis.png
│       │   ├── crisp_dm_2_analisis.png
│       │   ├── scrum_1_analisis.png
│       │   └── scrum_2_analisis.png
│       └── resumen_eda.txt
│
├── requirements/                          # Dependencias del proyecto
│   └── requirements.txt
│
├── README.md                              # Documentación principal
├── ESTRUCTURA.md                          # Este archivo
└── .gitignore                             # Archivos a ignorar en Git
```

---

## 🔄 Flujo de Ejecución

### 1. Generación de Datasets
**Script:** `src/generar_datasets_gobierno_ti_inteligente_dian.py`  
**Entrada:** Ninguna (genera datos sintéticos)  
**Salida:** `data/datasets/datasets_gobierno_ti_inteligente_dian/*.csv`

### 2. Análisis Exploratorio
**Script:** `src/eda_datasets_gobierno_ti_inteligente_dian.py`  
**Entrada:** `data/datasets/datasets_gobierno_ti_inteligente_dian/*.csv`  
**Salida:** `results/eda/graficos/*.png` y `results/eda/resumen_eda.txt`

### 3. Modelos Predictivos
**Script:** `src/modelos_predictivos.py`  
**Entrada:** `data/datasets/datasets_gobierno_ti_inteligente_dian/*.csv`  
**Salida:** `results/modelos/*.png` y `results/modelos/predicciones_*.csv`

---

## 📊 Descripción de Componentes

### Scripts Python (`src/`)

- **generar_datasets_gobierno_ti_inteligente_dian.py**: Genera 10 datasets sintéticos con datos históricos de 5 años
- **eda_datasets_gobierno_ti_inteligente_dian.py**: Realiza análisis exploratorio completo de cada dataset
- **modelos_predictivos.py**: Entrena y evalúa 10 modelos de machine learning (forecasting, clasificación, detección de anomalías)

### Datos (`data/datasets/`)

- 10 archivos CSV con datos históricos de KPIs (2020-2024)
- Formato estructurado y listo para análisis
- Total aproximado: ~28,000 registros

### Resultados (`results/`)

- **modelos/**: Gráficos y predicciones de los 10 modelos predictivos
- **eda/**: Gráficos y resúmenes del análisis exploratorio

---

## 🚀 Próximos Pasos

1. ✅ Estructura del repositorio creada
2. ✅ Scripts organizados en `src/`
3. ✅ Datasets copiados a `data/`
4. ✅ Resultados organizados en `results/`
5. ✅ README y documentación creados
6. 🔨 **SIGUIENTE:** Crear dashboard con Streamlit
7. 🔨 **DESPUÉS:** Exportar resultados para Power BI

