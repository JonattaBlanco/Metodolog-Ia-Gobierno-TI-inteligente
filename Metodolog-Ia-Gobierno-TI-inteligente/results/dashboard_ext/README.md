# Archivos Exportados para Dashboard (Power BI)

Este directorio contiene todos los archivos estructurados para crear el dashboard en Power BI.

## 📁 Estructura de Archivos

```
dashboard_ext/
├── predicciones/              # Archivos de predicciones futuras
│   ├── COBIT1_predicciones.csv
│   ├── COBIT1_historico_prediccion.csv
│   ├── ITIL2_predicciones.csv
│   ├── ITIL2_historico_prediccion.csv
│   ├── CMMI1_predicciones.csv
│   ├── CMMI1_historico_prediccion.csv
│   ├── CMMI2_predicciones.csv
│   ├── CMMI2_historico_prediccion.csv
│   ├── CRISP_DM2_predicciones.csv
│   └── CRISP_DM2_historico_prediccion.csv
│
├── clasificaciones/           # Archivos de clasificaciones de estados
│   ├── cobit2_estados.csv
│   ├── crisp_dm1_estados.csv
│   ├── scrum1_rendimiento.csv
│   └── scrum2_cumplimiento.csv
│
├── anomalias/                 # Archivos de anomalías detectadas
│   └── itil1_anomalias.csv
│
└── metricas/                  # Métricas consolidadas de todos los modelos
    └── metricas_modelos_consolidado.csv
```

## 📊 Descripción de Archivos

### Predicciones (`predicciones/`)

**Archivos `*_predicciones.csv`:**
- Contienen solo las predicciones futuras
- Columnas: `fecha`, `prediccion_*`, `kpi`, `tipo`
- Útiles para mostrar solo el período futuro

**Archivos `*_historico_prediccion.csv`:**
- Combinan datos históricos + predicciones futuras
- Columnas: `fecha`, `valor`, `kpi`, `tipo` (Historico/Prediccion)
- **RECOMENDADO:** Usar estos para gráficos de líneas que muestren histórico y futuro

### Clasificaciones (`clasificaciones/`)

Cada archivo contiene:
- `fecha`: Fecha del registro
- `estado` o `rendimiento`: Estado real
- `estado_predicho` o `rendimiento_predicho`: Estado predicho por el modelo
- `probabilidad`: Confianza de la predicción
- `kpi`: Nombre del KPI
- Columnas adicionales según el modelo (equipo, categoria_riesgo, etc.)

### Anomalías (`anomalias/`)

- `itil1_anomalias.csv`: Incidentes con tiempos de resolución anómalos
- Incluye información del servicio, fecha, tiempo promedio, probabilidad de anomalía

### Métricas (`metricas/`)

- `metricas_modelos_consolidado.csv`: Tabla con todas las métricas de evaluación
- Columnas: `modelo`, `kpi`, `tipo_modelo`, `mae`, `rmse`, `r2`, `accuracy`, `precision`, `recall`, `f1_score`

## 🎯 Cómo Usar en Power BI

### Paso 1: Importar Archivos

1. Abrir Power BI Desktop
2. **Obtener datos** → **Texto/CSV**
3. Importar todos los archivos de `predicciones/`, `clasificaciones/`, `anomalias/`, `metricas/`

### Paso 2: Configurar Tipos de Datos

- `fecha`: Cambiar a tipo **Fecha**
- `valor`, `prediccion_*`: Cambiar a tipo **Decimal**
- `probabilidad`: Cambiar a tipo **Decimal**
- `kpi`, `tipo`: Mantener como **Texto**

### Paso 3: Crear Relaciones

- Relacionar tablas por `fecha` cuando sea necesario
- Relacionar por `kpi` para unificar visualizaciones

### Paso 4: Crear Visualizaciones

**Página 1 - Resumen Ejecutivo:**
- Tarjetas KPI usando `metricas_modelos_consolidado.csv`
- Indicadores de estado con colores condicionales

**Página 2 - Predicciones:**
- Gráficos de líneas usando `*_historico_prediccion.csv`
- Filtrar por `tipo` para separar histórico de predicción
- Línea vertical en la fecha de corte

**Página 3 - Clasificaciones:**
- Tablas usando archivos de `clasificaciones/`
- Gráficos de barras por estado
- Filtros por fecha

**Página 4 - Anomalías:**
- Tabla de `itil1_anomalias.csv`
- Gráfico de dispersión tiempo vs probabilidad

**Página 5 - Métricas:**
- Tabla de `metricas_modelos_consolidado.csv`
- Gráficos comparativos de precisión

## 📝 Notas Importantes

- Los archivos se generan automáticamente al ejecutar `modelos_predictivos.py`
- Si ejecutas los modelos nuevamente, los archivos se sobrescriben
- Todos los archivos están en formato CSV UTF-8
- Las fechas están en formato ISO (YYYY-MM-DD)

## 🔄 Actualización de Datos

Para actualizar los archivos:
1. Ejecutar `python src/modelos_predictivos.py`
2. Los archivos se regeneran automáticamente en `results/dashboard_ext/`
3. En Power BI, usar **Actualizar** para recargar los datos

---

**Última actualización:** Generado automáticamente por `modelos_predictivos.py`
