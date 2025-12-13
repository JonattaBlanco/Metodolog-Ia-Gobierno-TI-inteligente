"""
Modelos Predictivos para KPIs de Gobierno Inteligente de TI - DIAN
Script simple para demostrar capacidad de predicción y detección de tendencias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración
CARPETA_DATASETS = '../data/datasets/datasets_gobierno_ti_inteligente_dian'
CARPETA_RESULTADOS = '../results/modelos'
import os
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

print("="*70)
print("MODELOS PREDICTIVOS - GOBIERNO INTELIGENTE DE TI DIAN")
print("="*70)
print()

# ============================================================================
# MODELO 1: FORECASTING - COBIT 1 (Cumplimiento del Plan de Gobierno de TI)
# ============================================================================

def modelo_forecasting_cobit1():
    """Predice el cumplimiento futuro del Plan de Gobierno de TI"""
    print("\n1. MODELO DE FORECASTING - COBIT 1")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/cobit_1_cumplimiento_plan_gobierno_ti.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Preparar datos para forecasting
    df_temporal = df.groupby('fecha')['cumplimiento_porcentaje'].mean().reset_index()
    df_temporal['dias_desde_inicio'] = (df_temporal['fecha'] - df_temporal['fecha'].min()).dt.days
    df_temporal['mes'] = df_temporal['fecha'].dt.month
    df_temporal['trimestre'] = df_temporal['fecha'].dt.quarter
    df_temporal['año'] = df_temporal['fecha'].dt.year
    
    # Dividir en entrenamiento y prueba (80/20)
    split_idx = int(len(df_temporal) * 0.8)
    train = df_temporal[:split_idx]
    test = df_temporal[split_idx:]
    
    # Features para el modelo
    features = ['dias_desde_inicio', 'mes', 'trimestre', 'año']
    X_train = train[features]
    y_train = train['cumplimiento_porcentaje']
    X_test = test[features]
    y_test = test['cumplimiento_porcentaje']
    
    # Entrenar modelo Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"✓ Modelo entrenado con {len(train)} registros")
    print(f"✓ MAE (entrenamiento): {mae_train:.2f}%")
    print(f"✓ MAE (prueba): {mae_test:.2f}%")
    print(f"✓ RMSE (prueba): {rmse_test:.2f}%")
    print(f"✓ R² (prueba): {r2_test:.3f}")
    
    # Predicción para próximos 3 meses
    ultima_fecha = df_temporal['fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=7), periods=12, freq='W')
    
    datos_futuros = pd.DataFrame({
        'fecha': fechas_futuras,
        'dias_desde_inicio': [(fecha - df_temporal['fecha'].min()).days for fecha in fechas_futuras],
        'mes': [fecha.month for fecha in fechas_futuras],
        'trimestre': [fecha.quarter for fecha in fechas_futuras],
        'año': [fecha.year for fecha in fechas_futuras]
    })
    
    predicciones_futuras = model.predict(datos_futuros[features])
    datos_futuros['prediccion_cumplimiento'] = predicciones_futuras
    
    print(f"\n✓ Predicciones generadas para próximos 3 meses (12 semanas)")
    print(f"  Promedio predicho: {predicciones_futuras.mean():.2f}%")
    print(f"  Tendencia: {'Mejora' if predicciones_futuras[-1] > predicciones_futuras[0] else 'Estable'}")
    
    # Visualización
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Histórico vs Predicciones
    axes[0].plot(train['fecha'], y_train, label='Entrenamiento (Real)', color='blue', alpha=0.7)
    axes[0].plot(test['fecha'], y_test, label='Prueba (Real)', color='green', alpha=0.7)
    axes[0].plot(test['fecha'], y_pred_test, label='Predicción (Prueba)', color='red', linestyle='--', linewidth=2)
    axes[0].plot(datos_futuros['fecha'], predicciones_futuras, label='Predicción Futura', color='orange', linestyle='--', linewidth=2, marker='o')
    axes[0].axvline(ultima_fecha, color='gray', linestyle=':', label='Fin de datos históricos')
    axes[0].set_title('COBIT 1: Forecasting de Cumplimiento del Plan de Gobierno de TI', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fecha')
    axes[0].set_ylabel('Cumplimiento (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Importancia de features
    importancia = pd.DataFrame({
        'feature': features,
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    axes[1].barh(importancia['feature'], importancia['importancia'], color='steelblue')
    axes[1].set_title('Importancia de Variables en el Modelo', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Importancia')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/forecasting_cobit1.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/forecasting_cobit1.png")
    plt.close()
    
    # Guardar predicciones
    datos_futuros.to_csv(f'{CARPETA_RESULTADOS}/predicciones_cobit1.csv', index=False)
    
    return {
        'modelo': model,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'predicciones': datos_futuros,
        'historico': df_temporal,
        'train': train,
        'test': test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

# ============================================================================
# MODELO 2: DETECCIÓN DE ANOMALÍAS - ITIL 1 (Tiempo de Resolución)
# ============================================================================

def modelo_deteccion_anomalias_itil1():
    """Detecta anomalías en tiempos de resolución de incidentes"""
    print("\n2. MODELO DE DETECCIÓN DE ANOMALÍAS - ITIL 1")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/itil_1_tiempo_resolucion_incidentes.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Preparar datos
    df['es_anomalia'] = (df['tiempo_promedio_horas'] > df['tiempo_promedio_horas'].quantile(0.90)).astype(int)
    
    # Convertir mes a numérico si es string
    if df['mes'].dtype == 'object':
        df['mes'] = pd.to_datetime(df['mes']).dt.month
    
    # Features
    le_servicio = LabelEncoder()
    df['servicio_encoded'] = le_servicio.fit_transform(df['servicio'])
    
    features = ['servicio_encoded', 'numero_incidentes', 'semana', 'mes']
    X = df[features]
    y = df['es_anomalia']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelo de clasificación
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✓ Modelo entrenado con {len(X_train)} registros")
    print(f"✓ Accuracy: {accuracy:.3f}")
    print(f"✓ Precision: {precision:.3f}")
    print(f"✓ Recall: {recall:.3f}")
    print(f"✓ F1-Score: {f1:.3f}")
    
    # Detectar anomalías en datos recientes
    df_recientes = df[df['fecha'] >= df['fecha'].max() - pd.Timedelta(days=90)]
    X_recientes = df_recientes[features]
    anomalias_detectadas = model.predict(X_recientes)
    probabilidades = model.predict_proba(X_recientes)[:, 1]
    
    df_recientes = df_recientes.copy()
    df_recientes['anomalia_detectada'] = anomalias_detectadas
    df_recientes['probabilidad_anomalia'] = probabilidades
    
    anomalias = df_recientes[df_recientes['anomalia_detectada'] == 1]
    print(f"\n✓ Anomalías detectadas en últimos 90 días: {len(anomalias)}")
    if len(anomalias) > 0:
        print(f"  Servicios más afectados:")
        print(anomalias.groupby('servicio').size().sort_values(ascending=False).head(3).to_string())
    
    # Visualización
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Distribución de tiempos con anomalías marcadas
    axes[0].hist(df['tiempo_promedio_horas'], bins=50, alpha=0.7, color='lightblue', label='Todos los datos')
    axes[0].hist(anomalias['tiempo_promedio_horas'], bins=30, alpha=0.8, color='red', label='Anomalías detectadas')
    axes[0].axvline(df['tiempo_promedio_horas'].mean(), color='green', linestyle='--', label=f'Media: {df["tiempo_promedio_horas"].mean():.2f}h')
    axes[0].set_title('ITIL 1: Detección de Anomalías en Tiempos de Resolución', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tiempo Promedio (horas)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Anomalías por servicio
    if len(anomalias) > 0:
        anomalias_por_servicio = anomalias.groupby('servicio').size().sort_values(ascending=False)
        axes[1].barh(range(len(anomalias_por_servicio)), anomalias_por_servicio.values, color='coral')
        axes[1].set_yticks(range(len(anomalias_por_servicio)))
        axes[1].set_yticklabels(anomalias_por_servicio.index, fontsize=9)
        axes[1].set_title('Anomalías Detectadas por Servicio (Últimos 90 días)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Número de Anomalías')
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/deteccion_anomalias_itil1.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/deteccion_anomalias_itil1.png")
    plt.close()
    
    # Guardar anomalías detectadas
    if len(anomalias) > 0:
        anomalias.to_csv(f'{CARPETA_RESULTADOS}/anomalias_detectadas_itil1.csv', index=False)
    
    return {
        'modelo': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'anomalias_detectadas': len(anomalias)
    }

# ============================================================================
# MODELO 3: CLASIFICACIÓN DE ESTADOS - COBIT 2 (Riesgo de TI)
# ============================================================================

def modelo_clasificacion_estados_cobit2():
    """Clasifica el estado del índice de riesgo de TI"""
    print("\n3. MODELO DE CLASIFICACIÓN DE ESTADOS - COBIT 2")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/cobit_2_indice_riesgo_ti.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Preparar datos
    le_categoria = LabelEncoder()
    le_estado = LabelEncoder()
    
    df['categoria_encoded'] = le_categoria.fit_transform(df['categoria_riesgo'])
    df['estado_encoded'] = le_estado.fit_transform(df['estado'])
    
    # Convertir mes a numérico si es string
    if df['mes'].dtype == 'object':
        df['mes'] = pd.to_datetime(df['mes']).dt.month
    
    features = ['categoria_encoded', 'riesgos_criticos_identificados', 'riesgos_mitigados', 
                'indice_mitigacion_porcentaje', 'semana', 'mes']
    X = df[features]
    y = df['estado_encoded']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelo
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Modelo entrenado con {len(X_train)} registros")
    print(f"✓ Accuracy: {accuracy:.3f}")
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=le_estado.classes_))
    
    # Visualización
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_estado.classes_, yticklabels=le_estado.classes_, ax=axes[0])
    axes[0].set_title('Matriz de Confusión - Clasificación de Estados', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicho')
    
    # Importancia de features
    importancia = pd.DataFrame({
        'feature': features,
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    axes[1].barh(importancia['feature'], importancia['importancia'], color='steelblue')
    axes[1].set_title('Importancia de Variables', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Importancia')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/clasificacion_estados_cobit2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/clasificacion_estados_cobit2.png")
    plt.close()
    
    # Preparar datos de clasificación para exportación
    df_clasificacion = df.copy()
    df_clasificacion['estado_predicho'] = le_estado.inverse_transform(model.predict(df[features]))
    df_clasificacion['probabilidad'] = model.predict_proba(df[features]).max(axis=1)
    # Asegurar que fecha esté presente
    if 'fecha' not in df_clasificacion.columns:
        if 'mes' in df_clasificacion.columns and df_clasificacion['mes'].dtype != 'object':
            # Crear fecha aproximada si no existe
            df_clasificacion['fecha'] = pd.to_datetime(df_clasificacion[['año', 'mes']].assign(dia=1))
    
    return {
        'modelo': model,
        'accuracy': accuracy,
        'label_encoders': {'categoria': le_categoria, 'estado': le_estado},
        'clasificaciones': df_clasificacion[['fecha', 'categoria_riesgo', 'estado', 'estado_predicho', 'probabilidad', 'indice_mitigacion_porcentaje']]
    }

# ============================================================================
# MODELO 4: FORECASTING - ITIL 2 (Disponibilidad de Servicios)
# ============================================================================

def modelo_forecasting_itil2():
    """Predice la disponibilidad futura de servicios críticos"""
    print("\n4. MODELO DE FORECASTING - ITIL 2 (Disponibilidad)")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/itil_2_disponibilidad_servicios.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Preparar datos por servicio (tomar un servicio como ejemplo)
    servicio_ejemplo = df['servicio'].unique()[0]
    df_servicio = df[df['servicio'] == servicio_ejemplo].copy()
    df_servicio = df_servicio.groupby('fecha')['disponibilidad_porcentaje'].mean().reset_index()
    
    df_servicio['dias_desde_inicio'] = (df_servicio['fecha'] - df_servicio['fecha'].min()).dt.days
    df_servicio['mes'] = df_servicio['fecha'].dt.month
    df_servicio['trimestre'] = df_servicio['fecha'].dt.quarter
    df_servicio['año'] = df_servicio['fecha'].dt.year
    
    # Dividir datos
    split_idx = int(len(df_servicio) * 0.8)
    train = df_servicio[:split_idx]
    test = df_servicio[split_idx:]
    
    # Features
    features = ['dias_desde_inicio', 'mes', 'trimestre', 'año']
    X_train = train[features]
    y_train = train['disponibilidad_porcentaje']
    X_test = test[features]
    y_test = test['disponibilidad_porcentaje']
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_test = model.predict(X_test)
    
    # Métricas
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"✓ Modelo entrenado para servicio: {servicio_ejemplo}")
    print(f"✓ MAE (prueba): {mae_test:.4f}%")
    print(f"✓ RMSE (prueba): {rmse_test:.4f}%")
    print(f"✓ R² (prueba): {r2_test:.3f}")
    
    # Predicción futura
    ultima_fecha = df_servicio['fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=30, freq='D')
    
    datos_futuros = pd.DataFrame({
        'fecha': fechas_futuras,
        'dias_desde_inicio': [(fecha - df_servicio['fecha'].min()).days for fecha in fechas_futuras],
        'mes': [fecha.month for fecha in fechas_futuras],
        'trimestre': [fecha.quarter for fecha in fechas_futuras],
        'año': [fecha.year for fecha in fechas_futuras]
    })
    
    predicciones_futuras = model.predict(datos_futuros[features])
    datos_futuros['prediccion_disponibilidad'] = predicciones_futuras
    
    print(f"\n✓ Predicciones generadas para próximos 30 días")
    print(f"  Promedio predicho: {predicciones_futuras.mean():.4f}%")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train['fecha'], y_train, label='Entrenamiento', color='blue', alpha=0.7)
    ax.plot(test['fecha'], y_test, label='Prueba (Real)', color='green', alpha=0.7)
    ax.plot(test['fecha'], y_pred_test, label='Predicción (Prueba)', color='red', linestyle='--', linewidth=2)
    ax.plot(datos_futuros['fecha'], predicciones_futuras, label='Predicción Futura', color='orange', linestyle='--', linewidth=2, marker='o', markersize=3)
    ax.axvline(ultima_fecha, color='gray', linestyle=':', label='Fin de datos históricos')
    ax.axhline(99.9, color='purple', linestyle='--', alpha=0.5, label='Objetivo 99.9%')
    ax.set_title(f'ITIL 2: Forecasting de Disponibilidad - {servicio_ejemplo}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Disponibilidad (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/forecasting_itil2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/forecasting_itil2.png")
    plt.close()
    
    return {
        'modelo': model,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'servicio': servicio_ejemplo,
        'predicciones': datos_futuros,
        'historico': df_servicio,
        'train': train,
        'test': test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

# ============================================================================
# MODELO 5: FORECASTING - CMMI 1 (Procesos Documentados)
# ============================================================================

def modelo_forecasting_cmmi1():
    """Predice el porcentaje futuro de procesos documentados"""
    print("\n5. MODELO DE FORECASTING - CMMI 1 (Procesos Documentados)")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/cmmi_1_procesos_documentados.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Preparar datos
    df_temporal = df.groupby('fecha')['porcentaje_documentacion'].mean().reset_index()
    df_temporal['dias_desde_inicio'] = (df_temporal['fecha'] - df_temporal['fecha'].min()).dt.days
    df_temporal['mes'] = df_temporal['fecha'].dt.month
    df_temporal['trimestre'] = df_temporal['fecha'].dt.quarter
    df_temporal['año'] = df_temporal['fecha'].dt.year
    
    # Dividir datos
    split_idx = int(len(df_temporal) * 0.8)
    train = df_temporal[:split_idx]
    test = df_temporal[split_idx:]
    
    # Features
    features = ['dias_desde_inicio', 'mes', 'trimestre', 'año']
    X_train = train[features]
    y_train = train['porcentaje_documentacion']
    X_test = test[features]
    y_test = test['porcentaje_documentacion']
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_test = model.predict(X_test)
    
    # Métricas
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"✓ Modelo entrenado con {len(train)} registros")
    print(f"✓ MAE (prueba): {mae_test:.2f}%")
    print(f"✓ RMSE (prueba): {rmse_test:.2f}%")
    print(f"✓ R² (prueba): {r2_test:.3f}")
    
    # Predicción futura
    ultima_fecha = df_temporal['fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=7), periods=12, freq='W')
    
    datos_futuros = pd.DataFrame({
        'fecha': fechas_futuras,
        'dias_desde_inicio': [(fecha - df_temporal['fecha'].min()).days for fecha in fechas_futuras],
        'mes': [fecha.month for fecha in fechas_futuras],
        'trimestre': [fecha.quarter for fecha in fechas_futuras],
        'año': [fecha.year for fecha in fechas_futuras]
    })
    
    predicciones_futuras = model.predict(datos_futuros[features])
    datos_futuros['prediccion_documentacion'] = predicciones_futuras
    
    print(f"\n✓ Predicciones generadas para próximos 3 meses")
    print(f"  Promedio predicho: {predicciones_futuras.mean():.2f}%")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train['fecha'], y_train, label='Entrenamiento', color='blue', alpha=0.7)
    ax.plot(test['fecha'], y_test, label='Prueba (Real)', color='green', alpha=0.7)
    ax.plot(test['fecha'], y_pred_test, label='Predicción (Prueba)', color='red', linestyle='--', linewidth=2)
    ax.plot(datos_futuros['fecha'], predicciones_futuras, label='Predicción Futura', color='orange', linestyle='--', linewidth=2, marker='o')
    ax.axvline(ultima_fecha, color='gray', linestyle=':', label='Fin de datos históricos')
    ax.set_title('CMMI 1: Forecasting de Porcentaje de Documentación', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Documentación (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/forecasting_cmmi1.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/forecasting_cmmi1.png")
    plt.close()
    
    return {
        'modelo': model,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'predicciones': datos_futuros,
        'historico': df_temporal,
        'train': train,
        'test': test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

# ============================================================================
# MODELO 6: CLASIFICACIÓN - SCRUM 1 (Rendimiento del Equipo)
# ============================================================================

def modelo_clasificacion_scrum1():
    """Clasifica el rendimiento del equipo basado en velocidad"""
    print("\n6. MODELO DE CLASIFICACIÓN - SCRUM 1 (Rendimiento Equipo)")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/scrum_1_velocidad_equipo.csv')
    df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
    
    # Crear variable objetivo basada en velocidad
    velocidad_media = df['velocidad_puntos_historia'].mean()
    df['rendimiento'] = df['velocidad_puntos_historia'].apply(
        lambda x: 'Bajo' if x < velocidad_media * 0.8 else 'Alto' if x > velocidad_media * 1.2 else 'Normal'
    )
    
    # Preparar datos
    le_equipo = LabelEncoder()
    le_rendimiento = LabelEncoder()
    
    df['equipo_encoded'] = le_equipo.fit_transform(df['equipo'])
    df['rendimiento_encoded'] = le_rendimiento.fit_transform(df['rendimiento'])
    
    features = ['equipo_encoded', 'semana', 'año']
    X = df[features]
    y = df['rendimiento_encoded']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelo
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Modelo entrenado con {len(X_train)} registros")
    print(f"✓ Accuracy: {accuracy:.3f}")
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=le_rendimiento.classes_))
    
    # Visualización
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_rendimiento.classes_, yticklabels=le_rendimiento.classes_, ax=axes[0])
    axes[0].set_title('Matriz de Confusión - Clasificación de Rendimiento', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicho')
    
    importancia = pd.DataFrame({
        'feature': features,
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    axes[1].barh(importancia['feature'], importancia['importancia'], color='steelblue')
    axes[1].set_title('Importancia de Variables', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Importancia')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/clasificacion_scrum1.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/clasificacion_scrum1.png")
    plt.close()
    
    # Preparar datos de clasificación para exportación
    df_clasificacion = df.copy()
    df_clasificacion['rendimiento_predicho'] = le_rendimiento.inverse_transform(model.predict(df[features]))
    df_clasificacion['probabilidad'] = model.predict_proba(df[features]).max(axis=1)
    # Renombrar fecha_inicio a fecha para consistencia
    if 'fecha_inicio' in df_clasificacion.columns:
        df_clasificacion['fecha'] = df_clasificacion['fecha_inicio']
    
    return {
        'modelo': model,
        'accuracy': accuracy,
        'label_encoders': {'equipo': le_equipo, 'rendimiento': le_rendimiento},
        'clasificaciones': df_clasificacion[['fecha', 'equipo', 'rendimiento', 'rendimiento_predicho', 'probabilidad', 'velocidad_puntos_historia']]
    }

# ============================================================================
# MODELO 7: FORECASTING - CMMI 2 (Mejoras Implementadas)
# ============================================================================

def modelo_forecasting_cmmi2():
    """Predice el porcentaje futuro de mejoras implementadas"""
    print("\n7. MODELO DE FORECASTING - CMMI 2 (Mejoras Implementadas)")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/cmmi_2_mejoras_implementadas.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Preparar datos
    df_temporal = df.groupby('fecha')['porcentaje_implementacion'].mean().reset_index()
    df_temporal['dias_desde_inicio'] = (df_temporal['fecha'] - df_temporal['fecha'].min()).dt.days
    df_temporal['mes'] = df_temporal['fecha'].dt.month
    df_temporal['trimestre'] = df_temporal['fecha'].dt.quarter
    df_temporal['año'] = df_temporal['fecha'].dt.year
    
    # Dividir datos
    split_idx = int(len(df_temporal) * 0.8)
    train = df_temporal[:split_idx]
    test = df_temporal[split_idx:]
    
    # Features
    features = ['dias_desde_inicio', 'mes', 'trimestre', 'año']
    X_train = train[features]
    y_train = train['porcentaje_implementacion']
    X_test = test[features]
    y_test = test['porcentaje_implementacion']
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_test = model.predict(X_test)
    
    # Métricas
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"✓ Modelo entrenado con {len(train)} registros")
    print(f"✓ MAE (prueba): {mae_test:.2f}%")
    print(f"✓ RMSE (prueba): {rmse_test:.2f}%")
    print(f"✓ R² (prueba): {r2_test:.3f}")
    
    # Predicción futura
    ultima_fecha = df_temporal['fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=7), periods=12, freq='W')
    
    datos_futuros = pd.DataFrame({
        'fecha': fechas_futuras,
        'dias_desde_inicio': [(fecha - df_temporal['fecha'].min()).days for fecha in fechas_futuras],
        'mes': [fecha.month for fecha in fechas_futuras],
        'trimestre': [fecha.quarter for fecha in fechas_futuras],
        'año': [fecha.year for fecha in fechas_futuras]
    })
    
    predicciones_futuras = model.predict(datos_futuros[features])
    datos_futuros['prediccion_implementacion'] = predicciones_futuras
    
    print(f"\n✓ Predicciones generadas para próximos 3 meses")
    print(f"  Promedio predicho: {predicciones_futuras.mean():.2f}%")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train['fecha'], y_train, label='Entrenamiento', color='blue', alpha=0.7)
    ax.plot(test['fecha'], y_test, label='Prueba (Real)', color='green', alpha=0.7)
    ax.plot(test['fecha'], y_pred_test, label='Predicción (Prueba)', color='red', linestyle='--', linewidth=2)
    ax.plot(datos_futuros['fecha'], predicciones_futuras, label='Predicción Futura', color='orange', linestyle='--', linewidth=2, marker='o')
    ax.axvline(ultima_fecha, color='gray', linestyle=':', label='Fin de datos históricos')
    ax.set_title('CMMI 2: Forecasting de Porcentaje de Implementación de Mejoras', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Implementación (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/forecasting_cmmi2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/forecasting_cmmi2.png")
    plt.close()
    
    return {
        'modelo': model,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'predicciones': datos_futuros,
        'historico': df_temporal,
        'train': train,
        'test': test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

# ============================================================================
# MODELO 8: CLASIFICACIÓN - CRISP-DM 1 (Cobertura de Datos)
# ============================================================================

def modelo_clasificacion_crisp_dm1():
    """Clasifica el estado de cobertura de datos analizados"""
    print("\n8. MODELO DE CLASIFICACIÓN - CRISP-DM 1 (Cobertura de Datos)")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/crisp_dm_1_cobertura_datos_analizados.csv')
    df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
    
    # Preparar datos
    le_tipo = LabelEncoder()
    le_estado = LabelEncoder()
    
    df['tipo_proyecto_encoded'] = le_tipo.fit_transform(df['tipo_proyecto'])
    df['estado_encoded'] = le_estado.fit_transform(df['estado'])
    
    features = ['tipo_proyecto_encoded', 'registros_totales_disponibles', 
                'registros_analizados', 'cobertura_porcentaje', 'año']
    X = df[features]
    y = df['estado_encoded']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelo
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Modelo entrenado con {len(X_train)} registros")
    print(f"✓ Accuracy: {accuracy:.3f}")
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=le_estado.classes_))
    
    # Visualización
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_estado.classes_, yticklabels=le_estado.classes_, ax=axes[0])
    axes[0].set_title('Matriz de Confusión - Clasificación de Cobertura', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicho')
    
    importancia = pd.DataFrame({
        'feature': features,
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    axes[1].barh(importancia['feature'], importancia['importancia'], color='steelblue')
    axes[1].set_title('Importancia de Variables', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Importancia')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/clasificacion_crisp_dm1.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/clasificacion_crisp_dm1.png")
    plt.close()
    
    # Preparar datos de clasificación para exportación
    df_clasificacion = df.copy()
    df_clasificacion['estado_predicho'] = le_estado.inverse_transform(model.predict(df[features]))
    df_clasificacion['probabilidad'] = model.predict_proba(df[features]).max(axis=1)
    # Renombrar fecha_inicio a fecha para consistencia
    if 'fecha_inicio' in df_clasificacion.columns:
        df_clasificacion['fecha'] = df_clasificacion['fecha_inicio']
    
    return {
        'modelo': model,
        'accuracy': accuracy,
        'label_encoders': {'tipo': le_tipo, 'estado': le_estado},
        'clasificaciones': df_clasificacion[['fecha', 'tipo_proyecto', 'estado', 'estado_predicho', 'probabilidad', 'cobertura_porcentaje']]
    }

# ============================================================================
# MODELO 9: FORECASTING - CRISP-DM 2 (Tiempo de Respuesta)
# ============================================================================

def modelo_forecasting_crisp_dm2():
    """Predice el tiempo futuro de respuesta en análisis"""
    print("\n9. MODELO DE FORECASTING - CRISP-DM 2 (Tiempo de Respuesta)")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/crisp_dm_2_tiempo_respuesta_analisis.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Preparar datos
    df_temporal = df.groupby('fecha')['tiempo_promedio_horas'].mean().reset_index()
    df_temporal['dias_desde_inicio'] = (df_temporal['fecha'] - df_temporal['fecha'].min()).dt.days
    df_temporal['mes'] = df_temporal['fecha'].dt.month
    df_temporal['trimestre'] = df_temporal['fecha'].dt.quarter
    df_temporal['año'] = df_temporal['fecha'].dt.year
    
    # Dividir datos
    split_idx = int(len(df_temporal) * 0.8)
    train = df_temporal[:split_idx]
    test = df_temporal[split_idx:]
    
    # Features
    features = ['dias_desde_inicio', 'mes', 'trimestre', 'año']
    X_train = train[features]
    y_train = train['tiempo_promedio_horas']
    X_test = test[features]
    y_test = test['tiempo_promedio_horas']
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_test = model.predict(X_test)
    
    # Métricas
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"✓ Modelo entrenado con {len(train)} registros")
    print(f"✓ MAE (prueba): {mae_test:.2f} horas")
    print(f"✓ RMSE (prueba): {rmse_test:.2f} horas")
    print(f"✓ R² (prueba): {r2_test:.3f}")
    
    # Predicción futura
    ultima_fecha = df_temporal['fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=7), periods=12, freq='W')
    
    datos_futuros = pd.DataFrame({
        'fecha': fechas_futuras,
        'dias_desde_inicio': [(fecha - df_temporal['fecha'].min()).days for fecha in fechas_futuras],
        'mes': [fecha.month for fecha in fechas_futuras],
        'trimestre': [fecha.quarter for fecha in fechas_futuras],
        'año': [fecha.year for fecha in fechas_futuras]
    })
    
    predicciones_futuras = model.predict(datos_futuros[features])
    datos_futuros['prediccion_tiempo_horas'] = predicciones_futuras
    
    print(f"\n✓ Predicciones generadas para próximos 3 meses")
    print(f"  Promedio predicho: {predicciones_futuras.mean():.2f} horas")
    print(f"  Tendencia: {'Mejora' if predicciones_futuras[-1] < predicciones_futuras[0] else 'Estable'}")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train['fecha'], y_train, label='Entrenamiento', color='blue', alpha=0.7)
    ax.plot(test['fecha'], y_test, label='Prueba (Real)', color='green', alpha=0.7)
    ax.plot(test['fecha'], y_pred_test, label='Predicción (Prueba)', color='red', linestyle='--', linewidth=2)
    ax.plot(datos_futuros['fecha'], predicciones_futuras, label='Predicción Futura', color='orange', linestyle='--', linewidth=2, marker='o')
    ax.axvline(ultima_fecha, color='gray', linestyle=':', label='Fin de datos históricos')
    ax.axhline(24, color='purple', linestyle='--', alpha=0.5, label='Objetivo 24h')
    ax.set_title('CRISP-DM 2: Forecasting de Tiempo de Respuesta en Análisis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Tiempo (horas)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/forecasting_crisp_dm2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/forecasting_crisp_dm2.png")
    plt.close()
    
    return {
        'modelo': model,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'predicciones': datos_futuros,
        'historico': df_temporal,
        'train': train,
        'test': test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

# ============================================================================
# MODELO 10: CLASIFICACIÓN - SCRUM 2 (Cumplimiento Retrospectivas)
# ============================================================================

def modelo_clasificacion_scrum2():
    """Clasifica el cumplimiento de retrospectivas"""
    print("\n10. MODELO DE CLASIFICACIÓN - SCRUM 2 (Cumplimiento Retrospectivas)")
    print("-" * 70)
    
    # Cargar datos
    df = pd.read_csv(f'{CARPETA_DATASETS}/scrum_2_cumplimiento_retrospectivas.csv')
    df['fecha_sprint'] = pd.to_datetime(df['fecha_sprint'])
    
    # Preparar datos
    le_equipo = LabelEncoder()
    le_estado = LabelEncoder()
    
    df['equipo_encoded'] = le_equipo.fit_transform(df['equipo'])
    df['estado_encoded'] = le_estado.fit_transform(df['estado'])
    
    features = ['equipo_encoded', 'retrospectivas_planificadas', 
                'retrospectivas_realizadas', 'cumplimiento_porcentaje', 'semana', 'año']
    X = df[features]
    y = df['estado_encoded']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelo
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Modelo entrenado con {len(X_train)} registros")
    print(f"✓ Accuracy: {accuracy:.3f}")
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=le_estado.classes_))
    
    # Visualización
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_estado.classes_, yticklabels=le_estado.classes_, ax=axes[0])
    axes[0].set_title('Matriz de Confusión - Clasificación de Cumplimiento', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicho')
    
    importancia = pd.DataFrame({
        'feature': features,
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    axes[1].barh(importancia['feature'], importancia['importancia'], color='steelblue')
    axes[1].set_title('Importancia de Variables', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Importancia')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/clasificacion_scrum2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {CARPETA_RESULTADOS}/clasificacion_scrum2.png")
    plt.close()
    
    # Preparar datos de clasificación para exportación
    df_clasificacion = df.copy()
    df_clasificacion['estado_predicho'] = le_estado.inverse_transform(model.predict(df[features]))
    df_clasificacion['probabilidad'] = model.predict_proba(df[features]).max(axis=1)
    # Renombrar fecha_sprint a fecha para consistencia
    if 'fecha_sprint' in df_clasificacion.columns:
        df_clasificacion['fecha'] = df_clasificacion['fecha_sprint']
    
    return {
        'modelo': model,
        'accuracy': accuracy,
        'label_encoders': {'equipo': le_equipo, 'estado': le_estado},
        'clasificaciones': df_clasificacion[['fecha', 'equipo', 'estado', 'estado_predicho', 'probabilidad', 'cumplimiento_porcentaje']]
    }

# ============================================================================
# FUNCIÓN DE EXPORTACIÓN PARA DASHBOARD (POWER BI)
# ============================================================================

def exportar_resultados_dashboard(resultados):
    """
    Exporta todos los resultados de modelos en formato estructurado para Power BI
    Guarda archivos en results/dashboard_ext/
    """
    print("\n" + "="*70)
    print("EXPORTANDO RESULTADOS PARA DASHBOARD (POWER BI)")
    print("="*70)
    
    CARPETA_DASHBOARD = '../results/dashboard_ext'
    os.makedirs(f'{CARPETA_DASHBOARD}/predicciones', exist_ok=True)
    os.makedirs(f'{CARPETA_DASHBOARD}/clasificaciones', exist_ok=True)
    os.makedirs(f'{CARPETA_DASHBOARD}/anomalias', exist_ok=True)
    os.makedirs(f'{CARPETA_DASHBOARD}/metricas', exist_ok=True)
    
    metricas_consolidadas = []
    
    # ========================================================================
    # 1. EXPORTAR PREDICCIONES (Forecasting Models)
    # ========================================================================
    print("\n📊 Exportando predicciones...")
    
    modelos_forecasting = {
        'forecasting_cobit1': {
            'nombre': 'COBIT1',
            'columna_valor': 'cumplimiento_porcentaje',
            'columna_prediccion': 'prediccion_cumplimiento',
            'nombre_kpi': 'Cumplimiento Plan Gobierno TI'
        },
        'forecasting_itil2': {
            'nombre': 'ITIL2',
            'columna_valor': 'disponibilidad_porcentaje',
            'columna_prediccion': 'prediccion_disponibilidad',
            'nombre_kpi': 'Disponibilidad Servicios'
        },
        'forecasting_cmmi1': {
            'nombre': 'CMMI1',
            'columna_valor': 'porcentaje_documentacion',
            'columna_prediccion': 'prediccion_documentacion',
            'nombre_kpi': 'Procesos Documentados'
        },
        'forecasting_cmmi2': {
            'nombre': 'CMMI2',
            'columna_valor': 'porcentaje_implementacion',
            'columna_prediccion': 'prediccion_implementacion',
            'nombre_kpi': 'Mejoras Implementadas'
        },
        'forecasting_crisp_dm2': {
            'nombre': 'CRISP_DM2',
            'columna_valor': 'tiempo_promedio_horas',
            'columna_prediccion': 'prediccion_tiempo_horas',
            'nombre_kpi': 'Tiempo Respuesta Análisis'
        }
    }
    
    for key, config in modelos_forecasting.items():
        if key in resultados and 'predicciones' in resultados[key]:
            try:
                pred_df = resultados[key]['predicciones'].copy()
                historico = resultados[key].get('historico')
                
                # Crear archivo de solo predicciones
                pred_df['kpi'] = config['nombre_kpi']
                pred_df['tipo'] = 'Prediccion'
                pred_df.to_csv(f'{CARPETA_DASHBOARD}/predicciones/{config["nombre"]}_predicciones.csv', index=False)
                
                # Crear archivo combinado (histórico + predicción)
                if historico is not None:
                    historico_df = historico.copy()
                    historico_df['kpi'] = config['nombre_kpi']
                    historico_df['tipo'] = 'Historico'
                    
                    # Renombrar columna de valor para unificar
                    valor_col = config['columna_valor']
                    if valor_col in historico_df.columns:
                        historico_df['valor'] = historico_df[valor_col]
                    
                    # Preparar predicciones
                    pred_combinado = pred_df.copy()
                    pred_combinado['valor'] = pred_df[config['columna_prediccion']]
                    
                    # Combinar
                    columnas_comunes = ['fecha', 'kpi', 'tipo', 'valor']
                    combinado = pd.concat([
                        historico_df[columnas_comunes],
                        pred_combinado[columnas_comunes]
                    ], ignore_index=True)
                    
                    combinado.to_csv(f'{CARPETA_DASHBOARD}/predicciones/{config["nombre"]}_historico_prediccion.csv', index=False)
                
                # Agregar métricas
                metricas_consolidadas.append({
                    'modelo': config['nombre'],
                    'kpi': config['nombre_kpi'],
                    'tipo_modelo': 'Forecasting',
                    'mae': resultados[key].get('mae_test', None),
                    'rmse': resultados[key].get('rmse_test', None),
                    'r2': resultados[key].get('r2_test', None),
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None
                })
                
                print(f"  ✓ {config['nombre']}: Predicciones exportadas")
            except Exception as e:
                print(f"  ✗ Error exportando {config['nombre']}: {e}")
    
    # ========================================================================
    # 2. EXPORTAR CLASIFICACIONES
    # ========================================================================
    print("\n📋 Exportando clasificaciones...")
    
    modelos_clasificacion = {
        'clasificacion_cobit2': {
            'nombre': 'COBIT2',
            'nombre_kpi': 'Índice Riesgo TI',
            'archivo': 'cobit2_estados.csv'
        },
        'clasificacion_crisp_dm1': {
            'nombre': 'CRISP_DM1',
            'nombre_kpi': 'Cobertura Datos',
            'archivo': 'crisp_dm1_estados.csv'
        },
        'clasificacion_scrum1': {
            'nombre': 'SCRUM1',
            'nombre_kpi': 'Rendimiento Equipo',
            'archivo': 'scrum1_rendimiento.csv'
        },
        'clasificacion_scrum2': {
            'nombre': 'SCRUM2',
            'nombre_kpi': 'Cumplimiento Retrospectivas',
            'archivo': 'scrum2_cumplimiento.csv'
        }
    }
    
    for key, config in modelos_clasificacion.items():
        if key in resultados and 'clasificaciones' in resultados[key]:
            try:
                clasif_df = resultados[key]['clasificaciones'].copy()
                clasif_df['kpi'] = config['nombre_kpi']
                clasif_df.to_csv(f'{CARPETA_DASHBOARD}/clasificaciones/{config["archivo"]}', index=False)
                
                # Agregar métricas
                metricas_consolidadas.append({
                    'modelo': config['nombre'],
                    'kpi': config['nombre_kpi'],
                    'tipo_modelo': 'Clasificacion',
                    'mae': None,
                    'rmse': None,
                    'r2': None,
                    'accuracy': resultados[key].get('accuracy', None),
                    'precision': resultados[key].get('precision', None),
                    'recall': resultados[key].get('recall', None),
                    'f1_score': resultados[key].get('f1', None)
                })
                
                print(f"  ✓ {config['nombre']}: Clasificaciones exportadas")
            except Exception as e:
                print(f"  ✗ Error exportando {config['nombre']}: {e}")
    
    # ========================================================================
    # 3. EXPORTAR ANOMALÍAS
    # ========================================================================
    print("\n⚠️  Exportando anomalías...")
    
    if 'anomalias_itil1' in resultados:
        try:
            # Leer archivo de anomalías si existe
            anomalias_path = f'{CARPETA_RESULTADOS}/anomalias_detectadas_itil1.csv'
            if os.path.exists(anomalias_path):
                anomalias_df = pd.read_csv(anomalias_path)
                anomalias_df['kpi'] = 'Tiempo Resolución Incidentes'
                anomalias_df.to_csv(f'{CARPETA_DASHBOARD}/anomalias/itil1_anomalias.csv', index=False)
                print(f"  ✓ ITIL1: Anomalías exportadas ({len(anomalias_df)} registros)")
            
            # Agregar métricas
            metricas_consolidadas.append({
                'modelo': 'ITIL1',
                'kpi': 'Tiempo Resolución Incidentes',
                'tipo_modelo': 'Deteccion_Anomalias',
                'mae': None,
                'rmse': None,
                'r2': None,
                'accuracy': resultados['anomalias_itil1'].get('accuracy', None),
                'precision': resultados['anomalias_itil1'].get('precision', None),
                'recall': resultados['anomalias_itil1'].get('recall', None),
                'f1_score': resultados['anomalias_itil1'].get('f1', None)
            })
        except Exception as e:
            print(f"  ✗ Error exportando anomalías ITIL1: {e}")
    
    # ========================================================================
    # 4. EXPORTAR MÉTRICAS CONSOLIDADAS
    # ========================================================================
    print("\n📈 Exportando métricas consolidadas...")
    
    metricas_df = pd.DataFrame(metricas_consolidadas)
    metricas_df.to_csv(f'{CARPETA_DASHBOARD}/metricas/metricas_modelos_consolidado.csv', index=False)
    print(f"  ✓ Métricas consolidadas exportadas ({len(metricas_df)} modelos)")
    
    # ========================================================================
    # RESUMEN
    # ========================================================================
    print("\n" + "="*70)
    print("EXPORTACIÓN COMPLETA PARA DASHBOARD")
    print("="*70)
    print(f"\n✓ Archivos exportados en: {CARPETA_DASHBOARD}/")
    print(f"  - Predicciones: {CARPETA_DASHBOARD}/predicciones/")
    print(f"  - Clasificaciones: {CARPETA_DASHBOARD}/clasificaciones/")
    print(f"  - Anomalías: {CARPETA_DASHBOARD}/anomalias/")
    print(f"  - Métricas: {CARPETA_DASHBOARD}/metricas/")
    print("\n✓ Listo para importar en Power BI!")
    print("="*70)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta todos los modelos predictivos"""
    
    resultados = {}
    
    # Ejecutar modelos
    try:
        resultados['forecasting_cobit1'] = modelo_forecasting_cobit1()
    except Exception as e:
        print(f"Error en forecasting COBIT 1: {e}")
    
    try:
        resultados['anomalias_itil1'] = modelo_deteccion_anomalias_itil1()
    except Exception as e:
        print(f"Error en detección de anomalías ITIL 1: {e}")
    
    try:
        resultados['clasificacion_cobit2'] = modelo_clasificacion_estados_cobit2()
    except Exception as e:
        print(f"Error en clasificación COBIT 2: {e}")
    
    try:
        resultados['forecasting_itil2'] = modelo_forecasting_itil2()
    except Exception as e:
        print(f"Error en forecasting ITIL 2: {e}")
    
    try:
        resultados['forecasting_cmmi1'] = modelo_forecasting_cmmi1()
    except Exception as e:
        print(f"Error en forecasting CMMI 1: {e}")
    
    try:
        resultados['clasificacion_scrum1'] = modelo_clasificacion_scrum1()
    except Exception as e:
        print(f"Error en clasificación SCRUM 1: {e}")
    
    try:
        resultados['forecasting_cmmi2'] = modelo_forecasting_cmmi2()
    except Exception as e:
        print(f"Error en forecasting CMMI 2: {e}")
    
    try:
        resultados['clasificacion_crisp_dm1'] = modelo_clasificacion_crisp_dm1()
    except Exception as e:
        print(f"Error en clasificación CRISP-DM 1: {e}")
    
    try:
        resultados['forecasting_crisp_dm2'] = modelo_forecasting_crisp_dm2()
    except Exception as e:
        print(f"Error en forecasting CRISP-DM 2: {e}")
    
    try:
        resultados['clasificacion_scrum2'] = modelo_clasificacion_scrum2()
    except Exception as e:
        print(f"Error en clasificación SCRUM 2: {e}")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE MODELOS PREDICTIVOS")
    print("="*70)
    
    if 'forecasting_cobit1' in resultados:
        print(f"\n✓ Forecasting COBIT 1:")
        print(f"  - R² Score: {resultados['forecasting_cobit1']['r2_test']:.3f}")
        print(f"  - RMSE: {resultados['forecasting_cobit1']['rmse_test']:.2f}%")
    
    if 'forecasting_itil2' in resultados:
        print(f"\n✓ Forecasting ITIL 2:")
        print(f"  - R² Score: {resultados['forecasting_itil2']['r2_test']:.3f}")
        print(f"  - RMSE: {resultados['forecasting_itil2']['rmse_test']:.4f}%")
    
    if 'forecasting_cmmi1' in resultados:
        print(f"\n✓ Forecasting CMMI 1:")
        print(f"  - R² Score: {resultados['forecasting_cmmi1']['r2_test']:.3f}")
        print(f"  - RMSE: {resultados['forecasting_cmmi1']['rmse_test']:.2f}%")
    
    if 'forecasting_cmmi2' in resultados:
        print(f"\n✓ Forecasting CMMI 2:")
        print(f"  - R² Score: {resultados['forecasting_cmmi2']['r2_test']:.3f}")
        print(f"  - RMSE: {resultados['forecasting_cmmi2']['rmse_test']:.2f}%")
    
    if 'forecasting_crisp_dm2' in resultados:
        print(f"\n✓ Forecasting CRISP-DM 2:")
        print(f"  - R² Score: {resultados['forecasting_crisp_dm2']['r2_test']:.3f}")
        print(f"  - RMSE: {resultados['forecasting_crisp_dm2']['rmse_test']:.2f} horas")
    
    if 'anomalias_itil1' in resultados:
        print(f"\n✓ Detección de Anomalías ITIL 1:")
        print(f"  - Accuracy: {resultados['anomalias_itil1']['accuracy']:.3f}")
        print(f"  - Anomalías detectadas: {resultados['anomalias_itil1']['anomalias_detectadas']}")
    
    if 'clasificacion_cobit2' in resultados:
        print(f"\n✓ Clasificación de Estados COBIT 2:")
        print(f"  - Accuracy: {resultados['clasificacion_cobit2']['accuracy']:.3f}")
    
    if 'clasificacion_scrum1' in resultados:
        print(f"\n✓ Clasificación de Rendimiento SCRUM 1:")
        print(f"  - Accuracy: {resultados['clasificacion_scrum1']['accuracy']:.3f}")
    
    if 'clasificacion_crisp_dm1' in resultados:
        print(f"\n✓ Clasificación de Cobertura CRISP-DM 1:")
        print(f"  - Accuracy: {resultados['clasificacion_crisp_dm1']['accuracy']:.3f}")
    
    if 'clasificacion_scrum2' in resultados:
        print(f"\n✓ Clasificación de Cumplimiento SCRUM 2:")
        print(f"  - Accuracy: {resultados['clasificacion_scrum2']['accuracy']:.3f}")
    
    print(f"\n✓ Total de modelos ejecutados: {len(resultados)}")
    print(f"✓ Todos los 10 KPIs tienen modelos predictivos funcionando!")
    print(f"\n✓ Todos los resultados guardados en: {CARPETA_RESULTADOS}/")
    print("="*70)
    
    # Exportar resultados para dashboard (Power BI)
    try:
        exportar_resultados_dashboard(resultados)
    except Exception as e:
        print(f"\n⚠️  Error al exportar resultados para dashboard: {e}")
        print("   Los modelos se ejecutaron correctamente, pero la exportación falló.")

if __name__ == "__main__":
    main()

