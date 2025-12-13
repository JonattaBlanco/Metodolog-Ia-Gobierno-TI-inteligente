"""
Script de Análisis Exploratorio de Datos (EDA) para los datasets de KPIs
Gobierno Inteligente de TI - DIAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Configuración
CARPETA_DATASETS = '../data/datasets/datasets_gobierno_ti_inteligente_dian'
CARPETA_RESULTADOS = '../results/eda'
AÑO_INICIO = 2020
AÑO_FIN = 2024

# Crear carpeta de resultados
Path(CARPETA_RESULTADOS).mkdir(exist_ok=True)
Path(f'{CARPETA_RESULTADOS}/graficos').mkdir(exist_ok=True)

def analizar_dataset(nombre_archivo, nombre_kpi):
    """Realiza EDA completo para un dataset"""
    print(f"\n{'='*70}")
    print(f"ANALIZANDO: {nombre_kpi}")
    print(f"{'='*70}")
    
    # Cargar dataset
    df = pd.read_csv(f'{CARPETA_DATASETS}/{nombre_archivo}')
    
    print(f"\n1. INFORMACIÓN GENERAL")
    print(f"   - Registros: {len(df):,}")
    print(f"   - Columnas: {len(df.columns)}")
    print(f"   - Memoria: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print(f"\n2. ESTRUCTURA DEL DATASET")
    print(f"   Columnas: {', '.join(df.columns.tolist())}")
    
    print(f"\n3. TIPOS DE DATOS")
    print(df.dtypes.to_string())
    
    print(f"\n4. VALORES FALTANTES")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0].to_string())
    else:
        print("   ✓ No hay valores faltantes")
    
    print(f"\n5. ESTADÍSTICAS DESCRIPTIVAS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())
    
    print(f"\n6. VALORES ÚNICOS EN COLUMNAS CATEGÓRICAS")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"   - {col}: {unique_count} valores únicos")
        if unique_count <= 20:
            print(f"     Valores: {', '.join(df[col].unique().astype(str)[:10])}")
    
    # Análisis temporal si existe columna de fecha
    fecha_cols = [col for col in df.columns if 'fecha' in col.lower() or 'date' in col.lower()]
    if fecha_cols:
        print(f"\n7. ANÁLISIS TEMPORAL")
        fecha_col = fecha_cols[0]
        df[fecha_col] = pd.to_datetime(df[fecha_col])
        print(f"   - Período: {df[fecha_col].min()} a {df[fecha_col].max()}")
        print(f"   - Duración: {(df[fecha_col].max() - df[fecha_col].min()).days} días")
        
        # Distribución por año
        if 'año' in df.columns:
            print(f"\n   Distribución por año:")
            print(df['año'].value_counts().sort_index().to_string())
    
    return df

def generar_graficos_cobit_1(df):
    """Gráficos específicos para COBIT 1"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('COBIT 1: Cumplimiento del Plan de Gobierno de TI', fontsize=16, fontweight='bold')
    
    # Convertir fecha si es necesario
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 1. Evolución temporal del cumplimiento
    if 'fecha' in df.columns and 'cumplimiento_porcentaje' in df.columns:
        df_temporal = df.groupby('fecha')['cumplimiento_porcentaje'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha'], df_temporal['cumplimiento_porcentaje'], linewidth=2)
        axes[0, 0].set_title('Evolución del Cumplimiento (%)')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Cumplimiento (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribución del cumplimiento
    if 'cumplimiento_porcentaje' in df.columns:
        axes[0, 1].hist(df['cumplimiento_porcentaje'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribución del Cumplimiento')
        axes[0, 1].set_xlabel('Cumplimiento (%)')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].axvline(df['cumplimiento_porcentaje'].mean(), color='r', linestyle='--', label=f'Media: {df["cumplimiento_porcentaje"].mean():.2f}%')
        axes[0, 1].legend()
    
    # 3. Cumplimiento por objetivo estratégico
    if 'objetivo_estrategico' in df.columns and 'cumplimiento_porcentaje' in df.columns:
        objetivo_cumplimiento = df.groupby('objetivo_estrategico')['cumplimiento_porcentaje'].mean().sort_values(ascending=False)
        axes[1, 0].barh(range(len(objetivo_cumplimiento)), objetivo_cumplimiento.values)
        axes[1, 0].set_yticks(range(len(objetivo_cumplimiento)))
        axes[1, 0].set_yticklabels(objetivo_cumplimiento.index, fontsize=8)
        axes[1, 0].set_title('Cumplimiento por Objetivo Estratégico')
        axes[1, 0].set_xlabel('Cumplimiento Promedio (%)')
    
    # 4. Distribución por estado
    if 'estado' in df.columns:
        estado_counts = df['estado'].value_counts()
        axes[1, 1].pie(estado_counts.values, labels=estado_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Distribución por Estado')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/cobit_1_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_cobit_2(df):
    """Gráficos específicos para COBIT 2"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('COBIT 2: Índice de Riesgo de TI', fontsize=16, fontweight='bold')
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 1. Evolución temporal de mitigación
    if 'fecha' in df.columns and 'indice_mitigacion_porcentaje' in df.columns:
        df_temporal = df.groupby('fecha')['indice_mitigacion_porcentaje'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha'], df_temporal['indice_mitigacion_porcentaje'], linewidth=2, color='green')
        axes[0, 0].set_title('Evolución del Índice de Mitigación (%)')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Mitigación (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mitigación por categoría de riesgo
    if 'categoria_riesgo' in df.columns and 'indice_mitigacion_porcentaje' in df.columns:
        riesgo_mitigacion = df.groupby('categoria_riesgo')['indice_mitigacion_porcentaje'].mean().sort_values()
        axes[0, 1].barh(range(len(riesgo_mitigacion)), riesgo_mitigacion.values, color='coral')
        axes[0, 1].set_yticks(range(len(riesgo_mitigacion)))
        axes[0, 1].set_yticklabels(riesgo_mitigacion.index)
        axes[0, 1].set_title('Mitigación por Categoría de Riesgo')
        axes[0, 1].set_xlabel('Mitigación Promedio (%)')
    
    # 3. Distribución de riesgos identificados
    if 'riesgos_criticos_identificados' in df.columns:
        axes[1, 0].hist(df['riesgos_criticos_identificados'], bins=15, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_title('Distribución de Riesgos Críticos Identificados')
        axes[1, 0].set_xlabel('Número de Riesgos')
        axes[1, 0].set_ylabel('Frecuencia')
    
    # 4. Distribución por estado
    if 'estado' in df.columns:
        estado_counts = df['estado'].value_counts()
        axes[1, 1].bar(estado_counts.index, estado_counts.values, color=['green', 'yellow', 'red'])
        axes[1, 1].set_title('Distribución por Estado')
        axes[1, 1].set_ylabel('Frecuencia')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/cobit_2_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_itil_1(df):
    """Gráficos específicos para ITIL 1"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ITIL 1: Tiempo de Resolución de Incidentes', fontsize=16, fontweight='bold')
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 1. Evolución temporal del tiempo promedio
    if 'fecha' in df.columns and 'tiempo_promedio_horas' in df.columns:
        df_temporal = df.groupby('fecha')['tiempo_promedio_horas'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha'], df_temporal['tiempo_promedio_horas'], linewidth=2, color='blue')
        axes[0, 0].set_title('Evolución del Tiempo Promedio de Resolución')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Tiempo (horas)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(24, color='r', linestyle='--', label='SLA 24h')
        axes[0, 0].legend()
    
    # 2. Tiempo promedio por servicio
    if 'servicio' in df.columns and 'tiempo_promedio_horas' in df.columns:
        servicio_tiempo = df.groupby('servicio')['tiempo_promedio_horas'].mean().sort_values()
        axes[0, 1].barh(range(len(servicio_tiempo)), servicio_tiempo.values, color='steelblue')
        axes[0, 1].set_yticks(range(len(servicio_tiempo)))
        axes[0, 1].set_yticklabels(servicio_tiempo.index, fontsize=8)
        axes[0, 1].set_title('Tiempo Promedio por Servicio')
        axes[0, 1].set_xlabel('Tiempo (horas)')
    
    # 3. Distribución del tiempo de resolución
    if 'tiempo_promedio_horas' in df.columns:
        axes[1, 0].hist(df['tiempo_promedio_horas'], bins=30, edgecolor='black', alpha=0.7, color='lightblue')
        axes[1, 0].set_title('Distribución del Tiempo de Resolución')
        axes[1, 0].set_xlabel('Tiempo (horas)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['tiempo_promedio_horas'].mean(), color='r', linestyle='--', 
                          label=f'Media: {df["tiempo_promedio_horas"].mean():.2f}h')
        axes[1, 0].legend()
    
    # 4. Cumplimiento SLA
    if 'cumplimiento_sla' in df.columns:
        sla_counts = df['cumplimiento_sla'].value_counts()
        axes[1, 1].pie(sla_counts.values, labels=sla_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Cumplimiento de SLA')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/itil_1_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_itil_2(df):
    """Gráficos específicos para ITIL 2"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ITIL 2: Disponibilidad de Servicios Críticos', fontsize=16, fontweight='bold')
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 1. Evolución temporal de disponibilidad
    if 'fecha' in df.columns and 'disponibilidad_porcentaje' in df.columns:
        df_temporal = df.groupby('fecha')['disponibilidad_porcentaje'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha'], df_temporal['disponibilidad_porcentaje'], linewidth=1, alpha=0.7, color='green')
        axes[0, 0].set_title('Evolución de la Disponibilidad')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Disponibilidad (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(99.9, color='r', linestyle='--', label='Objetivo 99.9%')
        axes[0, 0].legend()
    
    # 2. Disponibilidad por servicio
    if 'servicio' in df.columns and 'disponibilidad_porcentaje' in df.columns:
        servicio_disp = df.groupby('servicio')['disponibilidad_porcentaje'].mean().sort_values()
        axes[0, 1].barh(range(len(servicio_disp)), servicio_disp.values, color='lightgreen')
        axes[0, 1].set_yticks(range(len(servicio_disp)))
        axes[0, 1].set_yticklabels(servicio_disp.index, fontsize=8)
        axes[0, 1].set_title('Disponibilidad Promedio por Servicio')
        axes[0, 1].set_xlabel('Disponibilidad (%)')
    
    # 3. Distribución de disponibilidad
    if 'disponibilidad_porcentaje' in df.columns:
        axes[1, 0].hist(df['disponibilidad_porcentaje'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title('Distribución de Disponibilidad')
        axes[1, 0].set_xlabel('Disponibilidad (%)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['disponibilidad_porcentaje'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["disponibilidad_porcentaje"].mean():.4f}%')
        axes[1, 0].legend()
    
    # 4. Distribución por estado
    if 'estado' in df.columns:
        estado_counts = df['estado'].value_counts()
        axes[1, 1].bar(estado_counts.index, estado_counts.values, color=['green', 'yellow', 'red'])
        axes[1, 1].set_title('Distribución por Estado')
        axes[1, 1].set_ylabel('Frecuencia')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/itil_2_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_cmmi_1(df):
    """Gráficos específicos para CMMI 1"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CMMI 1: Procesos Documentados', fontsize=16, fontweight='bold')
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 1. Evolución temporal de documentación
    if 'fecha' in df.columns and 'porcentaje_documentacion' in df.columns:
        df_temporal = df.groupby('fecha')['porcentaje_documentacion'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha'], df_temporal['porcentaje_documentacion'], linewidth=2, color='purple')
        axes[0, 0].set_title('Evolución del Porcentaje de Documentación')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Documentación (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Documentación por área de proceso
    if 'area_proceso' in df.columns and 'porcentaje_documentacion' in df.columns:
        area_doc = df.groupby('area_proceso')['porcentaje_documentacion'].mean().sort_values()
        axes[0, 1].barh(range(len(area_doc)), area_doc.values, color='mediumpurple')
        axes[0, 1].set_yticks(range(len(area_doc)))
        axes[0, 1].set_yticklabels(area_doc.index, fontsize=7)
        axes[0, 1].set_title('Documentación por Área de Proceso')
        axes[0, 1].set_xlabel('Documentación Promedio (%)')
    
    # 3. Distribución del porcentaje de documentación
    if 'porcentaje_documentacion' in df.columns:
        axes[1, 0].hist(df['porcentaje_documentacion'], bins=30, edgecolor='black', alpha=0.7, color='plum')
        axes[1, 0].set_title('Distribución del Porcentaje de Documentación')
        axes[1, 0].set_xlabel('Documentación (%)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['porcentaje_documentacion'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["porcentaje_documentacion"].mean():.2f}%')
        axes[1, 0].legend()
    
    # 4. Distribución por estado
    if 'estado' in df.columns:
        estado_counts = df['estado'].value_counts()
        axes[1, 1].bar(estado_counts.index, estado_counts.values, color=['green', 'yellow', 'red'])
        axes[1, 1].set_title('Distribución por Estado')
        axes[1, 1].set_ylabel('Frecuencia')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/cmmi_1_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_cmmi_2(df):
    """Gráficos específicos para CMMI 2"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CMMI 2: Mejoras Implementadas', fontsize=16, fontweight='bold')
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 1. Evolución temporal de implementación
    if 'fecha' in df.columns and 'porcentaje_implementacion' in df.columns:
        df_temporal = df.groupby('fecha')['porcentaje_implementacion'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha'], df_temporal['porcentaje_implementacion'], linewidth=2, color='teal')
        axes[0, 0].set_title('Evolución del Porcentaje de Implementación')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Implementación (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Implementación por tipo de mejora
    if 'tipo_mejora' in df.columns and 'porcentaje_implementacion' in df.columns:
        tipo_impl = df.groupby('tipo_mejora')['porcentaje_implementacion'].mean().sort_values()
        axes[0, 1].barh(range(len(tipo_impl)), tipo_impl.values, color='darkcyan')
        axes[0, 1].set_yticks(range(len(tipo_impl)))
        axes[0, 1].set_yticklabels(tipo_impl.index)
        axes[0, 1].set_title('Implementación por Tipo de Mejora')
        axes[0, 1].set_xlabel('Implementación Promedio (%)')
    
    # 3. Distribución del porcentaje de implementación
    if 'porcentaje_implementacion' in df.columns:
        axes[1, 0].hist(df['porcentaje_implementacion'], bins=30, edgecolor='black', alpha=0.7, color='cyan')
        axes[1, 0].set_title('Distribución del Porcentaje de Implementación')
        axes[1, 0].set_xlabel('Implementación (%)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['porcentaje_implementacion'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["porcentaje_implementacion"].mean():.2f}%')
        axes[1, 0].legend()
    
    # 4. Distribución por estado
    if 'estado' in df.columns:
        estado_counts = df['estado'].value_counts()
        axes[1, 1].bar(estado_counts.index, estado_counts.values, color=['green', 'orange'])
        axes[1, 1].set_title('Distribución por Estado')
        axes[1, 1].set_ylabel('Frecuencia')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/cmmi_2_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_crisp_dm_1(df):
    """Gráficos específicos para CRISP-DM 1"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CRISP-DM 1: Cobertura de Datos Analizados', fontsize=16, fontweight='bold')
    
    if 'fecha_inicio' in df.columns:
        df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
    
    # 1. Evolución temporal de cobertura
    if 'fecha_inicio' in df.columns and 'cobertura_porcentaje' in df.columns:
        df_temporal = df.groupby('fecha_inicio')['cobertura_porcentaje'].mean().reset_index()
        axes[0, 0].scatter(df_temporal['fecha_inicio'], df_temporal['cobertura_porcentaje'], alpha=0.5, s=20)
        axes[0, 0].set_title('Evolución de la Cobertura de Datos')
        axes[0, 0].set_xlabel('Fecha de Inicio')
        axes[0, 0].set_ylabel('Cobertura (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cobertura por tipo de proyecto
    if 'tipo_proyecto' in df.columns and 'cobertura_porcentaje' in df.columns:
        tipo_cobertura = df.groupby('tipo_proyecto')['cobertura_porcentaje'].mean().sort_values()
        axes[0, 1].barh(range(len(tipo_cobertura)), tipo_cobertura.values, color='indigo')
        axes[0, 1].set_yticks(range(len(tipo_cobertura)))
        axes[0, 1].set_yticklabels(tipo_cobertura.index, fontsize=7)
        axes[0, 1].set_title('Cobertura por Tipo de Proyecto')
        axes[0, 1].set_xlabel('Cobertura Promedio (%)')
    
    # 3. Distribución de cobertura
    if 'cobertura_porcentaje' in df.columns:
        axes[1, 0].hist(df['cobertura_porcentaje'], bins=30, edgecolor='black', alpha=0.7, color='mediumslateblue')
        axes[1, 0].set_title('Distribución de Cobertura')
        axes[1, 0].set_xlabel('Cobertura (%)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['cobertura_porcentaje'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["cobertura_porcentaje"].mean():.2f}%')
        axes[1, 0].legend()
    
    # 4. Distribución de registros analizados
    if 'registros_analizados' in df.columns:
        axes[1, 1].hist(np.log10(df['registros_analizados']), bins=30, edgecolor='black', alpha=0.7, color='slateblue')
        axes[1, 1].set_title('Distribución de Registros Analizados (log10)')
        axes[1, 1].set_xlabel('Log10(Registros)')
        axes[1, 1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/crisp_dm_1_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_crisp_dm_2(df):
    """Gráficos específicos para CRISP-DM 2"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CRISP-DM 2: Tiempo de Respuesta en Análisis', fontsize=16, fontweight='bold')
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 1. Evolución temporal del tiempo de respuesta
    if 'fecha' in df.columns and 'tiempo_promedio_horas' in df.columns:
        df_temporal = df.groupby('fecha')['tiempo_promedio_horas'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha'], df_temporal['tiempo_promedio_horas'], linewidth=2, color='darkred')
        axes[0, 0].set_title('Evolución del Tiempo de Respuesta')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Tiempo (horas)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Tiempo por tipo de análisis
    if 'tipo_analisis' in df.columns and 'tiempo_promedio_horas' in df.columns:
        tipo_tiempo = df.groupby('tipo_analisis')['tiempo_promedio_horas'].mean().sort_values()
        axes[0, 1].barh(range(len(tipo_tiempo)), tipo_tiempo.values, color='crimson')
        axes[0, 1].set_yticks(range(len(tipo_tiempo)))
        axes[0, 1].set_yticklabels(tipo_tiempo.index)
        axes[0, 1].set_title('Tiempo Promedio por Tipo de Análisis')
        axes[0, 1].set_xlabel('Tiempo (horas)')
    
    # 3. Distribución del tiempo de respuesta
    if 'tiempo_promedio_horas' in df.columns:
        axes[1, 0].hist(df['tiempo_promedio_horas'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Distribución del Tiempo de Respuesta')
        axes[1, 0].set_xlabel('Tiempo (horas)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['tiempo_promedio_horas'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["tiempo_promedio_horas"].mean():.2f}h')
        axes[1, 0].legend()
    
    # 4. Distribución por estado
    if 'estado' in df.columns:
        estado_counts = df['estado'].value_counts()
        axes[1, 1].bar(estado_counts.index, estado_counts.values, color=['green', 'yellow', 'red'])
        axes[1, 1].set_title('Distribución por Estado')
        axes[1, 1].set_ylabel('Frecuencia')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/crisp_dm_2_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_scrum_1(df):
    """Gráficos específicos para SCRUM 1"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SCRUM 1: Velocidad del Equipo', fontsize=16, fontweight='bold')
    
    if 'fecha_inicio' in df.columns:
        df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
    
    # 1. Evolución temporal de velocidad
    if 'fecha_inicio' in df.columns and 'velocidad_puntos_historia' in df.columns:
        df_temporal = df.groupby('fecha_inicio')['velocidad_puntos_historia'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha_inicio'], df_temporal['velocidad_puntos_historia'], linewidth=2, color='darkgreen')
        axes[0, 0].set_title('Evolución de la Velocidad del Equipo')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Velocidad (puntos)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Velocidad por equipo
    if 'equipo' in df.columns and 'velocidad_puntos_historia' in df.columns:
        equipo_velocidad = df.groupby('equipo')['velocidad_puntos_historia'].mean().sort_values()
        axes[0, 1].barh(range(len(equipo_velocidad)), equipo_velocidad.values, color='forestgreen')
        axes[0, 1].set_yticks(range(len(equipo_velocidad)))
        axes[0, 1].set_yticklabels(equipo_velocidad.index)
        axes[0, 1].set_title('Velocidad Promedio por Equipo')
        axes[0, 1].set_xlabel('Velocidad (puntos)')
    
    # 3. Distribución de velocidad
    if 'velocidad_puntos_historia' in df.columns:
        axes[1, 0].hist(df['velocidad_puntos_historia'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Distribución de Velocidad')
        axes[1, 0].set_xlabel('Velocidad (puntos)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['velocidad_puntos_historia'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["velocidad_puntos_historia"].mean():.2f} puntos')
        axes[1, 0].legend()
    
    # 4. Distribución por tendencia
    if 'tendencia' in df.columns:
        tendencia_counts = df['tendencia'].value_counts()
        axes[1, 1].bar(tendencia_counts.index, tendencia_counts.values, color=['green', 'blue', 'orange'])
        axes[1, 1].set_title('Distribución por Tendencia')
        axes[1, 1].set_ylabel('Frecuencia')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/scrum_1_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_graficos_scrum_2(df):
    """Gráficos específicos para SCRUM 2"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SCRUM 2: Cumplimiento de Retrospectivas', fontsize=16, fontweight='bold')
    
    if 'fecha_sprint' in df.columns:
        df['fecha_sprint'] = pd.to_datetime(df['fecha_sprint'])
    
    # 1. Evolución temporal del cumplimiento
    if 'fecha_sprint' in df.columns and 'cumplimiento_porcentaje' in df.columns:
        df_temporal = df.groupby('fecha_sprint')['cumplimiento_porcentaje'].mean().reset_index()
        axes[0, 0].plot(df_temporal['fecha_sprint'], df_temporal['cumplimiento_porcentaje'], linewidth=2, color='darkblue')
        axes[0, 0].set_title('Evolución del Cumplimiento de Retrospectivas')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Cumplimiento (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(100, color='g', linestyle='--', label='Objetivo 100%')
        axes[0, 0].legend()
    
    # 2. Cumplimiento por equipo
    if 'equipo' in df.columns and 'cumplimiento_porcentaje' in df.columns:
        equipo_cumplimiento = df.groupby('equipo')['cumplimiento_porcentaje'].mean().sort_values()
        axes[0, 1].barh(range(len(equipo_cumplimiento)), equipo_cumplimiento.values, color='steelblue')
        axes[0, 1].set_yticks(range(len(equipo_cumplimiento)))
        axes[0, 1].set_yticklabels(equipo_cumplimiento.index)
        axes[0, 1].set_title('Cumplimiento Promedio por Equipo')
        axes[0, 1].set_xlabel('Cumplimiento (%)')
    
    # 3. Distribución del cumplimiento
    if 'cumplimiento_porcentaje' in df.columns:
        axes[1, 0].hist(df['cumplimiento_porcentaje'], bins=20, edgecolor='black', alpha=0.7, color='lightblue')
        axes[1, 0].set_title('Distribución del Cumplimiento')
        axes[1, 0].set_xlabel('Cumplimiento (%)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(df['cumplimiento_porcentaje'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["cumplimiento_porcentaje"].mean():.2f}%')
        axes[1, 0].legend()
    
    # 4. Distribución por estado
    if 'estado' in df.columns:
        estado_counts = df['estado'].value_counts()
        axes[1, 1].bar(estado_counts.index, estado_counts.values, color=['green', 'yellow', 'orange'])
        axes[1, 1].set_title('Distribución por Estado')
        axes[1, 1].set_ylabel('Frecuencia')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/graficos/scrum_2_analisis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_resumen_consolidado(datasets_info):
    """Genera un resumen consolidado de todos los análisis"""
    resumen_path = f'{CARPETA_RESULTADOS}/resumen_eda.txt'
    
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RESUMEN CONSOLIDADO - ANÁLISIS EXPLORATORIO DE DATOS\n")
        f.write("Gobierno Inteligente de TI - DIAN\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_registros = sum(info['registros'] for info in datasets_info.values())
        f.write(f"TOTAL DE REGISTROS ANALIZADOS: {total_registros:,}\n")
        f.write(f"NÚMERO DE DATASETS: {len(datasets_info)}\n\n")
        
        f.write("RESUMEN POR DATASET:\n")
        f.write("-"*70 + "\n")
        for nombre, info in datasets_info.items():
            f.write(f"\n{info['nombre_kpi']}:\n")
            f.write(f"  - Registros: {info['registros']:,}\n")
            f.write(f"  - Columnas: {info['columnas']}\n")
            f.write(f"  - Período: {info.get('periodo', 'N/A')}\n")
            if info.get('insights'):
                f.write(f"  - Insights principales:\n")
                for insight in info['insights']:
                    f.write(f"    • {insight}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("GRÁFICOS GENERADOS\n")
        f.write("="*70 + "\n")
        f.write("Todos los gráficos se encuentran en: eda_resultados/graficos/\n")
        f.write("- cobit_1_analisis.png\n")
        f.write("- cobit_2_analisis.png\n")
        f.write("- itil_1_analisis.png\n")
        f.write("- itil_2_analisis.png\n")
        f.write("- cmmi_1_analisis.png\n")
        f.write("- cmmi_2_analisis.png\n")
        f.write("- crisp_dm_1_analisis.png\n")
        f.write("- crisp_dm_2_analisis.png\n")
        f.write("- scrum_1_analisis.png\n")
        f.write("- scrum_2_analisis.png\n")
    
    print(f"\n✓ Resumen consolidado guardado en: {resumen_path}")

def main():
    """Función principal que ejecuta el EDA completo"""
    print("="*70)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("Gobierno Inteligente de TI - DIAN")
    print("="*70)
    
    # Mapeo de archivos a funciones de gráficos
    datasets_config = {
        'cobit_1_cumplimiento_plan_gobierno_ti.csv': {
            'nombre_kpi': 'COBIT 1: Cumplimiento del Plan de Gobierno de TI',
            'funcion_graficos': generar_graficos_cobit_1
        },
        'cobit_2_indice_riesgo_ti.csv': {
            'nombre_kpi': 'COBIT 2: Índice de Riesgo de TI',
            'funcion_graficos': generar_graficos_cobit_2
        },
        'itil_1_tiempo_resolucion_incidentes.csv': {
            'nombre_kpi': 'ITIL 1: Tiempo de Resolución de Incidentes',
            'funcion_graficos': generar_graficos_itil_1
        },
        'itil_2_disponibilidad_servicios.csv': {
            'nombre_kpi': 'ITIL 2: Disponibilidad de Servicios Críticos',
            'funcion_graficos': generar_graficos_itil_2
        },
        'cmmi_1_procesos_documentados.csv': {
            'nombre_kpi': 'CMMI 1: Procesos Documentados',
            'funcion_graficos': generar_graficos_cmmi_1
        },
        'cmmi_2_mejoras_implementadas.csv': {
            'nombre_kpi': 'CMMI 2: Mejoras Implementadas',
            'funcion_graficos': generar_graficos_cmmi_2
        },
        'crisp_dm_1_cobertura_datos_analizados.csv': {
            'nombre_kpi': 'CRISP-DM 1: Cobertura de Datos Analizados',
            'funcion_graficos': generar_graficos_crisp_dm_1
        },
        'crisp_dm_2_tiempo_respuesta_analisis.csv': {
            'nombre_kpi': 'CRISP-DM 2: Tiempo de Respuesta en Análisis',
            'funcion_graficos': generar_graficos_crisp_dm_2
        },
        'scrum_1_velocidad_equipo.csv': {
            'nombre_kpi': 'SCRUM 1: Velocidad del Equipo',
            'funcion_graficos': generar_graficos_scrum_1
        },
        'scrum_2_cumplimiento_retrospectivas.csv': {
            'nombre_kpi': 'SCRUM 2: Cumplimiento de Retrospectivas',
            'funcion_graficos': generar_graficos_scrum_2
        }
    }
    
    datasets_info = {}
    
    # Analizar cada dataset
    for archivo, config in datasets_config.items():
        df = analizar_dataset(archivo, config['nombre_kpi'])
        
        # Extraer información adicional
        fecha_cols = [col for col in df.columns if 'fecha' in col.lower()]
        periodo = 'N/A'
        if fecha_cols:
            fecha_col = fecha_cols[0]
            df_temp = df.copy()
            df_temp[fecha_col] = pd.to_datetime(df_temp[fecha_col])
            periodo = f"{df_temp[fecha_col].min().strftime('%Y-%m-%d')} a {df_temp[fecha_col].max().strftime('%Y-%m-%d')}"
        
        # Generar insights básicos
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Primeras 3 columnas numéricas
                mean_val = df[col].mean()
                insights.append(f"{col}: promedio = {mean_val:.2f}")
        
        datasets_info[archivo] = {
            'nombre_kpi': config['nombre_kpi'],
            'registros': len(df),
            'columnas': len(df.columns),
            'periodo': periodo,
            'insights': insights
        }
        
        # Generar gráficos específicos
        print(f"\nGenerando gráficos para {config['nombre_kpi']}...")
        config['funcion_graficos'](df)
        print(f"✓ Gráficos guardados en: {CARPETA_RESULTADOS}/graficos/")
    
    # Generar resumen consolidado
    generar_resumen_consolidado(datasets_info)
    
    print("\n" + "="*70)
    print("✓ ANÁLISIS EXPLORATORIO COMPLETADO")
    print("="*70)
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}/")
    print(f"- Gráficos: {CARPETA_RESULTADOS}/graficos/")
    print(f"- Resumen: {CARPETA_RESULTADOS}/resumen_eda.txt")

if __name__ == "__main__":
    main()

