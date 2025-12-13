"""
Script para generar datasets dummy de KPIs para el Gobierno Inteligente de TI de la DIAN
Metodología para un Gobierno Inteligente de TI: Un enfoque de IA para Evaluación y Mejora Continua
Genera datos realistas con volumen adecuado para modelos de IA predictivos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuración de semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# ============================================================================
# CONFIGURACIÓN DE VOLUMEN AUMENTADO
# ============================================================================
"""
VOLUMEN MEJORADO PARA INVESTIGACIÓN CON IA:

1. COBIT 1 (Trimestral): 20 registros (5 años de datos)
2. COBIT 2 (Mensual): 60 registros (5 años de datos)
3. ITIL 1 (Mensual + Incidentes individuales): 60 meses + ~3,000 incidentes individuales
4. ITIL 2 (Diario): 1,825 registros (5 años de datos)
5. CMMI 1 (Semestral): 10 registros (5 años de datos)
6. CMMI 2 (Trimestral): 20 registros (5 años de datos)
7. CRISP-DM 1 (Por proyecto): 50 proyectos
8. CRISP-DM 2 (Mensual): 60 registros (5 años de datos)
9. SCRUM 1 (Por sprint): 60 sprints (5 años)
10. SCRUM 2 (Por sprint): 60 sprints (5 años)

TOTAL APROXIMADO: ~6,000+ registros distribuidos en 10 datasets
"""

# Configuración de años
AÑO_INICIO = 2020
AÑO_FIN = 2024
NUM_SERVICIOS = 8  # Más servicios para mayor granularidad

# Configuración de nombres
NOMBRE_CARPETA = '../data/datasets/datasets_gobierno_ti_inteligente_dian'

def generar_cobit_1():
    """
    COBIT 1: Cumplimiento del Plan de Gobierno de TI
    Frecuencia: Semanal (para llegar a 1000+ registros)
    """
    print("Generando COBIT 1: Cumplimiento del Plan de Gobierno de TI...")
    
    # Cambiar a semanal para tener más registros (~260 semanas en 5 años)
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    # Generar múltiples objetivos estratégicos por semana
    objetivos_estrategicos = ['Modernización TI', 'Seguridad Cibernética', 'Transformación Digital',
                             'Optimización Infraestructura', 'Mejora Procesos', 'Gobierno de Datos',
                             'Automatización', 'Innovación Tecnológica', 'Cumplimiento Normativo',
                             'Experiencia Usuario', 'Resiliencia Operacional', 'Analítica Avanzada']
    
    datos = []
    for fecha in fechas:
        # Seleccionar 3-5 objetivos aleatorios por semana
        num_objetivos = np.random.randint(3, 6)
        objetivos_seleccionados = np.random.choice(objetivos_estrategicos, size=num_objetivos, replace=False)
        
        for objetivo in objetivos_seleccionados:
            objetivos_planificados = np.random.randint(5, 15)
            # Simular cumplimiento entre 65% y 96% con tendencia positiva
            años_transcurridos = (fecha.year - AÑO_INICIO) + (fecha.dayofyear / 365)
            cumplimiento_base = 0.65 + (años_transcurridos / 5) * 0.31  # Mejora del 65% al 96%
            cumplimiento_pct = cumplimiento_base + np.random.uniform(-0.05, 0.05)
            cumplimiento_pct = np.clip(cumplimiento_pct, 0.60, 0.98)
    
            objetivos_alcanzados = int(objetivos_planificados * cumplimiento_pct)
            
            datos.append({
                'fecha': fecha,
                'año': fecha.year,
                'mes': fecha.strftime('%Y-%m'),
                'semana': fecha.isocalendar()[1],
                'trimestre': f"Q{((fecha.month-1)//3)+1}",
                'objetivo_estrategico': objetivo,
        'objetivos_planificados': objetivos_planificados,
        'objetivos_alcanzados': objetivos_alcanzados,
                'cumplimiento_porcentaje': round(cumplimiento_pct * 100, 2),
                'estado': 'Excelente' if cumplimiento_pct >= 0.90 else 'Aceptable' if cumplimiento_pct >= 0.70 else 'Riesgo'
    })
    
    df = pd.DataFrame(datos)
    return df

def generar_cobit_2():
    """
    COBIT 2: Índice de riesgo de TI
    Frecuencia: Semanal (para llegar a 1000+ registros)
    """
    print("Generando COBIT 2: Índice de riesgo de TI...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    # Categorías de riesgo
    categorias_riesgo = ['Seguridad', 'Disponibilidad', 'Cumplimiento', 'Operacional', 'Tecnológico']
    
    datos = []
    for fecha in fechas:
        # Generar registros por cada categoría de riesgo
        for categoria in categorias_riesgo:
            riesgos_criticos_identificados = np.random.randint(2, 8)
            # Simular mitigación entre 65% y 96% con tendencia positiva
            años_transcurridos = (fecha.year - AÑO_INICIO) + (fecha.dayofyear / 365)
            mitigacion_base = 0.65 + (años_transcurridos / 5) * 0.31
            mitigacion_pct = mitigacion_base + np.random.uniform(-0.08, 0.08)
            mitigacion_pct = np.clip(mitigacion_pct, 0.60, 0.98)
            
            riesgos_mitigados = int(riesgos_criticos_identificados * mitigacion_pct)
            
            datos.append({
                'fecha': fecha,
                'año': fecha.year,
                'mes': fecha.strftime('%Y-%m'),
                'semana': fecha.isocalendar()[1],
                'categoria_riesgo': categoria,
        'riesgos_criticos_identificados': riesgos_criticos_identificados,
        'riesgos_mitigados': riesgos_mitigados,
                'indice_mitigacion_porcentaje': round(mitigacion_pct * 100, 2),
                'estado': 'Excelente' if mitigacion_pct >= 0.90 else 'Aceptable' if mitigacion_pct >= 0.70 else 'Riesgo Alto'
    })
    
    df = pd.DataFrame(datos)
    return df

def generar_itil_1():
    """
    ITIL 1: Tiempo promedio de resolución de incidentes
    Frecuencia: Semanal por servicio (para llegar a 1000+ registros)
    """
    print("Generando ITIL 1: Tiempo resolución incidentes...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    servicios = ['Sistema Tributario', 'Portal Web', 'API de Servicios', 'Sistema de Facturación', 
                 'Plataforma de Pagos', 'Sistema Aduanero', 'BI y Reportes', 'Infraestructura']
    
    datos = []
    
    for fecha in fechas:
        for servicio in servicios:
            num_incidentes = np.random.randint(10, 50)
        
        # Simular tiempos de resolución según distribución realista
        # Tiempo promedio mejorando con el tiempo
            años_transcurridos = (fecha.year - AÑO_INICIO) + (fecha.dayofyear / 365)
            factor_mejora = 1 - (años_transcurridos * 0.15) / 5
            factor_mejora = max(0.5, factor_mejora)
        
        # Tiempo base entre 8 y 48 horas
        tiempo_base = np.random.uniform(48, 8)
        tiempo_promedio = tiempo_base * factor_mejora
        
            # Agregar variación
        tiempo_promedio = tiempo_promedio + np.random.uniform(-5, 5)
            tiempo_promedio = max(2, tiempo_promedio)
        
            datos.append({
            'fecha': fecha,
            'mes': fecha.strftime('%Y-%m'),
            'año': fecha.year,
                'semana': fecha.isocalendar()[1],
                'servicio': servicio,
            'numero_incidentes': num_incidentes,
            'tiempo_promedio_horas': round(tiempo_promedio, 2),
            'tiempo_promedio_dias': round(tiempo_promedio / 24, 2),
            'cumplimiento_sla': 'Cumple' if tiempo_promedio <= 24 else 'Requiere atención'
        })
    
    df = pd.DataFrame(datos)
    return df

def generar_itil_2():
    """
    ITIL 2: Disponibilidad de servicios críticos
    Frecuencia: Diario - 5 AÑOS
    """
    print("Generando ITIL 2: Disponibilidad de servicios (5 años diario)...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='D')
    
    servicios = ['Sistema Tributario', 'Portal Web', 'API de Servicios', 'Sistema de Facturación', 
                 'Plataforma de Pagos', 'Sistema Aduanero', 'BI y Reportes', 'Infraestructura']
    
    datos = []
    
    for fecha in fechas:
        for servicio in servicios:
            # Disponibilidad base entre 98.5% y 99.95%
            disponibilidad_base = np.random.uniform(0.985, 0.9995)
            
            # Simular incidentes ocasionales (5% de probabilidad)
            if np.random.random() < 0.05:
                disponibilidad_base = np.random.uniform(0.92, 0.98)
            
            # Mejora a lo largo del tiempo
            años_transcurridos = (fecha.year - AÑO_INICIO) + (fecha.dayofyear / 365)
            factor_mejora = min(1.0, 1 + (años_transcurridos * 0.01))
            disponibilidad_pct = min(0.9999, disponibilidad_base * factor_mejora)
            
            tiempo_operativo_horas = disponibilidad_pct * 24
            tiempo_indisponible_horas = 24 - tiempo_operativo_horas
            
            datos.append({
                'fecha': fecha,
                'servicio': servicio,
                'disponibilidad_porcentaje': round(disponibilidad_pct * 100, 4),
                'tiempo_operativo_horas': round(tiempo_operativo_horas, 4),
                'tiempo_indisponible_horas': round(tiempo_indisponible_horas, 4),
                'estado': ['Alta' if x >= 99.9 else 'Aceptable' if x >= 99.0 else 'Riesgo Alto' 
                          for x in [disponibilidad_pct * 100]][0]
            })
    
    df = pd.DataFrame(datos)
    return df

def generar_cmmi_1():
    """
    CMMI 1: Procesos documentados
    Frecuencia: Semanal por área de proceso (para llegar a 1000+ registros)
    """
    print("Generando CMMI 1: Procesos documentados...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    areas_proceso = ['Gestión de Proyectos', 'Gestión de Requisitos', 'Desarrollo de Software',
                     'Gestión de Configuración', 'Aseguramiento de Calidad', 'Gestión de Riesgos',
                     'Gestión de Proveedores', 'Medición y Análisis', 'Mejora de Procesos',
                     'Gestión de Recursos Humanos']
    
    datos = []
    for fecha in fechas:
        for area in areas_proceso:
            procesos_totales = np.random.randint(5, 15)
            # Porcentaje de documentación mejorando con el tiempo
            años_transcurridos = (fecha.year - AÑO_INICIO) + (fecha.dayofyear / 365)
            documentacion_base = 0.70 + (años_transcurridos / 5) * 0.26
            documentacion_pct = documentacion_base + np.random.uniform(-0.05, 0.05)
            documentacion_pct = np.clip(documentacion_pct, 0.65, 0.98)
    
            procesos_documentados = int(procesos_totales * documentacion_pct)
            
            datos.append({
                'fecha': fecha,
                'mes': fecha.strftime('%Y-%m'),
                'año': fecha.year,
                'semana': fecha.isocalendar()[1],
                'trimestre': f"Q{((fecha.month-1)//3)+1}",
                'area_proceso': area,
        'procesos_totales': procesos_totales,
        'procesos_documentados': procesos_documentados,
                'porcentaje_documentacion': round(documentacion_pct * 100, 2),
                'estado': 'Alto nivel' if documentacion_pct >= 0.90 else 'Aceptable' if documentacion_pct >= 0.70 else 'Riesgo Alto'
    })
    
    df = pd.DataFrame(datos)
    return df

def generar_cmmi_2():
    """
    CMMI 2: Mejoras implementadas
    Frecuencia: Semanal por tipo de mejora (para llegar a 1000+ registros)
    """
    print("Generando CMMI 2: Mejoras implementadas...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    tipos_mejora = ['Proceso', 'Tecnología', 'Organizacional', 'Metodología', 'Herramienta']
    
    datos = []
    for fecha in fechas:
        for tipo in tipos_mejora:
            mejoras_planificadas = np.random.randint(2, 8)
            # Porcentaje de implementación entre 75% y 100%
            implementacion_pct = np.random.uniform(0.75, 1.0)
            
            mejoras_implementadas = int(mejoras_planificadas * implementacion_pct)
            
            datos.append({
                'fecha': fecha,
                'mes': fecha.strftime('%Y-%m'),
                'año': fecha.year,
                'semana': fecha.isocalendar()[1],
                'trimestre': f"Q{((fecha.month-1)//3)+1}",
                'tipo_mejora': tipo,
        'mejoras_planificadas': mejoras_planificadas,
        'mejoras_implementadas': mejoras_implementadas,
                'porcentaje_implementacion': round(implementacion_pct * 100, 2),
                'estado': 'Evolución positiva' if implementacion_pct >= 0.90 else 'Revisar plan'
    })
    
    df = pd.DataFrame(datos)
    return df

def generar_crisp_dm_1():
    """
    CRISP-DM 1: Cobertura de datos analizados
    Frecuencia: Por proyecto - Múltiples proyectos por período
    """
    print("Generando CRISP-DM 1: Cobertura de datos analizados...")
    
    # Generar proyectos distribuidos en 5 años pero con múltiples proyectos por semana
    fechas_base = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    num_proyectos = 1200
    
    proyectos = [f"Proyecto_{i+1:04d}" for i in range(num_proyectos)]
    
    tipos_proyecto = ['Análisis Predictivo', 'Optimización', 'Detección de Anomalías', 
                      'Clasificación', 'Regresión', 'Clustering', 'NLP', 'Visión']
    
    registros_totales = np.random.randint(10000, 500000, size=num_proyectos)
    # Cobertura entre 70% y 96%
    cobertura_pct = np.random.uniform(0.70, 0.96, size=num_proyectos)
    
    registros_analizados = (registros_totales * cobertura_pct).astype(int)
    
    # Distribuir proyectos a lo largo de las fechas base (múltiples por semana)
    proyectos_por_semana = num_proyectos // len(fechas_base) + 1
    fechas_inicio = []
    for fecha in fechas_base:
        for _ in range(proyectos_por_semana):
            if len(fechas_inicio) < num_proyectos:
                # Agregar variación de días dentro de la semana
                fecha_proyecto = fecha + timedelta(days=np.random.randint(0, 6))
                fechas_inicio.append(fecha_proyecto)
    
    fechas_inicio = fechas_inicio[:num_proyectos]  # Asegurar exactamente num_proyectos
    fechas_inicio = pd.to_datetime(fechas_inicio)
    
    df = pd.DataFrame({
        'proyecto': proyectos,
        'tipo_proyecto': np.random.choice(tipos_proyecto, size=num_proyectos),
        'fecha_inicio': fechas_inicio,
        'año': fechas_inicio.year,
        'registros_totales_disponibles': registros_totales,
        'registros_analizados': registros_analizados,
        'cobertura_porcentaje': cobertura_pct * 100,
        'estado': ['Alta representatividad' if x >= 90 else 'Aceptable' if x >= 70 else 'Riesgo de sesgo' 
                   for x in cobertura_pct * 100]
    })
    
    return df

def generar_crisp_dm_2():
    """
    CRISP-DM 2: Tiempo de respuesta en análisis
    Frecuencia: Semanal por tipo de análisis (para llegar a 1000+ registros)
    """
    print("Generando CRISP-DM 2: Tiempo de respuesta en análisis...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    tipos_analisis = ['Descriptivo', 'Predictivo', 'Prescriptivo', 'Diagnóstico', 'Exploratorio']
    
    datos = []
    for fecha in fechas:
        for tipo in tipos_analisis:
            num_informes = np.random.randint(2, 10)
            # Tiempo promedio mejorando con el tiempo
            años_transcurridos = (fecha.year - AÑO_INICIO) + (fecha.dayofyear / 365)
            tiempo_base = 72 - (años_transcurridos / 5) * 64  # De 72 a 8 horas
            tiempo_promedio_horas = tiempo_base + np.random.uniform(-5, 5)
            tiempo_promedio_horas = np.clip(tiempo_promedio_horas, 4, 80)
            
            datos.append({
                'fecha': fecha,
                'mes': fecha.strftime('%Y-%m'),
                'año': fecha.year,
                'semana': fecha.isocalendar()[1],
                'tipo_analisis': tipo,
        'numero_informes_generados': num_informes,
                'tiempo_promedio_horas': round(tiempo_promedio_horas, 2),
                'tiempo_promedio_dias': round(tiempo_promedio_horas / 24, 2),
                'estado': 'Excelente agilidad' if tiempo_promedio_horas <= 24 else 'Aceptable' if tiempo_promedio_horas <= 72 else 'Riesgo de retraso'
            })
    
    df = pd.DataFrame(datos)
    return df

def generar_scrum_1():
    """
    SCRUM 1: Velocidad del equipo
    Frecuencia: Por sprint y equipo (para llegar a 1000+ registros)
    """
    print("Generando SCRUM 1: Velocidad del equipo...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    equipos = ['Equipo Desarrollo', 'Equipo Infraestructura', 'Equipo Seguridad', 
                'Equipo Datos', 'Equipo Integración']
    
    datos = []
    sprint_num = 1
    
    for fecha in fechas:
        for equipo in equipos:
            # Velocidad en puntos de historia (entre 20 y 60 puntos)
            velocidad_base = np.random.randint(20, 60)
            # Tendencia de mejora a lo largo del tiempo
            años_transcurridos = (fecha.year - AÑO_INICIO) + (fecha.dayofyear / 365)
            factor_mejora = 1 + (años_transcurridos / 5) * 0.2  # Mejora del 20% en 5 años
            velocidad_puntos = int(velocidad_base * factor_mejora)
            velocidad_puntos = np.clip(velocidad_puntos, 15, 70)
    
    # Calcular tendencia
            media_velocidad = 40  # Media esperada
            tendencia = 'Estable' if abs(velocidad_puntos - media_velocidad) < 5 else \
                       'Creciente' if velocidad_puntos > media_velocidad else 'Decreciente'
            
            datos.append({
                'sprint': f"Sprint_{sprint_num:03d}",
                'fecha_inicio': fecha,
                'año': fecha.year,
                'semana': fecha.isocalendar()[1],
                'equipo': equipo,
                'velocidad_puntos_historia': velocidad_puntos,
        'tendencia': tendencia,
                'rendimiento': 'Buen rendimiento' if tendencia in ['Estable', 'Creciente'] else 'Posible sobrecarga'
    })
        sprint_num += 1
    
    df = pd.DataFrame(datos)
    return df

def generar_scrum_2():
    """
    SCRUM 2: Cumplimiento de retrospectivas
    Frecuencia: Por sprint y equipo (para llegar a 1000+ registros)
    """
    print("Generando SCRUM 2: Cumplimiento de retrospectivas...")
    
    fechas = pd.date_range(start=f'{AÑO_INICIO}-01-01', end=f'{AÑO_FIN}-12-31', freq='W')
    
    equipos = ['Equipo Desarrollo', 'Equipo Infraestructura', 'Equipo Seguridad', 
                'Equipo Datos', 'Equipo Integración']
    
    datos = []
    sprint_num = 1
    
    for fecha in fechas:
        for equipo in equipos:
            retrospectivas_planificadas = 1
            # Cumplimiento entre 85% y 100%
            cumplimiento_pct = np.random.uniform(0.85, 1.0)
            
            # Ocasionalmente no se realiza (5% de probabilidad)
            if np.random.random() < 0.05:
                retrospectivas_realizadas = 0
            else:
                retrospectivas_realizadas = 1
            
            cumplimiento_final = (retrospectivas_realizadas / retrospectivas_planificadas) * 100
            
            datos.append({
                'sprint': f"Sprint_{sprint_num:03d}",
                'fecha_sprint': fecha,
                'año': fecha.year,
                'semana': fecha.isocalendar()[1],
                'equipo': equipo,
        'retrospectivas_planificadas': retrospectivas_planificadas,
        'retrospectivas_realizadas': retrospectivas_realizadas,
        'cumplimiento_porcentaje': cumplimiento_final,
                'estado': 'Cumplimiento total' if cumplimiento_final == 100 else 'Aceptable' if cumplimiento_final >= 80 else 'Requiere acciones'
    })
        sprint_num += 1
    
    df = pd.DataFrame(datos)
    return df

def guardar_datasets():
    """Genera y guarda todos los datasets en formato CSV"""
    
    print("="*70)
    print("GENERADOR DE DATASETS - GOBIERNO INTELIGENTE DE TI DIAN")
    print("Metodología para un Gobierno Inteligente de TI")
    print("="*70)
    print()
    
    datasets = {}
    
    # Generar exactamente 10 datasets, uno por indicador
    datasets['cobit_1_cumplimiento_plan_gobierno_ti'] = generar_cobit_1()
    datasets['cobit_2_indice_riesgo_ti'] = generar_cobit_2()
    datasets['itil_1_tiempo_resolucion_incidentes'] = generar_itil_1()
    datasets['itil_2_disponibilidad_servicios'] = generar_itil_2()
    datasets['cmmi_1_procesos_documentados'] = generar_cmmi_1()
    datasets['cmmi_2_mejoras_implementadas'] = generar_cmmi_2()
    datasets['crisp_dm_1_cobertura_datos_analizados'] = generar_crisp_dm_1()
    datasets['crisp_dm_2_tiempo_respuesta_analisis'] = generar_crisp_dm_2()
    datasets['scrum_1_velocidad_equipo'] = generar_scrum_1()
    datasets['scrum_2_cumplimiento_retrospectivas'] = generar_scrum_2()
    
    # Crear directorio de salida
    import os
    output_dir = NOMBRE_CARPETA
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar cada dataset
    resumen = []
    total_registros = 0
    
    for nombre, df in datasets.items():
        archivo = f"{output_dir}/{nombre}.csv"
        df.to_csv(archivo, index=False, encoding='utf-8')
        registros = len(df)
        total_registros += registros
        resumen.append({
            'Indicador': nombre,
            'Registros': registros,
            'Archivo': archivo
        })
        print(f"✓ Guardado: {archivo} ({registros:,} registros)")
    
    # Guardar resumen
    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(f"{output_dir}/resumen_datasets.csv", index=False, encoding='utf-8')
    
    print(f"\n{'='*70}")
    print("RESUMEN DE DATASETS GENERADOS")
    print(f"{'='*70}")
    print(resumen_df.to_string(index=False))
    print(f"\n{'='*70}")
    print(f"Total de registros generados: {total_registros:,}")
    print(f"Archivos guardados en: {output_dir}/")
    print(f"{'='*70}\n")
    
    return datasets

if __name__ == "__main__":
    datasets = guardar_datasets()
    
    print("\n✓ Proceso completado exitosamente!")
    print(f"\n✓ Generados exactamente 10 datasets (uno por cada indicador)")
    print(f"✓ Dataset más pequeño: {min(len(df) for df in datasets.values()):,} registros")
    print("\nCaracterísticas de los datasets:")
    print("- Mínimo 1,000 registros por dataset")
    print("- 5 años de datos históricos (2020-2024)")
    print("- Datos realistas con variaciones y tendencias temporales")
    print("- Adecuados para modelos predictivos con IA")
    print("\nLos datasets están listos para:")
    print("- Modelos predictivos con IA")
    print("- Análisis de tendencias a largo plazo")
    print("- Detección de patrones")
    print("- Simulaciones de escenarios")

