import streamlit as st
import sqlite3
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import pandas as pd

# ---- CONFIGURACIÓN ----
DB_FILE = "sirfoc.db"
st.set_page_config(page_title="VECTOR DOCENTE", layout="wide")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RUBRICA_DATA = {
    "dominio_curricular": {
        "titulo": "1. Dominio Curricular y Pedagógico",
        "descriptores": [
            "Inicial: Aplico la currícula prescrita y sigo las guías establecidas. Busco comprender los fundamentos y la implementación del Plan de Estudio 2022 y la NEM.",
            "En Desarrollo: Comienzo a realizar adaptaciones a los contenidos curriculares para responder a las necesidades de mi grupo. Consulto diversas fuentes para enriquecer mi entendimiento de la NEM.",
            "Consolidado: Tomo decisiones pedagógicas fundamentadas para contextualizar la currícula, articulando los ejes, campos formativos y contenidos de manera coherente en mi planeación.",
            "Destacado: Lidero el codiseño curricular en mi colectivo, generando propuestas que articulan el Plan de Estudio 2022 con el proyecto escolar."
        ],
        "tags": "currículo, Plan de Estudio 2022, Nueva Escuela Mexicana, pedagogía, contextualización"
    },
    "reflexion_practica": {
        "titulo": "2. Reflexión y Transformación de la Práctica",
        "descriptores": [
            "Inicial: Identifico las problemáticas generales del aula, aunque me centro principalmente en factores externos.",
            "En Desarrollo: Empiezo a analizar cómo mis propias acciones y decisiones influyen en las dinámicas del aula, buscando alternativas para mejorar.",
            "Consolidado: Utilizo de manera sistemática herramientas de reflexión (como bitácoras) para analizar mi práctica y ajustar mis estrategias.",
            "Destacado: Promuevo activamente la reflexión colectiva. Sistematizo y comparto los hallazgos de mi práctica para transformar el quehacer docente del colectivo."
        ],
        "tags": "reflexión, práctica docente, transformación, saberes docentes, problematización, autoanálisis, mejora"
    },
    "colaboracion_dialogo": {
        "titulo": "3. Colaboración y Diálogo Profesional",
        "descriptores": [
            "Inicial: Mi enfoque de trabajo es principalmente individual, aunque participo en las actividades generales de la escuela.",
            "En Desarrollo: Intercambio ideas y materiales con colegas de confianza y participo en las discusiones del Consejo Técnico Escolar (CTE).",
            "Consolidado: Colaboro activamente en el diseño y ejecución de proyectos colectivos en el CTE, utilizando el diálogo para construir acuerdos.",
            "Destacado: Impulso y lidero comunidades de aprendizaje profesional, creando una cultura de confianza y colaboración."
        ],
        "tags": "colaboración, diálogo, trabajo en equipo, comunidades de aprendizaje, corresponsabilidad"
    },
    "liderazgo_autonomia": {
        "titulo": "4. Liderazgo y Autonomía Profesional",
        "descriptores": [
            "Inicial: Aplico las directrices curriculares e institucionales para asegurar la coherencia del servicio educativo.",
            "En Desarrollo: Propongo adaptaciones a la planeación, argumentando mis decisiones pedagógicas con base en las necesidades de mis estudiantes.",
            "Consolidado: Ejerzo mi autonomía profesional para tomar decisiones curriculares y de gestión que se alinean con el proyecto escolar.",
            "Destacado: Asumo un rol de liderazgo pedagógico, inspirando y coordinando acciones para la mejora continua y fomentando la autonomía de mis colegas."
        ],
        "tags": "autonomía, liderazgo, toma de decisiones, responsabilidad, función directiva"
    },
    "evaluacion_aprendizaje": {
        "titulo": "5. Evaluación para el Aprendizaje",
        "descriptores": [
            "Inicial: Me enfoco en la aplicación de instrumentos para obtener una calificación, identificando principalmente aciertos y errores.",
            "En Desarrollo: Busco incorporar la retroalimentación durante los procesos de aprendizaje para orientar a los estudiantes.",
            "Consolidado: Implemento de forma sistemática la autoevaluación y coevaluación, y ofrezco retroalimentación descriptiva que ayuda a mejorar.",
            "Destacado: He logrado que mis estudiantes se apropien del proceso evaluativo, utilizando la autoevaluación para autorregular su aprendizaje."
        ],
        "tags": "evaluación formativa, retroalimentación, autoevaluación, coevaluación, mejora del aprendizaje, enseñanza"
    },
    "atencion_diversidad": {
        "titulo": "6. Atención a la Diversidad e Inclusión",
        "descriptores": [
            "Inicial: Reconozco la diversidad en mi aula y busco estrategias para atender las necesidades que se presentan.",
            "En Desarrollo: Identifico las principales barreras para el aprendizaje que enfrentan algunos de mis estudiantes y busco información para atenderlas.",
            "Consolidado: Diseño e implemento de manera consistente planeaciones diversificadas y ajustes razonables para todos mis estudiantes.",
            "Destacado: Promuevo activamente una cultura de aula incluyente e intercultural, donde la diversidad es reconocida como una fortaleza."
        ],
        "tags": "diversidad, inclusión, equidad, empatía, respeto, derechos humanos, vulnerabilidad"
    },
    "gestion_recursos_tic": {
        "titulo": "7. Gestión de Recursos y Tecnologías",
        "descriptores": [
            "Inicial: Utilizo los recursos tecnológicos disponibles principalmente para tareas administrativas o para la exposición de información.",
            "En Desarrollo: Exploro y utilizo diversas herramientas y plataformas digitales como un recurso complementario en mis clases.",
            "Consolidado: Integro intencionadamente recursos tecnológicos en mi planeación para que los estudiantes investiguen, creen y colaboren.",
            "Destacado: Diseño experiencias de aprendizaje donde la tecnología es un mediador pedagógico clave y gestiono creativamente los recursos."
        ],
        "tags": "TIC, tecnología educativa, recursos didácticos, herramientas digitales, plataformas, materiales"
    },
    "gestion_desarrollo_profesional": {
        "titulo": "8. Gestión del Desarrollo Profesional Continuo",
        "descriptores": [
            "Inicial: Busco cursos y capacitaciones que me ofrezcan respuestas y soluciones claras a las problemáticas que enfrento.",
            "En Desarrollo: Intercambio experiencias con mis colegas y consulto diversas fuentes para mantenerme actualizado.",
            "Consolidado: Identifico de forma autónoma mis necesidades de formación y participo en trayectos que promueven la reflexión.",
            "Destacado: Lidero el diagnóstico de necesidades formativas en mi colectivo y promuevo la participación en trayectos de desarrollo."
        ],
        "tags": "formación continua, desarrollo profesional, necesidades formativas, planificación, monitoreo, evaluación de programas"
    },
    "adaptabilidad_resolucion": {
        "titulo": "9. Adaptabilidad y Resolución de Problemas",
        "descriptores": [
            "Inicial: Enfrento las situaciones complejas priorizando las tareas administrativas y de gestión.",
            "En Desarrollo: Comienzo a problematizar los retos cotidianos, analizando sus posibles causas y explorando alternativas.",
            "Consolidado: Abordo los desafíos de la práctica con una metodología clara, analizando causas y efectos para proponer soluciones.",
            "Destacado: Demuestro una visión estratégica para anticipar y resolver problemas complejos, transformando desafíos en oportunidades."
        ],
        "tags": "adaptabilidad, flexibilidad, resolución de problemas, gestión del cambio, retos, desafíos"
    },
    "documentacion_sistematizacion": {
        "titulo": "10. Documentación y Sistematización",
        "descriptores": [
            "Inicial: Mi práctica se basa principalmente en la experiencia y el intercambio oral, sin un registro formal.",
            "En Desarrollo: Realizo registros esporádicos sobre experiencias significativas para guiar mi reflexión personal.",
            "Consolidado: Documento de forma sistemática mis planeaciones, proyectos y reflexiones, creando un portafolio profesional.",
            "Destacado: Impulso la cultura de la documentación y sistematización en mi colectivo, generando evidencia para la toma de decisiones."
        ],
        "tags": "documentación, sistematización, registro, evidencia, informes"
    }
}

COMPETENCIAS_EVALUADAS = list(RUBRICA_DATA.keys())

# ---- CACHE DEL MODELO IA ----
@st.cache_resource
def load_model():
    """Carga y cachea el modelo de IA para evitar recargas innecesarias"""
    try:
        logger.info("Cargando modelo SentenceTransformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        st.error("Error al cargar el modelo de IA. Por favor, reinicia la aplicación.")
        return None

model = load_model()

# ========== ANÁLISIS SEMÁNTICO DEL CONTEXTO PERSONAL ==========
def extraer_insights_contexto(contexto_personal, model):
    """Extrae insights clave del contexto personal usando IA"""
    if not contexto_personal or not contexto_personal.strip():
        return {}
    
    # Palabras clave para identificar aspectos importantes
    keywords_map = {
        'nivel_educativo': ['primaria', 'secundaria', 'bachillerato', 'universidad', 'preescolar', 'infantil'],
        'recursos_tecnologicos': ['tecnología', 'digital', 'computadora', 'internet', 'plataforma', 'virtual', 'online'],
        'tamaño_grupo': ['grupo grande', 'grupo pequeño', 'muchos alumnos', 'pocos estudiantes', 'numeroso'],
        'experiencia': ['años', 'experiencia', 'nuevo', 'novato', 'veterano', 'experimentado'],
        'desafios': ['dificultad', 'problema', 'desafío', 'reto', 'limitación', 'obstáculo'],
        'intereses': ['interesa', 'gusta', 'motiva', 'pasión', 'enfoque', 'especialidad'],
        'modalidad': ['presencial', 'virtual', 'híbrido', 'mixto', 'distancia', 'remoto']
    }
    
    contexto_lower = contexto_personal.lower()
    insights = {}
    
    # Detectar características clave
    for categoria, keywords in keywords_map.items():
        matches = [kw for kw in keywords if kw in contexto_lower]
        if matches:
            insights[categoria] = matches
    
    # Calcular embedding del contexto para análisis semántico
    try:
        contexto_embedding = model.encode([contexto_personal])[0]
        insights['embedding_contexto'] = contexto_embedding
    except Exception as e:
        logger.error(f"Error al generar embedding del contexto: {e}")
        insights['embedding_contexto'] = None
    
    return insights

# ---- VALIDACIÓN DE DATOS ----
def validar_datos_entrada(responses):
    """Valida que los datos de entrada sean correctos"""
    if not responses:
        raise ValueError("No se recibieron respuestas")
    
    for key in COMPETENCIAS_EVALUADAS:
        if key not in responses:
            raise ValueError(f"Falta respuesta para {key}")
        if not responses[key]:
            raise ValueError(f"Respuesta vacía para {key}")
    
    logger.info("Validación de datos exitosa")
    return True

# ---- BASE DE DATOS ----
def inicializar_base_de_datos():
    """Inicializa la base de datos con estructura mejorada"""
    if os.path.exists(DB_FILE):
        logger.info("Base de datos existente encontrada")
        return
    
    logger.info("Creando nueva base de datos...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Tablas básicas
    cursor.execute('''CREATE TABLE IF NOT EXISTS Docentes (
        id_docente INTEGER PRIMARY KEY,
        nombre_docente TEXT,
        perfil_simulado TEXT,
        fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        contexto_actual TEXT DEFAULT ''
    );''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS Cursos (
        id_curso INTEGER PRIMARY KEY,
        clave_registrada TEXT UNIQUE,
        nombre_curso TEXT,
        horas INTEGER,
        descripcion TEXT,
        competencias_clave TEXT,
        modalidad TEXT DEFAULT 'Presencial',
        nivel_dificultad INTEGER DEFAULT 2
    );''')
    
    # Tabla de evaluaciones con timestamp
    columnas_sql = ", ".join([f"{comp} REAL" for comp in COMPETENCIAS_EVALUADAS])
    cursor.execute(f'''CREATE TABLE IF NOT EXISTS Evaluaciones (
        id_evaluacion INTEGER PRIMARY KEY,
        id_docente INTEGER,
        {columnas_sql},
        fecha_evaluacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        contexto_personal TEXT DEFAULT '',
        FOREIGN KEY (id_docente) REFERENCES Docentes (id_docente)
    );''')
    
    # Cursos con datos mejorados
    cursos_a_insertar = [
    # DOMINIO CURRICULAR Y PEDAGÓGICO
    ('C1420240080', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL EDUCACION INICIAL UN BUEN COMIENZO', 40,
     'Desarrolla competencias curriculares específicas para educación inicial basadas en los aprendizajes clave.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240081', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL EDUCACION PREESCOLAR', 40,
     'Fortalece el dominio curricular en preescolar mediante aprendizajes clave para la educación integral.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240082', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL PRIMERO PRIMARIA', 40,
     'Desarrolla competencias curriculares específicas para primer grado de primaria.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240083', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL SEGUNDO PRIMARIA', 40,
     'Fortalece el dominio curricular para segundo grado de primaria basado en aprendizajes clave.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240084', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL TERCERO PRIMARIA', 40,
     'Desarrolla competencias curriculares específicas para tercer grado de primaria.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240085', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL CUARTO PRIMARIA', 40,
     'Fortalece el dominio curricular para cuarto grado de primaria mediante aprendizajes clave.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240086', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL QUINTO PRIMARIA', 40,
     'Desarrolla competencias curriculares específicas para quinto grado de primaria.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240087', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL SEXTO PRIMARIA', 40,
     'Fortalece el dominio curricular para sexto grado de primaria basado en aprendizajes clave.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240088', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL LENGUA MATERNA ESPAÑOL EN LA EDUCACION SECUNDARIA', 40,
     'Desarrolla competencias curriculares específicas en Lengua Materna Español para secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240089', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL MATEMATICAS EN LA EDUCACION SECUNDARIA', 40,
     'Fortalece el dominio curricular en matemáticas para educación secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240090', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL ARTES EN LA EDUCACION SECUNDARIA', 40,
     'Desarrolla competencias curriculares específicas en artes para secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240092', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL CIENCIAS Y TECNOLOGIA EN LA EDUCACION SECUNDARIA', 40,
     'Fortalece el dominio curricular en ciencias y tecnología para educación secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240093', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL HISTORIA EN LA EDUCACION SECUNDARIA', 40,
     'Desarrolla competencias curriculares específicas en historia para secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240094', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL GEOGRAFIA EN LA EDUCACION SECUNDARIA', 40,
     'Fortalece el dominio curricular en geografía para educación secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240095', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL FORMACION CIVICA Y ETICA EN LA EDUCACION SECUNDARIA', 40,
     'Desarrolla competencias curriculares en formación cívica y ética para secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240096', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL TUTORIA Y EDUCACION SOCIOEMOCIONAL EN LA EDUCACION SECUNDARIA', 40,
     'Fortalece competencias en tutoría y educación socioemocional para secundaria.',
     'dominio_curricular,atencion_diversidad', 'Presencial', 3),
    
    ('C1420240097', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL INGLES EN LA EDUCACION BASICA', 40,
     'Desarrolla competencias curriculares específicas en inglés para educación básica.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240098', 'APRENDIZAJES CLAVE PARA LA EDUCACION INTEGRAL EDUCACION FISICA EN LA EDUCACION BASICA', 40,
     'Fortalece el dominio curricular en educación física para educación básica.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240099', 'PROYECTAR LA ENSEÑANZA EDUCACION PREESCOLAR', 40,
     'Desarrolla competencias para proyectar y planificar la enseñanza en preescolar.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240100', 'PROYECTAR LA ENSEÑANZA EDUCACION PRIMARIA', 40,
     'Fortalece competencias para proyectar y planificar la enseñanza en primaria.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240101', 'PROYECTAR LA ENSEÑANZA EDUCACION SECUNDARIA', 40,
     'Desarrolla competencias para proyectar y planificar la enseñanza en secundaria.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240102', 'PROYECTAR LA ENSEÑANZA EDUCACION TELESECUNDARIA', 40,
     'Fortalece competencias específicas para proyectar la enseñanza en telesecundaria.',
     'dominio_curricular,gestion_recursos_tic', 'Presencial', 3),
    
    ('C1420240114', 'ANALISIS Y REFLEXION DEL PLAN DE ESTUDIOS 2022', 120,
     'Analiza y reflexiona profundamente sobre el Plan de Estudios 2022 y su implementación.',
     'dominio_curricular,reflexion_practica', 'Presencial', 4),
    
    ('C1420240138', 'ANÁLISIS Y REFLEXIÓN DEL PLAN DE ESTUDIOS 2022', 120,
     'Curso intensivo para analizar y reflexionar sobre el Plan de Estudios 2022.',
     'dominio_curricular,reflexion_practica', 'Presencial', 4),
    
    ('C1420240133', 'CODISEÑO Y PROGRAMA ANALÍTICO. UNA EXPERIENCIA DESDE EL COLECTIVO DOCENTE', 40,
     'Desarrolla competencias para el codiseño curricular y creación de programas analíticos.',
     'dominio_curricular,colaboracion_dialogo', 'Presencial', 4),
    
    ('C1420240134', 'ELEMENTOS ESENCIALES DEL PROGRAMA ANALÍTICO PARA UNA ENSEÑANZA SITUADA', 20,
     'Identifica elementos esenciales para crear programas analíticos contextualizados.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240141', 'ESTRATEGIAS PARA LA CONSTITUCIÓN DEL PROGRAMA ANALÍTICO', 20,
     'Proporciona estrategias prácticas para constituir programas analíticos efectivos.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420240072', 'PLANEACION DIDACTICA ELEMENTOS ESENCIALES', 20,
     'Desarrolla competencias básicas en planeación didáctica y sus elementos esenciales.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240111', 'EL AULA MULTIGRADO UN ESPACIO DE APRENDIZAJE PERMANENTE', 40,
     'Fortalece competencias específicas para la enseñanza en aulas multigrado.',
     'dominio_curricular,adaptabilidad_resolucion', 'Presencial', 4),
    
    ('C1420240077', 'APRENDIZAJE A TRAVES DEL JUEGO EN PREESCOLAR MODALIDAD EN LINEA', 20,
     'Desarrolla estrategias de aprendizaje lúdico específicas para preescolar.',
     'dominio_curricular', 'Virtual', 2),
    
    ('C1420240076', 'JUGANDO CON LOS NUMEROS', 30,
     'Fortalece competencias para enseñanza de matemáticas mediante estrategias lúdicas.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420250031', 'LOS ELEMENTOS DE LA NUEVA ESCUELA MEXICANA; LA PLANIFICACIÓN Y EL DISEÑO DE PROYECTOS EDUCATIVOS', 40,
     'Desarrolla competencias para planificar y diseñar proyectos educativos en el marco de la NEM.',
     'dominio_curricular', 'Presencial', 3),
    
    ('C1420250009', 'CURSO INTRODUCCIÓN AL MARCO INSTRUCCIONAL STEM', 40,
     'Introduce metodologías STEM para integración curricular innovadora.',
     'dominio_curricular,gestion_recursos_tic', 'Presencial', 3),
    
    ('C1420250020', 'EDUCACIÓN MUSICAL EN MÉXICO', 20,
     'Fortalece competencias curriculares específicas en educación musical.',
     'dominio_curricular', 'Presencial', 2),
    
    ('C1420240203', 'PREESCOLAR Y LOS CUATRO CAMPOS FORMATIVOS', 20,
     'Desarrolla competencias curriculares específicas para los campos formativos de preescolar.',
     'dominio_curricular', 'Presencial', 2),
    
    # REFLEXIÓN Y TRANSFORMACIÓN DE LA PRÁCTICA
    ('C1420240063', 'FORTALECIMIENTO DEL PERFIL PROFESIONAL PARA LA PROMOCION HORIZONTAL DE DOCENTES EN EDUCACION BASICA', 40,
     'Fortalece el perfil profesional docente mediante la reflexión crítica sobre la práctica y el desarrollo de competencias.',
     'reflexion_practica,liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240112', 'UNA MAESTRA, UN MAESTRO QUE ASUME SU QUEHACER PROFESIONAL CON APEGO A LOS PRINCIPIOS FILOSOFICOS, ETICOS Y LEGALES DE LA EDUCACION MEXICANA', 30,
     'Reflexiona sobre la práctica docente desde principios éticos y legales de la educación mexicana.',
     'reflexion_practica,liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240121', 'UNA MAESTRA, UN MAESTRO QUE PARTICIPA Y COLABORA EN LA TRANSFORMACION Y MEJORA DE LA ESCUELA Y LA COMUNIDAD', 30,
     'Desarrolla competencias para la transformación escolar mediante la participación y colaboración.',
     'reflexion_practica,colaboracion_dialogo', 'Presencial', 3),
    
    ('C1420240123', 'UNA MAESTRA UN MAESTRO QUE CONOCE A SUS ALUMNOS PARA BRINDARLES UNA ATENCION EDUCATIVA CON INCLUSION EQUIDAD Y EXCELENCIA', 30,
     'Reflexiona sobre la práctica para brindar atención educativa inclusiva y equitativa.',
     'reflexion_practica,atencion_diversidad', 'Presencial', 3),
    
    ('C1420240129', 'UNA MAESTRA, UN MAESTRO QUE ASUME SU QUEHACER PROFESIONAL CON APEGO A LOS PRINCIPIOS FILOSÓFICOS, ÉTICOS Y LEGALES DE LA EDUCACIÓN MEXICANA', 30,
     'Reflexiona críticamente sobre la práctica docente desde principios filosóficos y éticos.',
     'reflexion_practica,liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240139', 'UNA MAESTRA, UN MAESTRO QUE CONOCE A SUS ALUMNOS PARA BRINDARLES UNA ATENCIÓN EDUCATIVA CON INCLUSIÓN, EQUIDAD Y EXCELENCIA', 30,
     'Desarrolla la reflexión sobre la práctica para atender la diversidad con equidad.',
     'reflexion_practica,atencion_diversidad', 'Presencial', 3),
    
    ('C1420240146', 'UNA MAESTRA, UN MAESTRO QUE PARTICIPA Y COLABORA EN LA TRANSFORMACIÓN Y MEJORA DE LA ESCUELA Y LA COMUNIDAD', 30,
     'Fortalece la reflexión sobre el rol docente en la transformación de la comunidad educativa.',
     'reflexion_practica,colaboracion_dialogo', 'Presencial', 3),
    
    ('C1420240132', 'LA SISTEMATIZACIÓN DE EXPERIENCIAS EN LA EDUCACIÓN BÁSICA', 20,
     'Desarrolla competencias para sistematizar y reflexionar sobre experiencias educativas.',
     'reflexion_practica,documentacion_sistematizacion', 'Presencial', 3),
    
    ('C1420240136', 'FORTALECIMIENTO DEL PERFIL PROFESIONAL PARA LA PROMOCIÓN HORIZONTAL DE DOCENTES DE EDUCACIÓN BÁSICA', 40,
     'Fortalece el perfil profesional mediante la reflexión y transformación de la práctica.',
     'reflexion_practica,liderazgo_autonomia', 'Presencial', 3),
    
    # COLABORACIÓN Y DIÁLOGO PROFESIONAL
    ('C1420240113', 'FORTALECIMIENTO DE LA DIRECCION ESCOLAR PARA LA MEJORA EDUCATIVA', 120,
     'Desarrolla competencias colaborativas y de liderazgo para directivos escolares.',
     'colaboracion_dialogo,liderazgo_autonomia', 'Presencial', 4),
    
    ('C1420240193', 'EL LIDERAZGO DEL SUPERVISOR INCLUYENTE, UN TRABAJO COLABORATIVO CON LA COMUNIDAD EDUCATIVA', 40,
     'Fortalece el liderazgo colaborativo e incluyente para supervisores educativos.',
     'colaboracion_dialogo,liderazgo_autonomia', 'Presencial', 4),
    
    ('C1420240194', 'EL LIDERAZGO DIRECTIVO INCLUYENTE, UN TRABAJO COLABORATIVO CON LA COMUNIDAD EDUCATIVA', 40,
     'Desarrolla competencias de liderazgo colaborativo para directivos escolares.',
     'colaboracion_dialogo,liderazgo_autonomia', 'Presencial', 4),
    
    ('C1420240195', 'EL LIDERAZGO DOCENTE INCLUYENTE, UN TRABAJO COLABORATIVO CON LA COMUNIDAD EDUCATIVA.', 40,
     'Fortalece el liderazgo colaborativo e incluyente en la práctica docente.',
     'colaboracion_dialogo,liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240130', 'DIÁLOGO COMO HERRAMIENTA PARA EL RESPETO A LA DIVERSIDAD', 30,
     'Desarrolla competencias dialógicas para el respeto y atención a la diversidad.',
     'colaboracion_dialogo,atencion_diversidad', 'Presencial', 2),
    
    ('C1420250037', 'RECONSTRUIMOS COMUNIDAD', 40,
     'Fortalece competencias para la reconstrucción colaborativa del tejido comunitario.',
     'colaboracion_dialogo', 'Presencial', 3),
    
    ('C1420250022', 'ESTRATEGIAS PARA EL CULTIVO DE LA EMPATÍA EN LAS COMUNIDADES DE APRENDIZAJE EN Y PARA LA VIDA CAV', 40,
     'Desarrolla estrategias colaborativas para cultivar la empatía en comunidades de aprendizaje.',
     'colaboracion_dialogo,atencion_diversidad', 'Presencial', 3),
    
    # LIDERAZGO Y AUTONOMÍA PROFESIONAL
    ('C1420240067', 'FUNDAMENTOS PROFESIONALES PARA LA PROMOCION DIRECTIVA Y DE SUPERVISION EN EDUCACION BASICA', 40,
     'Desarrolla fundamentos profesionales para el ejercicio de funciones directivas y de supervisión.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240109', 'CONOCIMIENTOS Y HABILIDADES PARA LA FUNCION DE DIRECTOR DE EDUCACION BASICA', 40,
     'Fortalece conocimientos y habilidades específicas para la función directiva.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240110', 'CONOCIMIENTOS Y HABILIDADES PARA LA FUNCION DE SUPERVISOR DE EDUCACION BASICA', 40,
     'Desarrolla competencias específicas para la función de supervisión educativa.',
     'liderazgo_autonomia', 'Presencial', 4),
    
    ('C1420240115', 'UN DIRECTIVO QUE ASUME SU PRACTICA Y DESARROLLO PROFESIONAL CON APEGO A LOS PRINCIPIOS FILOSOFICOS, ETICOS Y LEGALES DE LA EDUCACION MEXICANA', 30,
     'Fortalece la autonomía y liderazgo directivo basado en principios éticos y legales.',
     'liderazgo_autonomia,reflexion_practica', 'Presencial', 3),
    
    ('C1420240116', 'GESTION ESCOLAR, LIDERAZGO DIRECTIVO DE IMPACTO PARA EL DESARROLLO INTEGRAL DEL ESTUDIANTE', 20,
     'Desarrolla competencias de liderazgo directivo para el desarrollo integral estudiantil.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240117', 'UN DIRECTIVO QUE RECONOCE LA IMPORTANCIA DE SU FUNCION PARA CONSTRUIR DE MANERA COLECTIVA UNA CULTURA ESCOLAR CENTRADA EN LA EQUIDAD, LA INCLUSION, LA INTERCULTURALIDAD Y LA EXCELENCIA', 30,
     'Fortalece el liderazgo directivo para construir culturas escolares inclusivas y equitativas.',
     'liderazgo_autonomia,atencion_diversidad', 'Presencial', 4),
    
    ('C1420240119', 'UN DIRECTIVO QUE ORGANIZA EL FUNCIONAMIENTO DE LA ESCUELA COMO UN ESPACIO PARA LA FORMACION INTEGRAL DE LAS NIÑAS, LOS NIÑOS Y ADOLESCENTES', 30,
     'Desarrolla competencias para organizar la escuela como espacio de formación integral.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240120', 'UN DIRECTIVO QUE PROPICIA LA CORRESPONSABILIDAD DE LA ESCUELA CON LAS FAMILIAS, LA COMUNIDAD Y LAS AUTORIDADES EDUCATIVAS PARA FAVORECER LA FORMACION INTEGRAL Y EL BIENESTAR DE LOS ALUMNOS', 30,
     'Fortalece el liderazgo para propiciar corresponsabilidad entre escuela, familia y comunidad.',
     'liderazgo_autonomia,colaboracion_dialogo', 'Presencial', 4),
    
    ('C1420240142', 'UN DIRECTIVO QUE PROPICIA LA CORRESPONSABILIDAD DE LA ESCUELA CON LAS FAMILIAS, LA COMUNIDAD Y LAS AUTORIDADES EDUCATIVAS PARA FAVORECER LA FORMACIÓN INTEGRAL Y EL BIENESTAR DE LOS ALUMNOS', 30,
     'Desarrolla liderazgo para la corresponsabilidad educativa con familias y comunidad.',
     'liderazgo_autonomia,colaboracion_dialogo', 'Presencial', 4),
    
    ('C1420240143', 'UN DIRECTIVO QUE ORGANIZA EL FUNCIONAMIENTO DE LA ESCUELA COMO UN ESPACIO PARA LA FORMACIÓN INTEGRAL DE LAS NIÑAS, LOS NIÑOS Y ADOLESCENTES', 30,
     'Fortalece competencias directivas para organizar espacios de formación integral.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240144', 'UN DIRECTIVO QUE RECONOCE LA IMPORTANCIA DE LA FUNCIÓN PARA CONSTRUIR DE MANERA COLECTIVA UNA CULTURA ESCOLAR CENTRADA EN LA EQUIDAD, LA INCLUSIÓN, LA INTERCULTURALIDAD Y LA EXCELENCIA', 30,
     'Desarrolla liderazgo directivo para construir culturas escolares centradas en la equidad.',
     'liderazgo_autonomia,atencion_diversidad', 'Presencial', 4),
    
    ('C1420240145', 'UN DIRECTIVO QUE ASUME SU PRÁCTICA Y DESARROLLO PROFESIONAL CON APEGO A LOS PRINCIPIOS FILOSÓFICOS, ÉTICOS Y LEGALES DE LA EDUCACIÓN MEXICANA', 30,
     'Fortalece la autonomía profesional directiva basada en principios éticos y legales.',
     'liderazgo_autonomia,reflexion_practica', 'Presencial', 3),
    
    ('C1420240149', 'FUNDAMENTOS PROFESIONALES PARA LA PROMOCIÓN DIRECTIVA Y DE SUPERVISIÓN EN EDUCACIÓN BÁSICA', 40,
     'Desarrolla fundamentos profesionales para funciones de liderazgo educativo.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420240197', 'HERRAMIENTAS Y HABILIDADES PARA EL FORTALECIMIENTO PROFESIONAL DIRECTIVO, LOGRO DE LOS APRENDIZAJES DE LOS NIÑOS Y LAS NIÑAS DE EDUCACIÓN BÁSICA', 40,
     'Fortalece herramientas y habilidades para el liderazgo directivo efectivo.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    ('C1420250025', 'HERRAMIENTAS Y HABILIDADES PARA EL FORTALECIMIENTO PROFESIONAL DOCENTE. LOGRO DE LOS APRENDIZAJES DE LOS NIÑOS Y NIÑAS DE EDUCACIÓN BÁSICA', 40,
     'Desarrolla herramientas para el fortalecimiento de la autonomía profesional docente.',
     'liderazgo_autonomia', 'Presencial', 3),
    
    # EVALUACIÓN PARA EL APRENDIZAJE
    ('C1420240070', 'EVALUACION DIAGNOSTICA PARA LAS ALUMNAS Y LOS ALUMNOS DE EDUCACION BASICA', 20,
     'Desarrolla competencias para realizar evaluación diagnóstica efectiva en educación básica.',
     'evaluacion_aprendizaje', 'Presencial', 2),
    
    ('C1420240071', 'MEJOREMOS LA EVALUACION EN EL AULA', 40,
     'Fortalece prácticas de evaluación formativa para mejorar los aprendizajes.',
     'evaluacion_aprendizaje', 'Presencial', 2),
    
    ('C1420240118', 'EVALUACION AUTENTICA, APRENDIZAJE SITUADO BASADO EN PROYECTOS', 20,
     'Desarrolla competencias en evaluación auténtica mediante aprendizaje basado en proyectos.',
     'evaluacion_aprendizaje,dominio_curricular', 'Presencial', 3),
    
    ('C1420240131', 'EL ENFOQUE FORMATIVO DE LA EVALUACIÓN EN LA EDUCACIÓN BÁSICA', 120,
     'Curso intensivo sobre enfoque formativo de la evaluación para transformar la práctica.',
     'evaluacion_aprendizaje', 'Presencial', 4),
    
    ('C1420240137', 'EVALUACIÓN AUTÉNTICA, APRENDIZAJE SITUADO BASADO EN PROYECTOS', 20,
     'Fortalece competencias en evaluación auténtica y aprendizaje situado.',
     'evaluacion_aprendizaje,dominio_curricular', 'Presencial', 3),
    
    ('C1420240150', 'LA EVALUACIÓN FORMATIVA EN EDUCACIÓN BÁSICA', 20,
     'Desarrolla competencias básicas en evaluación formativa para educación básica.',
     'evaluacion_aprendizaje', 'Presencial', 2),
    
    ('C1420240155', 'TALLER DE EVALUACIÓN DIAGNÓSTICA PARA LAS ALUMNAS Y LOS ALUMNOS DE EDUCACIÓN BÁSICA', 20,
     'Taller práctico para desarrollar competencias en evaluación diagnóstica.',
     'evaluacion_aprendizaje', 'Presencial', 2),
    
    ('C1420240201', 'LA EVALUACIÓN FORMATIVA', 120,
     'Curso intensivo para dominar la evaluación formativa como herramienta de mejora.',
     'evaluacion_aprendizaje', 'Presencial', 4),
    
    ('C1420250028', 'LA EVALUACIÓN FORMATIVA DESDE LOS CAMPOS FORMATIVOS EN LA NUEVA ESCUELA MEXICANA', 120,
     'Desarrolla competencias avanzadas en evaluación formativa en el marco de la NEM.',
     'evaluacion_aprendizaje,dominio_curricular', 'Presencial', 4),
    
    ('C1420250039', 'TALLER DE EVALUACIÓN DIAGNÓSTICA DE LOS APRENDIZAJES CON ENFOQUE FORMATIVO EN EL MARCO DE LA NEM', 20,
     'Taller para desarrollar evaluación diagnóstica formativa en el contexto de la NEM.',
     'evaluacion_aprendizaje,dominio_curricular', 'Presencial', 3),
    
    # ATENCIÓN A LA DIVERSIDAD E INCLUSIÓN
    ('C1420240075', 'LA ESCUELA INCLUSIVA, UNA PROPUESTA DE EDUCACION PARA TODOS', 20,
     'Desarrolla competencias para crear escuelas inclusivas que atiendan la diversidad.',
     'atencion_diversidad', 'Presencial', 2),
    
    ('C1420240103', 'CONVIVENCIA SIN VIOLENCIA', 40,
     'Fortalece competencias para promover la convivencia pacífica y sin violencia.',
     'atencion_diversidad', 'Presencial', 2),
    
    ('C1420240104', 'DERECHOS HUMANOS Y COMUNIDAD ESCOLAR', 40,
     'Desarrolla competencias en derechos humanos para la comunidad escolar.',
     'atencion_diversidad', 'Presencial', 3),
    
    ('C1420240105', 'DERECHOS HUMANOS EN EL SERVICIO PUBLICO', 40,
     'Fortalece la comprensión de derechos humanos en el servicio educativo público.',
     'atencion_diversidad', 'Presencial', 3),
    
    ('C1420240106', 'NOMBRAR NOS Y HABITAR NOS DESDE LA PERSPECTIVA DE GENERO', 40,
     'Desarrolla competencias para abordar la perspectiva de género en educación.',
     'atencion_diversidad', 'Presencial', 3),
    
    ('C1420240107', 'PREVENCION DEL ACOSO Y HOSTIGAMIENTO SEXUAL', 40,
     'Fortalece competencias para prevenir y atender el acoso y hostigamiento sexual.',
     'atencion_diversidad', 'Presencial', 3),
    
    ('C1420240108', 'PREVENCION Y ATENCION DE LA VIOLENCIA SEXUAL EN CONTRA DE LAS INFANCIAS EN CENTROS ESCOLARES', 40,
     'Desarrolla competencias para prevenir y atender violencia sexual contra menores.',
     'atencion_diversidad', 'Presencial', 4)
    ]
    
    cursor.executemany("""INSERT INTO Cursos 
        (clave_registrada, nombre_curso, horas, descripcion, competencias_clave, modalidad, nivel_dificultad)
        VALUES (?, ?, ?, ?, ?, ?, ?)""", cursos_a_insertar)
    
    conn.commit()
    conn.close()
    logger.info("Base de datos inicializada correctamente")

def actualizar_base_de_datos_contexto():
    """Actualiza la base de datos para incluir contexto personal"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Verificar si la columna ya existe
    cursor.execute("PRAGMA table_info(Evaluaciones)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'contexto_personal' not in columns:
        try:
            cursor.execute("""
                ALTER TABLE Evaluaciones 
                ADD COLUMN contexto_personal TEXT DEFAULT ''
            """)
            logger.info("Columna contexto_personal agregada exitosamente")
        except sqlite3.Error as e:
            logger.error(f"Error al agregar columna: {e}")
    
    # También agregar a tabla Docentes para persistencia
    cursor.execute("PRAGMA table_info(Docentes)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'contexto_actual' not in columns:
        try:
            cursor.execute("""
                ALTER TABLE Docentes 
                ADD COLUMN contexto_actual TEXT DEFAULT ''
            """)
            logger.info("Columna contexto_actual agregada a Docentes")
        except sqlite3.Error as e:
            logger.error(f"Error al agregar contexto_actual: {e}")
    
    conn.commit()
    conn.close()

# ========== FUNCIÓN MEJORADA PARA GENERAR PERFIL CON CONTEXTO ==========
def generar_prompt_perfil_con_contexto(scores, contexto_personal=""):
    """Genera un prompt enriquecido que incluye el contexto personal del docente"""
    niveles = ["Muy bajo", "Bajo", "Medio", "Alto"]
    
    # Identificar fortalezas y debilidades
    fortalezas = []
    debilidades = []
    competencias_medias = []
    
    for i, (comp, score) in enumerate(zip(COMPETENCIAS_EVALUADAS, scores)):
        nivel = niveles[int(score)-1]
        comp_titulo = RUBRICA_DATA[comp]["titulo"]
        
        if score >= 3.5:
            fortalezas.append(f"{comp_titulo} ({nivel})")
        elif score <= 2:
            debilidades.append(f"{comp_titulo} ({nivel})")
        else:
            competencias_medias.append(f"{comp_titulo} ({nivel})")
    
    # Crear prompt estructurado y enriquecido
    prompt_parts = [
        "=== PERFIL DOCENTE PARA RECOMENDACIÓN DE CURSOS ===",
        ""
    ]
    
    # SECCIÓN 1: CONTEXTO PERSONAL (NUEVO)
    if contexto_personal and contexto_personal.strip():
        prompt_parts.extend([
            "📝 CONTEXTO PERSONAL Y PRÁCTICA ACTUAL:",
            f"'{contexto_personal.strip()}'",
            ""
        ])
    
    # SECCIÓN 2: ANÁLISIS DE COMPETENCIAS
    prompt_parts.extend([
        "📊 ANÁLISIS DE COMPETENCIAS:",
        f"💪 FORTALEZAS IDENTIFICADAS: {', '.join(fortalezas) if fortalezas else 'En desarrollo general'}",
        f"🎯 ÁREAS PRIORITARIAS DE MEJORA: {', '.join(debilidades) if debilidades else 'Perfil equilibrado'}",
        f"⚖️ COMPETENCIAS EN DESARROLLO: {', '.join(competencias_medias) if competencias_medias else 'Ninguna'}",
        ""
    ])
    
    # SECCIÓN 3: PERFIL DETALLADO POR COMPETENCIA
    prompt_parts.append("🔍 PERFIL DETALLADO POR COMPETENCIA:")
    for comp, score in zip(COMPETENCIAS_EVALUADAS, scores):
        nivel = niveles[int(score)-1]
        tags = ', '.join(RUBRICA_DATA[comp]["tags"])
        descriptor = RUBRICA_DATA[comp]["descriptores"][int(score)-1]
        prompt_parts.append(
            f"• {comp.replace('_', ' ').title()}: {nivel} (Puntuación: {score}/4) "
            f"- {descriptor} [Tags: {tags}]"
        )
    
    # SECCIÓN 4: RESUMEN ESTADÍSTICO
    promedio = np.mean(scores)
    desviacion = np.std(scores)
    prompt_parts.extend([
        "",
        "📈 RESUMEN ESTADÍSTICO:",
        f"• Promedio general: {promedio:.2f}/4",
        f"• Variabilidad del perfil: {desviacion:.2f}",
        f"• Fortalezas identificadas: {len(fortalezas)}",
        f"• Áreas de mejora prioritarias: {len(debilidades)}"
    ])
    
    return "\n".join(prompt_parts)

# ========== JUSTIFICACIÓN MEJORADA CON CONTEXTO ==========
def generar_justificacion_con_contexto(curso_data, scores, insights_contexto):
    """Genera justificación personalizada considerando contexto personal"""
    justificaciones = []
    
    # Justificación por competencias (original)
    competencias_curso = curso_data.get('competencias_clave', '').split(',')
    debilidades = []
    
    for i, (comp, score) in enumerate(zip(COMPETENCIAS_EVALUADAS, scores)):
        if score <= 2 and comp in competencias_curso:
            debilidades.append(RUBRICA_DATA[comp]["titulo"])
    
    if debilidades:
        justificaciones.append(f"Fortalecerá tus áreas de mejora en: {', '.join(debilidades)}")
    
    # Justificación por contexto personal
    if insights_contexto:
        # Modalidad
        if 'modalidad' in insights_contexto:
            modalidad_curso = curso_data.get('modalidad', '')
            if any(mod in modalidad_curso.lower() for mod in insights_contexto['modalidad']):
                justificaciones.append(f"Se adapta a tu preferencia por modalidad {modalidad_curso.lower()}")
        
        # Recursos tecnológicos
        if 'recursos_tecnologicos' in insights_contexto and len(insights_contexto['recursos_tecnologicos']) > 0:
            if curso_data.get('modalidad') in ['Virtual', 'Mixto']:
                justificaciones.append("Aprovecha tu interés en tecnología educativa")
        
        # Desafíos mencionados
        if 'desafios' in insights_contexto:
            justificaciones.append("Aborda algunos de los desafíos que mencionaste en tu práctica")
        
        # Intereses específicos
        if 'intereses' in insights_contexto:
            justificaciones.append("Conecta con los intereses que expresaste")
    
    # Justificación por defecto si no hay específicas
    if not justificaciones:
        justificaciones.append("Complementa tu perfil profesional actual y te permitirá seguir creciendo")
    
    return " • ".join(justificaciones)

# ========== SISTEMA DE RECOMENDACIÓN DIVERSIFICADO ==========
def recomendar_cursos_diversificado(id_docente, contexto_personal="", num_recomendaciones=5,
                                   filtro_tiempo=None, filtro_modalidad=None):
    """
    Sistema de recomendación DIVERSIFICADO que garantiza cobertura de competencias débiles
    """
    if not model:
        return "Error: Modelo de IA no disponible"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Obtener puntuaciones del docente
        cursor.execute(f"""
            SELECT {', '.join(COMPETENCIAS_EVALUADAS)}, contexto_personal
            FROM Evaluaciones 
            WHERE id_docente = ?
            ORDER BY fecha_evaluacion DESC
            LIMIT 1
        """, (id_docente,))
        
        result = cursor.fetchone()
        if not result:
            return "Sin perfil registrado."
        
        scores = result[:-1]
        contexto_bd = result[-1] or ""
        contexto_final = contexto_personal or contexto_bd
        
        # Obtener cursos con filtros
        query = """
            SELECT nombre_curso, horas, descripcion, competencias_clave,
                   modalidad, nivel_dificultad, clave_registrada
            FROM Cursos WHERE 1=1
        """
        params = []
        
        if filtro_tiempo:
            query += " AND horas <= ?"
            params.append(filtro_tiempo)
        
        if filtro_modalidad and filtro_modalidad != "Todas":
            query += " AND modalidad = ?"
            params.append(filtro_modalidad)
        
        cursor.execute(query, params)
        cursos = cursor.fetchall()
        conn.close()
        
        if not cursos:
            return []
        
        # PASO 1: IDENTIFICAR COMPETENCIAS DÉBILES (≤ 2)
        competencias_debiles = []
        for i, (comp, score) in enumerate(zip(COMPETENCIAS_EVALUADAS, scores)):
            if score <= 2:
                competencias_debiles.append({
                    'competencia': comp,
                    'score': score,
                    'prioridad': 3 - score  # Más débil = mayor prioridad
                })
        
        # Ordenar por prioridad (más débiles primero)
        competencias_debiles.sort(key=lambda x: x['prioridad'], reverse=True)
        
        logger.info(f"Competencias débiles identificadas: {len(competencias_debiles)}")
        
        # PASO 2: ANÁLISIS DEL CONTEXTO PERSONAL
        insights_contexto = extraer_insights_contexto(contexto_final, model)
        
        # PASO 3: GENERAR EMBEDDINGS Y SCORES BASE
        curso_textos = []
        for curso in cursos:
            texto_curso = f"""
Curso: {curso[0]}
Descripción: {curso[2]}
Competencias: {curso[3]}
Modalidad: {curso[4]}
Nivel: {curso[5]}
Duración: {curso[1]} horas
"""
            curso_textos.append(texto_curso.strip())
        
        curso_embeddings = model.encode(curso_textos)
        
        # Generar perfil simplificado (MÁS FOCUSO)
        perfil_prompt = generar_perfil_focusado(scores, contexto_final, competencias_debiles)
        perfil_embedding = model.encode([perfil_prompt])[0].reshape(1, -1)
        
        # Similitudes base
        sim_scores = cosine_similarity(perfil_embedding, curso_embeddings).flatten()
        
        # PASO 4: ALGORITMO DE DIVERSIFICACIÓN FORZADA
        recomendaciones_finales = []
        cursos_seleccionados = set()
        competencias_cubiertas = set()
        
        # FASE A: Una recomendación por competencia débil (máximo)
        for comp_info in competencias_debiles:
            if len(recomendaciones_finales) >= num_recomendaciones:
                break
                
            competencia = comp_info['competencia']
            mejor_curso_idx = None
            mejor_score = -1
            
            # Buscar el MEJOR curso para esta competencia específica
            for i, curso in enumerate(cursos):
                if i in cursos_seleccionados:
                    continue
                    
                competencias_curso = curso[3].split(',') if curso[3] else []
                competencias_curso = [c.strip() for c in competencias_curso]
                
                # Solo cursos que atienden esta competencia
                if competencia not in competencias_curso:
                    continue
                
                # Calcular score mejorado para esta competencia
                score_final = calcular_score_competencia_especifica(
                    curso, i, sim_scores[i], scores, competencia, insights_contexto
                )
                
                if score_final > mejor_score:
                    mejor_score = score_final
                    mejor_curso_idx = i
            
            # Agregar el mejor curso para esta competencia
            if mejor_curso_idx is not None:
                curso_data = crear_objeto_recomendacion(
                    cursos[mejor_curso_idx], mejor_score, scores, insights_contexto, competencia
                )
                recomendaciones_finales.append(curso_data)
                cursos_seleccionados.add(mejor_curso_idx)
                competencias_cubiertas.add(competencia)
                
                logger.info(f"Seleccionado para {competencia}: {cursos[mejor_curso_idx][0]}")
        
        # FASE B: Completar con mejores cursos generales (sin repetir competencias ya cubiertas)
        scores_restantes = []
        indices_restantes = []
        
        for i, curso in enumerate(cursos):
            if i in cursos_seleccionados:
                continue
                
            # Preferir cursos que cubran competencias débiles no cubiertas aún
            competencias_curso = curso[3].split(',') if curso[3] else []
            competencias_curso = [c.strip() for c in competencias_curso]
            
            # Bonus por cubrir competencias débiles no atendidas
            bonus_nuevas_competencias = 0
            for comp_info in competencias_debiles:
                if comp_info['competencia'] not in competencias_cubiertas:
                    if comp_info['competencia'] in competencias_curso:
                        bonus_nuevas_competencias += 0.8 * comp_info['prioridad']
            
            score_final = sim_scores[i] + bonus_nuevas_competencias
            scores_restantes.append(score_final)
            indices_restantes.append(i)
        
        # Seleccionar los mejores restantes
        if indices_restantes and len(recomendaciones_finales) < num_recomendaciones:
            restantes_ordenados = sorted(
                zip(scores_restantes, indices_restantes), 
                key=lambda x: x[0], 
                reverse=True
            )
            
            for score, idx in restantes_ordenados:
                if len(recomendaciones_finales) >= num_recomendaciones:
                    break
                
                curso_data = crear_objeto_recomendacion(
                    cursos[idx], score, scores, insights_contexto, "general"
                )
                recomendaciones_finales.append(curso_data)
        
        logger.info(f"Recomendaciones diversificadas generadas: {len(recomendaciones_finales)}")
        logger.info(f"Competencias cubiertas: {len(competencias_cubiertas)}/{len(competencias_debiles)}")
        
        return recomendaciones_finales
        
    except Exception as e:
        logger.error(f"Error en recomendación diversificada: {e}")
        return f"Error al generar recomendaciones: {str(e)}"


def generar_perfil_focusado(scores, contexto_personal, competencias_debiles):
    """Genera un perfil más focuseado para mejorar la precisión de embeddings"""
    
    prompt_parts = ["=== PERFIL DOCENTE PARA RECOMENDACIONES ==="]
    
    # Contexto personal (condensado)
    if contexto_personal and contexto_personal.strip():
        prompt_parts.append(f"CONTEXTO: {contexto_personal[:200]}...")
    
    # Competencias prioritarias
    if competencias_debiles:
        prompt_parts.append("COMPETENCIAS PRIORITARIAS A DESARROLLAR:")
        for comp_info in competencias_debiles[:5]:  # Solo top 5
            comp = comp_info['competencia']
            titulo = RUBRICA_DATA[comp]["titulo"]
            score = comp_info['score']
            prompt_parts.append(f"• {titulo} (Nivel actual: {score}/4)")
    
    # Tags de competencias débiles
    tags_prioritarios = []
    for comp_info in competencias_debiles:
        comp = comp_info['competencia']
        tags = RUBRICA_DATA[comp]["tags"].split(', ')
        tags_prioritarios.extend(tags)
    
    if tags_prioritarios:
        prompt_parts.append(f"PALABRAS CLAVE PRIORITARIAS: {', '.join(set(tags_prioritarios))}")
    
    return "\n".join(prompt_parts)


def calcular_score_competencia_especifica(curso, indice, sim_base, scores, competencia_objetivo, insights_contexto):
    """Calcula score específico para una competencia objetivo"""
    
    score_final = sim_base  # Base de similitud semántica
    
    # BONUS 1: Competencia objetivo (PESO ALTO)
    comp_index = COMPETENCIAS_EVALUADAS.index(competencia_objetivo)
    score_competencia = scores[comp_index]
    
    if score_competencia <= 2:
        bonus_competencia = 1.0 * (3 - score_competencia)  # 1.0 a 2.0 de bonus
        score_final += bonus_competencia
    
    # BONUS 2: Contexto personal
    modalidad_curso = curso[4]
    duracion_curso = curso[1]
    nivel_curso = curso[5]
    
    bonus_contexto = 0
    if insights_contexto:
        # Modalidad preferida
        if 'modalidad' in insights_contexto:
            if any(mod in modalidad_curso.lower() for mod in insights_contexto['modalidad']):
                bonus_contexto += 0.3
        
        # Experiencia vs nivel del curso
        if 'experiencia' in insights_contexto:
            exp_keywords = insights_contexto['experiencia']
            if any(kw in ['nuevo', 'novato'] for kw in exp_keywords) and nivel_curso <= 2:
                bonus_contexto += 0.2
            elif any(kw in ['veterano', 'experimentado'] for kw in exp_keywords) and nivel_curso >= 3:
                bonus_contexto += 0.2
        
        # Tecnología
        if 'recursos_tecnologicos' in insights_contexto and len(insights_contexto['recursos_tecnologicos']) > 0:
            if modalidad_curso in ['Virtual', 'Mixto']:
                bonus_contexto += 0.15
    
    score_final += bonus_contexto
    
    # BONUS 3: Ajuste por duración (preferir cursos más largos para competencias muy débiles)
    if score_competencia == 1 and duracion_curso >= 40:
        score_final += 0.1
    
    return score_final


def crear_objeto_recomendacion(curso, score, scores_docente, insights_contexto, competencia_foco):
    """Crea el objeto de recomendación con justificación mejorada"""
    
    # Generar justificación específica
    justificacion_parts = []
    
    if competencia_foco != "general":
        comp_titulo = RUBRICA_DATA[competencia_foco]["titulo"]
        comp_index = COMPETENCIAS_EVALUADAS.index(competencia_foco)
        score_comp = scores_docente[comp_index]
        justificacion_parts.append(f"Fortalecerá específicamente tu '{comp_titulo}' (actual nivel {score_comp}/4)")
    
    # Justificación por contexto
    if insights_contexto:
        if 'desafios' in insights_contexto:
            justificacion_parts.append("Aborda desafíos que mencionaste en tu contexto")
        if 'intereses' in insights_contexto:
            justificacion_parts.append("Conecta con tus intereses expresados")
    
    # Justificación por características del curso
    modalidad = curso[4]
    horas = curso[1]
    
    if horas >= 40:
        justificacion_parts.append("Curso intensivo que profundiza en el tema")
    
    if modalidad == "Virtual" and insights_contexto.get('recursos_tecnologicos'):
        justificacion_parts.append("Modalidad virtual acorde a tu interés tecnológico")
    
    if not justificacion_parts:
        justificacion_parts.append("Complementa tu perfil profesional actual")
    
    return {
        "curso": curso[0],
        "horas": curso[1],
        "descripcion": curso[2],
        "competencias_clave": curso[3],
        "modalidad": curso[4],
        "nivel_dificultad": curso[5],
        "clave_registrada": curso[6],
        "similaridad": score,
        "competencia_principal": competencia_foco if competencia_foco != "general" else "Desarrollo integral",
        "justificacion": " • ".join(justificacion_parts)
    }
     
# ---- VISUALIZACIONES ----
def mostrar_radar_competencias(scores):
    """Muestra un gráfico radar del perfil de competencias"""
    competencias_labels = [RUBRICA_DATA[comp]["titulo"].replace(".", "").strip() for comp in COMPETENCIAS_EVALUADAS]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=competencias_labels,
        fill='toself',
        name='Tu Perfil',
        line=dict(color='rgb(1, 87, 155)'),
        fillcolor='rgba(1, 87, 155, 0.25)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 4],
                tickvals=[1, 2, 3, 4],
                ticktext=["Inicial", "En Desarrollo", "Consolidado", "Destacado"]
            )),
        showlegend=True,
        title="Tu Perfil de Competencias Docentes",
        height=500
    )
    
    return fig

def mostrar_analisis_competencias(scores):
    """Muestra análisis detallado de competencias"""
    niveles = ["Inicial", "En Desarrollo", "Consolidado", "Destacado"]
    colores = ["#ff4444", "#ff8800", "#44aa44", "#0088ff"]
    
    datos_competencias = []
    for comp, score in zip(COMPETENCIAS_EVALUADAS, scores):
        datos_competencias.append({
            "Competencia": RUBRICA_DATA[comp]["titulo"],
            "Nivel": niveles[int(score)-1],
            "Puntaje": score,
            "Color": colores[int(score)-1]
        })
    
    df = pd.DataFrame(datos_competencias)
    
    fig = px.bar(
        df, 
        x="Competencia", 
        y="Puntaje",
        color="Nivel",
        color_discrete_map={
            "Inicial": "#ff4444",
            "En Desarrollo": "#ff8800", 
            "Consolidado": "#44aa44",
            "Destacado": "#0088ff"
        },
        title="Análisis Detallado por Competencia"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        yaxis=dict(range=[0, 4.5])
    )
    
    return fig

# AÑADE ESTA NUEVA FUNCIÓN
def mostrar_distribucion_niveles(scores):
    """
    Genera un gráfico de anillo (dona) que muestra la distribución de competencias por nivel.
    """
    # Contamos cuántas competencias hay en cada nivel
    niveles = ["Inicial", "En Desarrollo", "Consolidado", "Destacado"]
    conteo = [
        sum(1 for s in scores if s == 1),
        sum(1 for s in scores if s == 2),
        sum(1 for s in scores if s == 3),
        sum(1 for s in scores if s == 4)
    ]
    
    # Creamos el gráfico de anillo
    fig = go.Figure(data=[go.Pie(
        labels=niveles, 
        values=conteo, 
        hole=.4, # Esto crea el agujero en el centro
        marker_colors=['#d7191c', '#fdae61', '#a6d96a', '#1a9641'] # Rojo, Naranja, Verde claro, Verde oscuro
    )])
    
    fig.update_layout(
        title_text="Distribución de Competencias por Nivel de Dominio"
    )
    return fig
# ========== FUNCIÓN PARA MOSTRAR ANÁLISIS DE DIVERSIFICACIÓN ==========
def mostrar_analisis_diversificacion(recomendaciones, scores):
    """Muestra análisis de qué tan bien cubren las recomendaciones las competencias débiles"""
    
    if not isinstance(recomendaciones, list) or not recomendaciones:
        return
    
    # Identificar competencias débiles
    competencias_debiles = []
    for i, (comp, score) in enumerate(zip(COMPETENCIAS_EVALUADAS, scores)):
        if score <= 2:
            competencias_debiles.append(comp)
    
    # Verificar cobertura
    competencias_cubiertas = set()
    for rec in recomendaciones:
        competencias_curso = rec.get('competencias_clave', '').split(',')
        for comp in competencias_curso:
            comp = comp.strip()
            if comp in competencias_debiles:
                competencias_cubiertas.add(comp)
    
    # Mostrar métricas
    st.subheader("📊 Análisis de Diversificación")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 Competencias Débiles", 
            len(competencias_debiles),
            help="Competencias con nivel ≤ 2"
        )
    
    with col2:
        st.metric(
            "✅ Competencias Cubiertas", 
            len(competencias_cubiertas),
            help="Competencias débiles atendidas por las recomendaciones"
        )
    
    with col3:
        cobertura = (len(competencias_cubiertas) / len(competencias_debiles) * 100) if competencias_debiles else 100
        st.metric(
            "📈 % Cobertura", 
            f"{cobertura:.0f}%",
            help="Porcentaje de competencias débiles cubiertas"
        )
    
    # Detalles de cobertura
    if competencias_debiles:
        st.write("**🔍 Detalle de Cobertura:**")
        
        for comp in competencias_debiles:
            comp_titulo = RUBRICA_DATA[comp]["titulo"]
            if comp in competencias_cubiertas:
                st.success(f"✅ {comp_titulo}")
            else:
                st.warning(f"⚠️ {comp_titulo} - No cubierta directamente")

####################################################################
# ========== VERSIÓN ACTUALIZADA DE LA PÁGINA DE RECOMENDACIONES ==========
def show_recommendations_page_diversificada():
    """Página de recomendaciones con análisis de diversificación"""
    st.header("📊 Resultados del Diagnóstico y Recomendaciones IA Diversificadas")
    
    recomendaciones = st.session_state.get('recomendaciones', [])
    scores = st.session_state.get('scores', [])
    
    if not isinstance(recomendaciones, list) or not recomendaciones:
        st.warning("No se encontraron recomendaciones.")
        if st.button("🔄 Realizar otro diagnóstico"):
            st.session_state.page = 'diagnostic'
            st.rerun()
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Recomendaciones",
        "📈 Tu Perfil", 
        "📊 Análisis de Diversificación",
        "👨‍🏫 Contexto"
    ])
    
    with tab1:
        st.success("🎉 Recomendaciones DIVERSIFICADAS basadas en análisis avanzado:")
        
        for i, curso in enumerate(recomendaciones):
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns([2.5, 1, 1, 1, 1])
                
                with col1:
                    st.subheader(f"🏆 Recomendación {i+1}")
                    st.write(f"**{curso.get('curso', 'N/A')}**")
                    st.caption(f"Clave: {curso.get('clave_registrada', 'N/A')}")
                
                with col2:
                    st.metric("⏱️ Horas", curso.get('horas', 'N/A'))
                
                with col3:
                    relevancia = curso.get('similaridad', 0)
                    st.metric("📈 Score", f"{relevancia:.2f}")
                
                with col4:
                    modalidad = curso.get('modalidad', 'N/A')
                    emoji_modalidad = {
                        "Presencial": "🏫",
                        "Virtual": "💻", 
                        "Mixto": "🔄"
                    }.get(modalidad, "📚")
                    st.metric(f"{emoji_modalidad} Modalidad", modalidad)
                
                with col5:
                    nivel = curso.get('nivel_dificultad', 1)
                    estrellas = "⭐" * nivel
                    st.metric("🎯 Nivel", estrellas)
                
                # Competencia principal
                comp_principal = curso.get('competencia_principal', 'N/A')
                if comp_principal != 'Desarrollo integral':
                    st.info(f"🎯 **Competencia principal que desarrolla:** {RUBRICA_DATA.get(comp_principal, {}).get('titulo', comp_principal)}")
                
                st.write("**📝 Descripción:**")
                st.write(curso.get('descripcion', 'N/A'))
                
                justificacion = curso.get('justificacion', 'Complementa tu perfil actual.')
                st.success(f"💡 **¿Por qué este curso?** {justificacion}")
    
    with tab2:
        if scores:
            st.subheader("📊 Visualización de tu Perfil Docente")
            
            radar_fig = mostrar_radar_competencias(scores)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            bar_fig = mostrar_analisis_competencias(scores)
            st.plotly_chart(bar_fig, use_container_width=True)

            dist_fig = mostrar_distribucion_niveles(scores)
            st.plotly_chart(dist_fig, use_container_width=True)

    with tab3:
        mostrar_analisis_diversificacion(recomendaciones, scores)
    
    with tab4:
        contexto_personal = st.session_state.get('contexto_personal', '')
        st.subheader("👨‍🏫 Análisis de tu Contexto Personal")
        
        if contexto_personal and contexto_personal.strip():
            st.info(f"**Contexto ingresado:**\n\n{contexto_personal}")
        else:
            st.warning("No se proporcionó contexto personal.")
        
        if st.button("🔄 Realizar otro diagnóstico"):
            st.session_state.page = 'diagnostic'
            st.rerun()

# --- INTERFAZ DE USUARIO ---
if 'page' not in st.session_state:
    st.session_state.page = 'diagnostic'

# ========== 6. INTERFAZ MEJORADA CON CAMPO DE CONTEXTO ==========
def show_diagnostic_page_con_contexto():
    """Página de diagnóstico mejorada con campo de contexto personal"""
    st.header("🎯 Herramienta de Autodiagnóstico Docente PRO")
    st.write("Evalúa tu perfil docente en 10 competencias clave para recibir recomendaciones personalizadas.")
    
    # Actualizar BD para contexto
    actualizar_base_de_datos_contexto()
    
    with st.form("rubric_form_with_context"):
        # INICIALIZAR responses AQUÍ DENTRO DEL FORMULARIO
        responses = {}
        
        # Sección 1: Evaluación de competencias
        st.subheader("📊 Evaluación de Competencias Docentes")
        
        col1, col2 = st.columns(2)
        
        competencias_col1 = COMPETENCIAS_EVALUADAS[:5]
        competencias_col2 = COMPETENCIAS_EVALUADAS[5:]
        
        with col1:
            for key in competencias_col1:
                value = RUBRICA_DATA[key]
                st.subheader(value["titulo"])
                responses[key] = st.radio(
                    "Selecciona tu nivel:",
                    options=value["descriptores"],
                    key=f"radio_{key}",
                    label_visibility="collapsed"
                )
                st.divider()
        
        with col2:
            for key in competencias_col2:
                value = RUBRICA_DATA[key]
                st.subheader(value["titulo"])
                responses[key] = st.radio(
                    "Selecciona tu nivel:",
                    options=value["descriptores"],
                    key=f"radio_{key}",
                    label_visibility="collapsed"
                )
                st.divider()
        
        # Sección 2: CONTEXTO PERSONAL
        st.markdown("---")
        st.subheader("👨‍🏫 Contexto Personal de tu Práctica Docente")
        st.write("*Esta información es opcional pero mejorará significativamente tus recomendaciones*")
        
        contexto_personal = st.text_area(
            label="Cuéntanos sobre tu práctica docente actual:",
            placeholder="""Ejemplo: Soy maestra de primaria de 5º grado, trabajo con grupos de 35 estudiantes.
Me interesa mucho la tecnología pero mi escuela tiene recursos limitados.
Mi mayor desafío es mantener la atención de todos los niños durante las clases.
Me gustaría aprender más sobre evaluación porque siento que solo califico exámenes...""",
            max_chars=600,
            height=120,
            help="""
💡 Incluye información como:
• Nivel educativo en el que enseñas
• Tamaño de tus grupos
• Recursos disponibles
• Principales desafíos
• Intereses específicos
• Contexto de tu escuela
""",
            key="contexto_personal"
        )
        
        # Mostrar contador de caracteres
        if contexto_personal:
            char_count = len(contexto_personal)
            color = "green" if char_count >= 100 else "orange" if char_count >= 50 else "red"
            st.markdown(f"<p style='color: {color}; font-size: 0.8em;'>Caracteres: {char_count}/600 - {'¡Excelente detalle!' if char_count >= 100 else '¡Agrega más detalles para mejores recomendaciones!' if char_count >= 50 else 'Muy breve, considera agregar más información'}</p>", unsafe_allow_html=True)
        
        # Botón de envío
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            submitted = st.form_submit_button(
                "🔍 Analizar Perfil con IA Avanzada",
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            try:
                # Validar datos
                validar_datos_entrada(responses)
                
                with st.spinner("🧠 Analizando tu perfil y contexto con IA avanzada..."):
                    scores = []
                    for key in COMPETENCIAS_EVALUADAS:
                        selected = responses[key]
                        score = RUBRICA_DATA[key]["descriptores"].index(selected) + 1
                        scores.append(score)
                    
                    # Guardar en base de datos CON CONTEXTO
                    id_demo = 1
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    
                    # Actualizar docente
                    cursor.execute("""
                        INSERT OR REPLACE INTO Docentes 
                        (id_docente, nombre_docente, perfil_simulado, contexto_actual)
                        VALUES (?, ?, ?, ?)
                    """, (id_demo, 'Docente Interactivo', 'Autodiagnóstico', contexto_personal))
                    
                    # Insertar evaluación con contexto
                    column_names = ', '.join(COMPETENCIAS_EVALUADAS)
                    placeholders = ', '.join(['?'] * len(COMPETENCIAS_EVALUADAS))
                    sql_insert = f"""
                        INSERT OR REPLACE INTO Evaluaciones 
                        (id_docente, {column_names}, contexto_personal)
                        VALUES (?, {placeholders}, ?)
                    """
                    cursor.execute(sql_insert, [id_demo] + scores + [contexto_personal])
                    
                    conn.commit()
                    conn.close()
                    
                    # Generar recomendaciones CON CONTEXTO
                    st.session_state.recomendaciones = recomendar_cursos_diversificado(
                        id_demo, contexto_personal, 5
                    )
                    st.session_state.scores = scores
                    st.session_state.page = 'recommendations'
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error al procesar el diagnóstico: {str(e)}")
                logger.error(f"Error en diagnóstico con contexto: {e}")


def migrar_datos_existentes():
    """Migra datos existentes para añadir campos de contexto"""
    try:
        actualizar_base_de_datos_contexto()
        st.success("✅ Base de datos actualizada para contexto personal")
    except Exception as e:
        st.error(f"Error en migración: {e}")
        logger.error(f"Error en migración: {e}")

def main():
    """Función principal de la aplicación con contexto personal integrado"""
    st.title("🚀 VECTOR DOCENTE: Motor de Recomendación Docente con IA 🤖")
    st.write("**Versión PRO:** Motor de recomendación semántico con embeddings BERT, análisis de contexto personal y visualización avanzada.")
    
    # Verificar que el modelo esté cargado
    if not model:
        st.error("⚠️ Error: No se pudo cargar el modelo de IA. Por favor, reinicia la aplicación.")
        return
    
    # Botón para migrar datos existentes (solo aparece si hay datos)
    with st.sidebar:
        st.subheader("🔧 Herramientas")
        if st.button("🔄 Actualizar Base de Datos"):
            migrar_datos_existentes()
        
        # Información del sistema
        st.info("""
**🆕 Nuevas características:**
• Análisis de contexto personal
• Recomendaciones más precisas
• Justificaciones personalizadas
• Embeddings semánticos mejorados
""")
    
    st.markdown("---")
    
    # Inicializar base de datos (actualizada)
    inicializar_base_de_datos()
    actualizar_base_de_datos_contexto()
    
    # Navegación entre páginas
    if st.session_state.page == 'diagnostic':
        show_diagnostic_page_con_contexto()  # Nueva función con contexto
    else:
        show_recommendations_page_diversificada()  # Función mejorada para mostrar resultados


if __name__ == "__main__":
    main()