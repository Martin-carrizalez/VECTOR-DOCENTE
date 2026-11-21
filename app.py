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
import re
import hashlib
import pandas as pd

# ---- CONFIGURACIÓN ----
DB_FILE = "sirfoc.db"
st.set_page_config(page_title="VECTOR DOCENTE", layout="wide")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Funciones del Marco de Referencia de Saberes y Conocimientos Docentes
# (MEJOREDU), tal como se aplican en el DNF de Jalisco. Cada competencia de la
# rúbrica declara a cuál corresponde, para que los resultados de esta app puedan
# cruzarse con el diagnóstico estatal en lugar de vivir en una taxonomía aparte.
FUNCIONES_MEJOREDU = {
    "F": "Favorece el desarrollo integral de niñas, niños y adolescentes conforme al marco normativo, bajo el enfoque de equidad, inclusión e interculturalidad.",
    "G": "Gestiona el aprendizaje a través de procesos de enseñanza organizados y sistemáticos, con estrategias y situaciones didácticas contextualizadas.",
    "C": "Colabora y establece alianzas con actores internos y externos para la mejora de los procesos institucionales y de la práctica profesional.",
    "I": "Innova permanentemente su práctica docente como resultado del análisis reflexivo y crítico de su hacer profesional.",
    "R": "Reconoce la trascendencia de su función social a través del fortalecimiento de su identidad profesional como actor autónomo, corresponsable y ético.",
}

# Etiqueta breve de cada función para mostrar en pantalla sin ocupar un párrafo.
FUNCIONES_MEJOREDU_CORTO = {
    "F": "Favorece el desarrollo integral",
    "G": "Gestiona el aprendizaje",
    "C": "Colabora y establece alianzas",
    "I": "Innova su práctica",
    "R": "Reconoce su función social",
}

def etiqueta_funcion(comp_key):
    """Texto 'Función X · nombre corto' de una competencia de la rúbrica."""
    f = RUBRICA_DATA[comp_key]["funcion_mejoredu"]
    return f"Función {f} · {FUNCIONES_MEJOREDU_CORTO[f]}"

# Campos por competencia:
#   funcion_mejoredu -> clave de FUNCIONES_MEJOREDU (correspondencia PROPUESTA,
#                       pendiente de validación por el área académica).
#   observable       -> True si la competencia admite una segunda fuente además
#                       de la autopercepción (observación de aula, evidencia
#                       documental, registro administrativo). Si es False, esta
#                       competencia NUNCA podrá triangularse y su resultado se
#                       queda siempre en percepción declarada.
RUBRICA_DATA = {
    "dominio_curricular": {
        "titulo": "1. Dominio Curricular y Pedagógico",
        "descriptores": [
            "Inicial: Aplico la currícula prescrita y sigo las guías establecidas. Busco comprender los fundamentos y la implementación del Plan de Estudio 2022 y la NEM.",
            "En Desarrollo: Comienzo a realizar adaptaciones a los contenidos curriculares para responder a las necesidades de mi grupo. Consulto diversas fuentes para enriquecer mi entendimiento de la NEM.",
            "Consolidado: Tomo decisiones pedagógicas fundamentadas para contextualizar la currícula, articulando los ejes, campos formativos y contenidos de manera coherente en mi planeación.",
            "Destacado: Lidero el codiseño curricular en mi colectivo, generando propuestas que articulan el Plan de Estudio 2022 con el proyecto escolar."
        ],
        "tags": "currículo, Plan de Estudio 2022, Nueva Escuela Mexicana, pedagogía, contextualización",
        "funcion_mejoredu": "G",
        "observable": True,  # planeación didáctica entregable
    },
    "reflexion_practica": {
        "titulo": "2. Reflexión y Transformación de la Práctica",
        "descriptores": [
            "Inicial: Identifico las problemáticas generales del aula, aunque me centro principalmente en factores externos.",
            "En Desarrollo: Empiezo a analizar cómo mis propias acciones y decisiones influyen en las dinámicas del aula, buscando alternativas para mejorar.",
            "Consolidado: Utilizo de manera sistemática herramientas de reflexión (como bitácoras) para analizar mi práctica y ajustar mis estrategias.",
            "Destacado: Promuevo activamente la reflexión colectiva. Sistematizo y comparto los hallazgos de mi práctica para transformar el quehacer docente del colectivo."
        ],
        "tags": "reflexión, práctica docente, transformación, saberes docentes, problematización, autoanálisis, mejora",
        "funcion_mejoredu": "I",
        "observable": False,  # solo autorreporte: no hay artefacto externo
    },
    "colaboracion_dialogo": {
        "titulo": "3. Colaboración y Diálogo Profesional",
        "descriptores": [
            "Inicial: Mi enfoque de trabajo es principalmente individual, aunque participo en las actividades generales de la escuela.",
            "En Desarrollo: Intercambio ideas y materiales con colegas de confianza y participo en las discusiones del Consejo Técnico Escolar (CTE).",
            "Consolidado: Colaboro activamente en el diseño y ejecución de proyectos colectivos en el CTE, utilizando el diálogo para construir acuerdos.",
            "Destacado: Impulso y lidero comunidades de aprendizaje profesional, creando una cultura de confianza y colaboración."
        ],
        "tags": "colaboración, diálogo, trabajo en equipo, comunidades de aprendizaje, corresponsabilidad",
        "funcion_mejoredu": "C",
        "observable": True,  # actas y productos del colectivo
    },
    "liderazgo_autonomia": {
        "titulo": "4. Liderazgo y Autonomía Profesional",
        "descriptores": [
            "Inicial: Aplico las directrices curriculares e institucionales para asegurar la coherencia del servicio educativo.",
            "En Desarrollo: Propongo adaptaciones a la planeación, argumentando mis decisiones pedagógicas con base en las necesidades de mis estudiantes.",
            "Consolidado: Ejerzo mi autonomía profesional para tomar decisiones curriculares y de gestión que se alinean con el proyecto escolar.",
            "Destacado: Asumo un rol de liderazgo pedagógico, inspirando y coordinando acciones para la mejora continua y fomentando la autonomía de mis colegas."
        ],
        "tags": "autonomía, liderazgo, toma de decisiones, responsabilidad, función directiva",
        "funcion_mejoredu": "R",
        "observable": False,  # solo autorreporte
    },
    "evaluacion_aprendizaje": {
        "titulo": "5. Evaluación para el Aprendizaje",
        "descriptores": [
            "Inicial: Me enfoco en la aplicación de instrumentos para obtener una calificación, identificando principalmente aciertos y errores.",
            "En Desarrollo: Busco incorporar la retroalimentación durante los procesos de aprendizaje para orientar a los estudiantes.",
            "Consolidado: Implemento de forma sistemática la autoevaluación y coevaluación, y ofrezco retroalimentación descriptiva que ayuda a mejorar.",
            "Destacado: He logrado que mis estudiantes se apropien del proceso evaluativo, utilizando la autoevaluación para autorregular su aprendizaje."
        ],
        "tags": "evaluación formativa, retroalimentación, autoevaluación, coevaluación, mejora del aprendizaje, enseñanza",
        "funcion_mejoredu": "G",
        "observable": True,  # instrumentos de evaluación del docente
    },
    "atencion_diversidad": {
        "titulo": "6. Atención a la Diversidad e Inclusión",
        "descriptores": [
            "Inicial: Reconozco la diversidad en mi aula y busco estrategias para atender las necesidades que se presentan.",
            "En Desarrollo: Identifico las principales barreras para el aprendizaje que enfrentan algunos de mis estudiantes y busco información para atenderlas.",
            "Consolidado: Diseño e implemento de manera consistente planeaciones diversificadas y ajustes razonables para todos mis estudiantes.",
            "Destacado: Promuevo activamente una cultura de aula incluyente e intercultural, donde la diversidad es reconocida como una fortaleza."
        ],
        "tags": "diversidad, inclusión, equidad, empatía, respeto, derechos humanos, vulnerabilidad",
        "funcion_mejoredu": "F",
        "observable": True,  # observable en aula
    },
    "gestion_recursos_tic": {
        "titulo": "7. Gestión de Recursos y Tecnologías",
        "descriptores": [
            "Inicial: Utilizo los recursos tecnológicos disponibles principalmente para tareas administrativas o para la exposición de información.",
            "En Desarrollo: Exploro y utilizo diversas herramientas y plataformas digitales como un recurso complementario en mis clases.",
            "Consolidado: Integro intencionadamente recursos tecnológicos en mi planeación para que los estudiantes investiguen, creen y colaboren.",
            "Destacado: Diseño experiencias de aprendizaje donde la tecnología es un mediador pedagógico clave y gestiono creativamente los recursos."
        ],
        "tags": "TIC, tecnología educativa, recursos didácticos, herramientas digitales, plataformas, materiales",
        "funcion_mejoredu": "G",
        "observable": True,  # observable en aula
    },
    "gestion_desarrollo_profesional": {
        "titulo": "8. Gestión del Desarrollo Profesional Continuo",
        "descriptores": [
            "Inicial: Busco cursos y capacitaciones que me ofrezcan respuestas y soluciones claras a las problemáticas que enfrento.",
            "En Desarrollo: Intercambio experiencias con mis colegas y consulto diversas fuentes para mantenerme actualizado.",
            "Consolidado: Identifico de forma autónoma mis necesidades de formación y participo en trayectos que promueven la reflexión.",
            "Destacado: Lidero el diagnóstico de necesidades formativas en mi colectivo y promuevo la participación en trayectos de desarrollo."
        ],
        "tags": "formación continua, desarrollo profesional, necesidades formativas, planificación, monitoreo, evaluación de programas",
        "funcion_mejoredu": "R",
        "observable": True,  # verificable contra el registro de acreditación
    },
    "adaptabilidad_resolucion": {
        "titulo": "9. Adaptabilidad y Resolución de Problemas",
        "descriptores": [
            "Inicial: Enfrento las situaciones complejas priorizando las tareas administrativas y de gestión.",
            "En Desarrollo: Comienzo a problematizar los retos cotidianos, analizando sus posibles causas y explorando alternativas.",
            "Consolidado: Abordo los desafíos de la práctica con una metodología clara, analizando causas y efectos para proponer soluciones.",
            "Destacado: Demuestro una visión estratégica para anticipar y resolver problemas complejos, transformando desafíos en oportunidades."
        ],
        "tags": "adaptabilidad, flexibilidad, resolución de problemas, gestión del cambio, retos, desafíos",
        "funcion_mejoredu": "F",
        "observable": False,  # correspondencia más débil del crosswalk: revisar con el área académica
    },
    "documentacion_sistematizacion": {
        "titulo": "10. Documentación y Sistematización",
        "descriptores": [
            "Inicial: Mi práctica se basa principalmente en la experiencia y el intercambio oral, sin un registro formal.",
            "En Desarrollo: Realizo registros esporádicos sobre experiencias significativas para guiar mi reflexión personal.",
            "Consolidado: Documento de forma sistemática mis planeaciones, proyectos y reflexiones, creando un portafolio profesional.",
            "Destacado: Impulso la cultura de la documentación y sistematización en mi colectivo, generando evidencia para la toma de decisiones."
        ],
        "tags": "documentación, sistematización, registro, evidencia, informes",
        "funcion_mejoredu": "I",
        "observable": True,  # evidencia documental
    }
}

COMPETENCIAS_EVALUADAS = list(RUBRICA_DATA.keys())

# ---- REACTIVO DE CONTROL DE LECTURA ----
# Bloque con la misma apariencia que los 10 de la rúbrica, pero cuya respuesta
# correcta está escrita en el propio enunciado. No bloquea el envío: solo
# registra si la persona leyó. Marcar es mejor que bloquear, porque un filtro
# mal calibrado descarta autodiagnósticos válidos.
#
# La opción correcta es la 2 y no la 4 a propósito: quien va marcando el nivel
# más alto en todo (el patrón dominante cuando alguien se autocalifica) la falla.
CONTROL_TITULO = "Reactivo de control del instrumento"
CONTROL_OPCIONES = [
    "Inicial: Reviso los contenidos del instrumento antes de responderlo.",
    "En Desarrollo: Esta es la opción que pide este reactivo de control. Selecciónela.",
    "Consolidado: Respondo los instrumentos institucionales de forma sistemática.",
    "Destacado: Promuevo entre mis colegas el uso reflexivo de los instrumentos.",
]
CONTROL_CORRECTA = 2   # índice 1-based dentro de CONTROL_OPCIONES

# Posición en la que se intercala dentro de la columna izquierda del formulario.
# A media lista, que es donde cae la atención, no al inicio ni al final.
CONTROL_POSICION = 3

# ---- CATÁLOGOS DE SEGMENTACIÓN ----
# Las categorías son LAS MISMAS que usa la Encuesta de aprendizajes y
# satisfacción. Si se cambian aquí, los dos instrumentos dejan de poder
# cruzarse y la brecha de oferta pierde su valor comparativo.
NIVELES_EDUCATIVOS = [
    "Educación Preescolar", "Educación Primaria", "Secundaria general",
    "Secundaria técnica", "Telesecundaria", "Educación especial",
    "Educación física", "Educación inicial", "Otro",
]
FUNCIONES_DOCENTES = [
    "Docente", "Directiva", "Supervisión", "Asesoría técnica pedagógica", "Otra",
]
CONTEXTOS_ESCUELA = ["Urbano", "Rural"]

# ---- CACHE DEL MODELO IA ----
@st.cache_resource
def load_model():
    """Carga y cachea el modelo de IA para evitar recargas innecesarias"""
    try:
        logger.info("Cargando modelo SentenceTransformer...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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
    
    # El embedding del contexto se calculaba aquí y no lo leía ninguna función:
    # el análisis del contexto es, hoy, coincidencia de palabras clave. Se deja
    # la clave en None para no romper consumidores y se evita el encode inútil.
    insights['embedding_contexto'] = None
    
    return insights

# ---- VALIDACIÓN DE CURP ----
# El formato por expresión regular solo comprueba la forma. El dígito
# verificador comprueba que la CURP exista como tal: una base de datos real da
# ~100% de dígitos válidos y la captura manual da alrededor de 3% de erratas,
# así que la diferencia sirve además como señal de calidad del dato.
CURP_RE = re.compile(r"^[A-ZÑ]{4}\d{6}[HM][A-ZÑ]{5}[A-Z0-9]\d$")
_CURP_DIC = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

def digito_verificador_curp(curp):
    """Devuelve el dígito de control que le corresponde a la CURP, o None."""
    if not curp or len(curp) != 18:
        return None
    suma = 0
    for i, ch in enumerate(curp[:17]):
        pos = _CURP_DIC.find(ch)
        if pos < 0:
            return None
        suma += pos * (18 - i)
    return str((10 - (suma % 10)) % 10)

def validar_curp(curp):
    """(es_valida, mensaje). Acepta minúsculas y espacios: se normaliza antes."""
    c = (curp or "").strip().upper().replace(" ", "")
    if not c:
        return False, "Escribe tu CURP."
    if len(c) != 18:
        return False, f"La CURP tiene 18 caracteres; escribiste {len(c)}."
    if not CURP_RE.match(c):
        return False, "El formato no corresponde a una CURP."
    dv = digito_verificador_curp(c)
    if dv is None or c[17] != dv:
        return False, "El dígito verificador no coincide. Revisa si hay un carácter mal."
    return True, c

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
    # curp es la llave real del docente: es la que permite unir este
    # autodiagnóstico con el registro de acreditación y con la encuesta de
    # satisfacción posterior, que es la única fuente de resultado que existe.
    cursor.execute('''CREATE TABLE IF NOT EXISTS Docentes (
        id_docente INTEGER PRIMARY KEY AUTOINCREMENT,
        curp TEXT UNIQUE,
        nombre_docente TEXT,
        perfil_simulado TEXT,
        fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        contexto_actual TEXT DEFAULT '',
        nivel_educativo TEXT,
        funcion TEXT,
        contexto TEXT
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
        fuente TEXT DEFAULT 'autopercepcion',  -- autopercepcion | observacion | evidencia_producto
        herramientas_deseadas TEXT DEFAULT '',  -- qué pide aprender: demanda prospectiva
        control_respuesta INTEGER,             -- opción marcada en el reactivo (1-4)
        control_ok INTEGER,                    -- 1 si acertó, 0 si no, NULL si no aplicó
        FOREIGN KEY (id_docente) REFERENCES Docentes (id_docente)
    );''')
    
    # ---- REGISTRO DE BRECHA DE OFERTA ----
    # Cada renglón es una necesidad formativa detectada que el catálogo NO pudo
    # atender. Es el insumo que convierte este recomendador en un instrumento de
    # planeación: a los mil diagnósticos, agregado por nivel y función, dice qué
    # cursos hay que diseñar, con evidencia y no por opinión del área.
    cursor.execute('''CREATE TABLE IF NOT EXISTS BrechaOferta (
        id_brecha INTEGER PRIMARY KEY AUTOINCREMENT,
        id_docente INTEGER,
        id_evaluacion INTEGER,
        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        competencia TEXT,            -- clave de RUBRICA_DATA
        funcion_mejoredu TEXT,       -- F/G/C/I/R, para leerlo en el marco del DNF
        nivel_declarado INTEGER,     -- 1-4 que se autoasignó el docente
        nivel_educativo TEXT,
        funcion TEXT,
        contexto TEXT,
        cubierta INTEGER,            -- 1 si el catálogo ofreció curso, 0 si no
        motivo TEXT,                 -- por qué no se cubrió
        control_ok INTEGER,          -- calidad del diagnóstico que originó el registro
        FOREIGN KEY (id_docente) REFERENCES Docentes (id_docente)
    );''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_brecha_comp ON BrechaOferta(competencia)")

    # ---- CACHÉ DE EMBEDDINGS DEL CATÁLOGO ----
    # El texto de un curso no cambia entre diagnósticos, así que vectorizarlo en
    # cada recomendación es trabajo tirado: con mil docentes serían 83 mil encodes
    # de 83 textos idénticos. Se guarda el vector y se recalcula solo si cambia
    # el texto, detectado por hash.
    cursor.execute('''CREATE TABLE IF NOT EXISTS CursoEmbedding (
        clave_registrada TEXT PRIMARY KEY,
        hash_texto TEXT,
        vector BLOB,
        dim INTEGER,
        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    if 'fuente' not in columns:
        try:
            cursor.execute("""
                ALTER TABLE Evaluaciones 
                ADD COLUMN fuente TEXT DEFAULT 'autopercepcion'
            """)
            logger.info("Columna fuente agregada exitosamente")
        except sqlite3.Error as e:
            logger.error(f"Error al agregar columna fuente: {e}")

    for col, tipo in (("control_respuesta", "INTEGER"), ("control_ok", "INTEGER"),
                      ("herramientas_deseadas", "TEXT")):
        if col not in columns:
            try:
                cursor.execute(f"ALTER TABLE Evaluaciones ADD COLUMN {col} {tipo}")
                logger.info(f"Columna {col} agregada exitosamente")
            except sqlite3.Error as e:
                logger.error(f"Error al agregar columna {col}: {e}")
    
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

    # En bases creadas antes de esta versión la CURP no existía. Se agrega sin
    # UNIQUE porque SQLite no permite añadir una restricción UNIQUE con ALTER;
    # la unicidad se garantiza en obtener_o_crear_docente, que consulta primero.
    for col in ("nivel_educativo", "funcion", "contexto"):
        if col not in columns:
            try:
                cursor.execute(f"ALTER TABLE Docentes ADD COLUMN {col} TEXT")
                logger.info(f"Columna {col} agregada a Docentes")
            except sqlite3.Error as e:
                logger.error(f"Error al agregar {col}: {e}")

    # La tabla de brecha puede no existir en bases creadas antes de esta versión.
    cursor.execute('''CREATE TABLE IF NOT EXISTS BrechaOferta (
        id_brecha INTEGER PRIMARY KEY AUTOINCREMENT,
        id_docente INTEGER, id_evaluacion INTEGER,
        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        competencia TEXT, funcion_mejoredu TEXT, nivel_declarado INTEGER,
        nivel_educativo TEXT, funcion TEXT, contexto TEXT,
        cubierta INTEGER, motivo TEXT, control_ok INTEGER
    );''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_brecha_comp ON BrechaOferta(competencia)")
    cursor.execute('''CREATE TABLE IF NOT EXISTS CursoEmbedding (
        clave_registrada TEXT PRIMARY KEY, hash_texto TEXT, vector BLOB, dim INTEGER,
        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );''')

    if 'curp' not in columns:
        try:
            cursor.execute("ALTER TABLE Docentes ADD COLUMN curp TEXT")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_docentes_curp "
                           "ON Docentes(curp) WHERE curp IS NOT NULL")
            logger.info("Columna curp agregada a Docentes")
        except sqlite3.Error as e:
            logger.error(f"Error al agregar curp: {e}")
    
    conn.commit()
    conn.close()

def texto_curso(curso):
    """Texto único que representa a un curso para vectorizarlo.

    Sin encabezados de plantilla: lo que se vectoriza debe ser el contenido del
    curso y nada más. Cualquier texto fijo que compartan todos los cursos los
    acerca entre sí en el espacio vectorial y borra justo lo que los distingue.
    """
    nombre, horas, desc, comps, modalidad, nivel = curso[0], curso[1], curso[2], curso[3], curso[4], curso[5]
    tags = ", ".join(t for c in (comps or "").split(",")
                     if (t := RUBRICA_DATA.get(c.strip(), {}).get("tags", "")))
    return (f"{nombre}. {desc} "
            f"Aborda {tags}. "
            f"Modalidad {modalidad}, {horas} horas.").strip()


def embeddings_catalogo(cursos):
    """Vectores de los cursos, leídos de caché y recalculados solo si cambiaron.

    Devuelve un arreglo alineado con la lista `cursos` que recibe.
    """
    textos = [texto_curso(c) for c in cursos]
    claves = [c[6] for c in cursos]
    hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in textos]

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        guardados = {k: (h, v, d) for k, h, v, d in cursor.execute(
            "SELECT clave_registrada, hash_texto, vector, dim FROM CursoEmbedding")}
        vectores, faltantes = [None] * len(cursos), []
        for i, (k, h) in enumerate(zip(claves, hashes)):
            g = guardados.get(k)
            if g and g[0] == h and g[1]:
                vectores[i] = np.frombuffer(g[1], dtype=np.float32)
            else:
                faltantes.append(i)

        if faltantes:
            logger.info(f"Vectorizando {len(faltantes)} curso(s); "
                        f"{len(cursos)-len(faltantes)} salieron de caché")
            nuevos = model.encode([textos[i] for i in faltantes])
            filas = []
            for j, i in enumerate(faltantes):
                v = np.asarray(nuevos[j], dtype=np.float32)
                vectores[i] = v
                filas.append((claves[i], hashes[i], v.tobytes(), int(v.shape[0])))
            cursor.executemany(
                "INSERT OR REPLACE INTO CursoEmbedding "
                "(clave_registrada, hash_texto, vector, dim) VALUES (?,?,?,?)", filas)
            conn.commit()
        else:
            logger.info(f"{len(cursos)} curso(s) servidos desde caché, sin vectorizar")
        return np.vstack(vectores)
    finally:
        conn.close()


def obtener_o_crear_docente(curp, contexto="", nivel_educativo=None, funcion=None,
                            contexto_escuela=None):
    """Devuelve el id_docente de esa CURP, creándolo si es la primera vez.

    La CURP es la llave: dos autodiagnósticos de la misma persona caen en el
    mismo id_docente y quedan como dos renglones de Evaluaciones con fechas
    distintas, que es lo que permite ver su trayectoria.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id_docente FROM Docentes WHERE curp = ?", (curp,))
        fila = cursor.fetchone()
        if fila:
            cursor.execute("""UPDATE Docentes SET contexto_actual = ?, nivel_educativo = ?,
                              funcion = ?, contexto = ? WHERE id_docente = ?""",
                           (contexto, nivel_educativo, funcion, contexto_escuela, fila[0]))
            conn.commit()
            return fila[0]
        cursor.execute("""INSERT INTO Docentes (curp, nombre_docente, perfil_simulado,
                          contexto_actual, nivel_educativo, funcion, contexto)
                          VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (curp, f"Docente {curp[:4]}", "Autodiagnóstico", contexto,
                        nivel_educativo, funcion, contexto_escuela))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def registrar_brecha(id_docente, competencias_debiles, competencias_cubiertas,
                     control_ok=None, id_evaluacion=None):
    """Guarda qué necesidades formativas detectadas pudo y no pudo atender el catálogo.

    Se escribe SIEMPRE, cubiertas y no cubiertas, para poder calcular la tasa de
    cobertura. Sin las cubiertas el denominador no existe y el reporte diría
    'faltan N cursos' sin decir de cuántos.
    """
    if not competencias_debiles:
        return 0
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        seg = cursor.execute(
            "SELECT nivel_educativo, funcion, contexto FROM Docentes WHERE id_docente = ?",
            (id_docente,)).fetchone() or (None, None, None)
        filas = []
        for info in competencias_debiles:
            comp = info["competencia"]
            cubierta = 1 if comp in competencias_cubiertas else 0
            # El motivo distingue las dos causas de no cobertura, que piden
            # decisiones distintas: diseñar un curso nuevo, o ampliar el cupo
            # y la modalidad de los que ya existen.
            n_cursos = _cursos_para_competencia(cursor, comp)
            motivo = None if cubierta else (
                f"el catálogo casi no tiene oferta para esta competencia ({n_cursos})"
                if n_cursos <= 2
                else "hay cursos, pero ninguno quedó entre las recomendaciones")
            filas.append((id_docente, id_evaluacion, comp,
                          RUBRICA_DATA[comp]["funcion_mejoredu"], info["score"],
                          seg[0], seg[1], seg[2], cubierta, motivo, control_ok))
        cursor.executemany("""INSERT INTO BrechaOferta
            (id_docente, id_evaluacion, competencia, funcion_mejoredu, nivel_declarado,
             nivel_educativo, funcion, contexto, cubierta, motivo, control_ok)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""", filas)
        conn.commit()
        logger.info(f"Brecha registrada: {len(filas)} necesidades, "
                    f"{sum(1 for f in filas if f[8]==0)} sin cubrir")
        return len(filas)
    except sqlite3.Error as e:
        logger.error(f"Error al registrar brecha: {e}")
        return 0
    finally:
        conn.close()


def _cursos_para_competencia(cursor, competencia):
    """Cuántos cursos del catálogo declaran esa competencia."""
    return cursor.execute(
        "SELECT COUNT(*) FROM Cursos WHERE competencias_clave LIKE ?",
        (f"%{competencia}%",)).fetchone()[0]


def resumen_brecha(solo_control_ok=False):
    """Agregado por competencia: demanda detectada, cubierta y sin cubrir.

    solo_control_ok limita el reporte a los diagnósticos que superaron el
    reactivo de control, que es como debe presentarse a un área que va a tomar
    decisiones de presupuesto con esto.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        filtro = " WHERE control_ok = 1" if solo_control_ok else ""
        filas = conn.execute(f"""
            SELECT competencia, funcion_mejoredu,
                   COUNT(*) AS detectada,
                   SUM(CASE WHEN cubierta = 0 THEN 1 ELSE 0 END) AS sin_cubrir,
                   AVG(nivel_declarado) AS nivel_medio
            FROM BrechaOferta{filtro}
            GROUP BY competencia ORDER BY sin_cubrir DESC, detectada DESC""").fetchall()
        cursor = conn.cursor()
        return [dict(competencia=f[0], titulo=RUBRICA_DATA.get(f[0], {}).get("titulo", f[0]),
                     funcion=f[1], detectada=f[2], sin_cubrir=f[3] or 0,
                     nivel_medio=round(f[4] or 0, 2),
                     cursos_catalogo=_cursos_para_competencia(cursor, f[0]))
                for f in filas]
    finally:
        conn.close()


def brecha_por_segmento(dimension="nivel_educativo", solo_control_ok=False):
    """Matriz competencia x segmento con el conteo de necesidades sin cubrir.

    Es la vista que responde la pregunta de planeación: qué hay que diseñar y
    para quién. dimension debe ser una columna de BrechaOferta.
    """
    if dimension not in ("nivel_educativo", "funcion", "contexto"):
        raise ValueError("dimensión no permitida")
    conn = sqlite3.connect(DB_FILE)
    try:
        filtro = " AND control_ok = 1" if solo_control_ok else ""
        filas = conn.execute(f"""
            SELECT competencia, {dimension}, COUNT(*)
            FROM BrechaOferta
            WHERE cubierta = 0 AND {dimension} IS NOT NULL{filtro}
            GROUP BY competencia, {dimension}""").fetchall()
        if not filas:
            return pd.DataFrame()
        d = pd.DataFrame(filas, columns=["competencia", dimension, "sin_cubrir"])
        d["competencia"] = d["competencia"].map(
            lambda c: RUBRICA_DATA.get(c, {}).get("titulo", c))
        return d.pivot_table(index="competencia", columns=dimension,
                             values="sin_cubrir", fill_value=0, aggfunc="sum")
    finally:
        conn.close()


def demanda_declarada(dimension=None, solo_control_ok=False, min_menciones=3):
    """Qué dicen los docentes que quieren aprender, agregado por términos.

    Es la contraparte prospectiva de BrechaOferta: la brecha dice qué necesidad
    detectada no se pudo atender con el catálogo; esto dice qué pide la gente por
    su cuenta, incluso si no corresponde a ninguna competencia de la rúbrica.
    Ahí es donde aparecen los temas que todavía no están en el marco.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        cols = "e.herramientas_deseadas" + (f", d.{dimension}" if dimension else "")
        filtro = " AND e.control_ok = 1" if solo_control_ok else ""
        filas = conn.execute(
            f"""SELECT {cols} FROM Evaluaciones e
                JOIN Docentes d ON d.id_docente = e.id_docente
                WHERE e.herramientas_deseadas IS NOT NULL
                  AND LENGTH(TRIM(e.herramientas_deseadas)) > 10{filtro}""").fetchall()
    finally:
        conn.close()
    if not filas:
        return pd.DataFrame()

    # Stopwords mínimas y en español. Se deja fuera el vocabulario de relleno
    # que aparece en toda respuesta y no distingue a un docente de otro.
    STOP = set("""de la el los las un una unos unas y o a en con para por que se su sus
        me mi mis lo al del es son como mas más sobre todo todos poder quiero quisiera
        gustaria gustaría aprender saber manera forma cosas tema temas parte bien mejor
        muy ser estar tener hacer puedo pueda sean sea desde hasta cuando donde porque
        este esta estos estas ese esa eso nos nuestro practica práctica docente""".split())
    def tokens(t):
        t = re.sub(r"[^\wáéíóúñü ]", " ", str(t).lower())
        return [w for w in t.split() if len(w) > 3 and w not in STOP]

    registros = []
    for fila in filas:
        texto = fila[0]
        seg = fila[1] if dimension and len(fila) > 1 else "Todos"
        tok = tokens(texto)
        bigramas = [" ".join(tok[i:i+2]) for i in range(len(tok)-1)]
        for term in set(tok) | set(bigramas):
            registros.append({"termino": term, "segmento": seg or "Sin dato"})
    if not registros:
        return pd.DataFrame()
    d = pd.DataFrame(registros)
    g = (d.groupby(["termino", "segmento"]).size().reset_index(name="menciones")
           if dimension else
         d.groupby("termino").size().reset_index(name="menciones"))
    return g[g["menciones"] >= min_menciones].sort_values("menciones", ascending=False)


def historial_docente(id_docente):
    """Evaluaciones previas de ese docente, de la más reciente a la más antigua."""
    conn = sqlite3.connect(DB_FILE)
    try:
        cols = ", ".join(COMPETENCIAS_EVALUADAS)
        return conn.execute(
            f"SELECT fecha_evaluacion, {cols}, control_ok FROM Evaluaciones "
            f"WHERE id_docente = ? ORDER BY fecha_evaluacion DESC", (id_docente,)).fetchall()
    finally:
        conn.close()


# NOTA: aquí vivían generar_prompt_perfil_con_contexto y
# generar_justificacion_con_contexto. Ninguna se llamaba desde ningún lado
# (la ruta viva usa generar_perfil_focusado) y la primera hacía
# ', '.join(tags) sobre un string, partiéndolo letra por letra.

# ========== SISTEMA DE RECOMENDACIÓN DIVERSIFICADO ==========
def recomendar_cursos_diversificado(id_docente, contexto_personal="", num_recomendaciones=5,
                                   filtro_tiempo=None, filtro_modalidad=None,
                                   herramientas_deseadas=""):
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
            SELECT {', '.join(COMPETENCIAS_EVALUADAS)}, contexto_personal, herramientas_deseadas
            FROM Evaluaciones 
            WHERE id_docente = ?
            ORDER BY fecha_evaluacion DESC
            LIMIT 1
        """, (id_docente,))
        
        result = cursor.fetchone()
        if not result:
            return "Sin perfil registrado."
        
        scores = result[:-2]
        contexto_bd = result[-2] or ""
        herramientas_bd = result[-1] or ""
        contexto_final = contexto_personal or contexto_bd
        herramientas_deseadas = herramientas_deseadas or herramientas_bd
        
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
        
        # PASO 3: EMBEDDINGS DEL CATÁLOGO (desde caché) Y SCORES BASE
        curso_embeddings = embeddings_catalogo(cursos)

        # Generar perfil del docente para compararlo contra el catálogo
        perfil_prompt = generar_perfil_focusado(scores, contexto_final, competencias_debiles,
                                                herramientas_deseadas)
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

        # Lo que antes solo se escribía en el log y se perdía: qué necesidades
        # quedaron sin atender. Se devuelve para que quien llame pueda
        # registrarlas en BrechaOferta.
        st.session_state['_debiles'] = competencias_debiles
        st.session_state['_cubiertas'] = competencias_cubiertas

        return recomendaciones_finales
        
    except Exception as e:
        logger.error(f"Error en recomendación diversificada: {e}")
        return f"Error al generar recomendaciones: {str(e)}"


def generar_perfil_focusado(scores, contexto_personal, competencias_debiles,
                            herramientas_deseadas=""):
    """Texto del docente que se compara contra el catálogo.

    Tres decisiones deliberadas, todas por la misma razón: el modelo es un
    sentence-transformer, entrenado con oraciones y no con fichas.

    1. Sin encabezados fijos. Un '=== PERFIL DOCENTE ===' idéntico en todos los
       perfiles acerca entre sí los vectores de docentes distintos y borra las
       diferencias que queremos medir.
    2. El contexto va completo. Antes se cortaba en 200 caracteres sobre un campo
       de 600: se tiraban dos tercios de lo único que el docente escribió.
    3. Lo que el docente dice que quiere aprender va PRIMERO. Es una declaración
       de necesidad, más informativa para elegir un curso que el nivel que se
       autoasignó en una rúbrica.
    """
    partes = []

    if herramientas_deseadas and herramientas_deseadas.strip():
        partes.append(f"Quiero aprender {herramientas_deseadas.strip()}")

    if contexto_personal and contexto_personal.strip():
        partes.append(contexto_personal.strip())

    if competencias_debiles:
        titulos = [RUBRICA_DATA[c["competencia"]]["titulo"].split(". ", 1)[-1].lower()
                   for c in competencias_debiles[:5]]
        partes.append("Necesito fortalecer " + ", ".join(titulos))

        # Los tags, como frase y no como lista separada por comas: el modelo
        # embebe mejor una oración que un volcado de keywords.
        tags = []
        for c in competencias_debiles:
            for t in RUBRICA_DATA[c["competencia"]]["tags"].split(", "):
                if t not in tags:
                    tags.append(t)
        if tags:
            partes.append("Me interesan temas de " + ", ".join(tags))

    return ". ".join(partes) + "." if partes else "Docente de educación básica."


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
def mostrar_lectura_mejoredu(scores):
    """
    Reagrupa las 10 competencias de la rúbrica en las 5 funciones del marco
    MEJOREDU (F, G, C, I, R). Es la vista que permite leer el resultado de la
    app en el mismo idioma que el Diagnóstico de Necesidades de Formación.
    """
    st.subheader("🧭 Tu autopercepción leída desde el Marco de referencia MEJOREDU")
    st.caption("Las 10 competencias de este autodiagnóstico se agrupan en las 5 funciones "
               "del marco que utiliza el Diagnóstico de Necesidades de Formación estatal. "
               "La correspondencia es una propuesta sujeta a validación académica.")

    niveles = ["Inicial", "En Desarrollo", "Consolidado", "Destacado"]
    filas = []
    for f, corto in FUNCIONES_MEJOREDU_CORTO.items():
        comps = [c for c in COMPETENCIAS_EVALUADAS if RUBRICA_DATA[c]["funcion_mejoredu"] == f]
        if not comps:
            # Una función sin competencias es un hueco de cobertura del instrumento.
            filas.append({"Función": f"{f} · {corto}", "Nivel promedio declarado": None,
                          "Competencias que la cubren": "— sin cobertura —",
                          "Admiten 2ª fuente": "0 de 0"})
            continue
        vals = [scores[COMPETENCIAS_EVALUADAS.index(c)] for c in comps]
        filas.append({
            "Función": f"{f} · {corto}",
            "Nivel promedio declarado": round(sum(vals) / len(vals), 2),
            "Competencias que la cubren": ", ".join(
                f"{RUBRICA_DATA[c]['titulo']} ({niveles[int(scores[COMPETENCIAS_EVALUADAS.index(c)]) - 1]})"
                for c in comps),
            "Admiten 2ª fuente": f"{sum(RUBRICA_DATA[c]['observable'] for c in comps)} de {len(comps)}",
        })
    df = pd.DataFrame(filas)

    graf = df.dropna(subset=["Nivel promedio declarado"])
    fig = px.bar(graf, x="Función", y="Nivel promedio declarado",
                 text="Nivel promedio declarado",
                 title="Nivel promedio declarado por función del marco")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, yaxis=dict(range=[0, 4.5], title="Nivel declarado (1–4)"),
                      xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Funciones con una sola competencia: su promedio depende de una respuesta.
    delgadas = [f"{f} · {corto}" for f, corto in FUNCIONES_MEJOREDU_CORTO.items()
                if sum(RUBRICA_DATA[c]["funcion_mejoredu"] == f for c in COMPETENCIAS_EVALUADAS) == 1]
    if delgadas:
        st.warning("⚠️ " + "; ".join(delgadas) + " — se estima con una sola competencia "
                   "de la rúbrica, por lo que su lectura es menos robusta que la del resto.")


def mostrar_radar_competencias(scores):
    """Muestra un gráfico radar del perfil de competencias"""
    competencias_labels = [RUBRICA_DATA[comp]["titulo"].replace(".", "").strip() for comp in COMPETENCIAS_EVALUADAS]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=competencias_labels,
        fill='toself',
        name='Autopercepción',
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
        title="Tu autopercepción de desempeño docente",
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
        title="Análisis detallado por competencia (nivel declarado)"
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
        title_text="Distribución de competencias por nivel declarado"
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

    # Lectura del reactivo de control. Va arriba de todo porque condiciona cómo
    # debe leerse el resto del perfil, no es un detalle al pie.
    control_ok = st.session_state.get('control_ok')
    if control_ok == 0:
        st.warning(
            "⚠️ **El reactivo de control no se contestó como pedía el enunciado.** "
            "El perfil que sigue es válido como autopercepción, pero conviene "
            "tomarlo con reserva: en instrumentos de autocalificación, fallar el "
            "control se asocia con marcar la misma opción en todos los reactivos. "
            "Si fue un descuido, puedes repetir el diagnóstico."
        )
    elif control_ok == 1:
        st.caption("✅ Reactivo de control superado.")

    n_previas = st.session_state.get('n_previas', 0)
    if n_previas:
        st.info(f"📈 Esta CURP tiene {n_previas} autodiagnóstico(s) anterior(es) registrado(s). "
                "Se conservan todos para poder comparar la trayectoria.")

    if not isinstance(recomendaciones, list) or not recomendaciones:
        st.warning("No se encontraron recomendaciones.")
        if st.button("🔄 Realizar otro diagnóstico"):
            st.session_state.page = 'diagnostic'
            st.rerun()
        return
    
    tab1, tab_mej, tab2, tab3, tab4 = st.tabs([
        "🎯 Recomendaciones",
        "🧭 Marco MEJOREDU",
        "📈 Tu autopercepción", 
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
                    st.info(f"🎯 **Competencia principal que desarrolla:** {RUBRICA_DATA.get(comp_principal, {}).get('titulo', comp_principal)}"
                            f"  \n🧭 **Marco MEJOREDU:** {etiqueta_funcion(comp_principal)}")
                
                st.write("**📝 Descripción:**")
                st.write(curso.get('descripcion', 'N/A'))
                
                justificacion = curso.get('justificacion', 'Complementa tu perfil actual.')
                st.success(f"💡 **¿Por qué este curso?** {justificacion}")
    
    with tab_mej:
        if scores:
            mostrar_lectura_mejoredu(scores)

    with tab2:
        if scores:
            st.subheader("📊 Visualización de tu autopercepción")
            
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

def show_brecha_page():
    """Vista de planeación: qué necesidades formativas no cubre el catálogo."""
    st.header("🧩 Brecha de oferta formativa")
    st.write("Necesidades detectadas en los autodiagnósticos que el catálogo actual "
             "no pudo atender. Es el insumo para decidir qué cursos diseñar.")

    solo_ok = st.toggle(
        "Considerar solo diagnósticos que superaron el reactivo de control",
        value=True,
        help="Recomendado para decisiones de presupuesto: excluye los autodiagnósticos "
             "contestados sin leer.")

    resumen = resumen_brecha(solo_control_ok=solo_ok)
    if not resumen:
        st.info("Todavía no hay registros de brecha. Se generan con cada autodiagnóstico "
                "que detecte al menos una competencia débil.")
        return

    total_det = sum(r["detectada"] for r in resumen)
    total_sin = sum(r["sin_cubrir"] for r in resumen)
    c1, c2, c3 = st.columns(3)
    c1.metric("Necesidades detectadas", f"{total_det:,}")
    c2.metric("Sin cubrir", f"{total_sin:,}")
    c3.metric("Cobertura del catálogo",
              f"{100*(total_det-total_sin)/total_det:.0f}%" if total_det else "—")

    d = pd.DataFrame(resumen)
    d["% sin cubrir"] = (100 * d["sin_cubrir"] / d["detectada"]).round(1)
    d = d.rename(columns={"titulo": "Competencia", "detectada": "Detectada",
                          "sin_cubrir": "Sin cubrir", "nivel_medio": "Nivel declarado medio",
                          "cursos_catalogo": "Cursos en catálogo"})

    fig = px.bar(d.sort_values("Sin cubrir"), x="Sin cubrir", y="Competencia",
                 orientation="h", text="Sin cubrir",
                 title="Necesidades sin cubrir por competencia")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=430, yaxis_title=None, xaxis_title="Docentes sin oferta")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(d[["Competencia", "Detectada", "Sin cubrir", "% sin cubrir",
                    "Cursos en catálogo", "Nivel declarado medio"]],
                 use_container_width=True, hide_index=True)

    # Una competencia con demanda alta y catálogo vacío es una propuesta de curso
    # nuevo. Una con catálogo amplio y demanda sin cubrir es un problema de cupo
    # o de modalidad, no de diseño: son dos decisiones distintas.
    disenar = d[(d["Sin cubrir"] > 0) & (d["Cursos en catálogo"] <= 2)]
    ampliar = d[(d["Sin cubrir"] > 0) & (d["Cursos en catálogo"] > 2)]
    if not disenar.empty:
        st.error("**Diseñar oferta nueva** — hay demanda y el catálogo casi no tiene cursos:\n\n"
                 + "\n".join(f"- {r['Competencia']} — {int(r['Sin cubrir'])} docentes, "
                              f"{int(r['Cursos en catálogo'])} curso(s) en catálogo"
                              for _, r in disenar.iterrows()))
    if not ampliar.empty:
        st.warning("**Revisar cupo o modalidad** — hay cursos, pero no llegaron a estos docentes:\n\n"
                   + "\n".join(f"- {r['Competencia']} — {int(r['Sin cubrir'])} docentes, "
                                f"{int(r['Cursos en catálogo'])} cursos disponibles"
                                for _, r in ampliar.iterrows()))

    st.subheader("Para quién hay que diseñarla")
    dim = st.radio("Desagregar por:",
                   ["nivel_educativo", "funcion", "contexto"],
                   format_func=lambda x: {"nivel_educativo": "Nivel educativo",
                                          "funcion": "Función",
                                          "contexto": "Contexto de la escuela"}[x],
                   horizontal=True)
    m = brecha_por_segmento(dim, solo_control_ok=solo_ok)
    if m.empty:
        st.info("Aún no hay suficientes registros con ese dato capturado.")
    else:
        st.dataframe(m, use_container_width=True)
        st.caption("Cada celda es el número de docentes de ese segmento cuya necesidad "
                   "en esa competencia quedó sin atender.")

    # ---- DEMANDA DECLARADA ----
    # La brecha de arriba se calcula contra la rúbrica, así que solo puede
    # detectar necesidades que la rúbrica ya contempla. Esto no: aquí aparece lo
    # que los docentes piden con sus palabras, incluso si no corresponde a
    # ninguna de las 10 competencias. Es donde se ven los temas emergentes.
    st.markdown("---")
    st.subheader("🧰 Lo que los docentes piden aprender, con sus palabras")
    dd = demanda_declarada(solo_control_ok=solo_ok)
    if dd.empty:
        st.info("Aún no hay respuestas suficientes a la pregunta de herramientas deseadas.")
    else:
        top = dd.head(25)
        fig2 = px.bar(top.sort_values("menciones"), x="menciones", y="termino",
                      orientation="h", text="menciones",
                      title="Términos más pedidos")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=max(400, 18 * len(top)), yaxis_title=None,
                           xaxis_title="Docentes que lo mencionan")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("Un término que aparece aquí y no corresponde a ninguna competencia "
                   "de la rúbrica es una señal de que el marco se quedó corto, no de "
                   "que el docente se equivocó.")

        dim2 = st.selectbox("Ver por segmento:",
                            [None, "nivel_educativo", "funcion", "contexto"],
                            format_func=lambda x: {None: "Sin desagregar",
                                                   "nivel_educativo": "Nivel educativo",
                                                   "funcion": "Función",
                                                   "contexto": "Contexto"}[x],
                            key="sel_dim_demanda")
        if dim2:
            dseg = demanda_declarada(dim2, solo_control_ok=solo_ok)
            if dseg.empty:
                st.info("Sin suficientes menciones para desagregar.")
            else:
                piv = dseg.pivot_table(index="termino", columns="segmento",
                                       values="menciones", fill_value=0, aggfunc="sum")
                piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index[:25]]
                st.dataframe(piv, use_container_width=True)


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

        # Sección 0: IDENTIFICACIÓN
        # La CURP es lo que permite que este autodiagnóstico deje de ser anónimo
        # y pueda unirse después con el registro de acreditación y con la
        # encuesta de satisfacción del curso que se le recomiende.
        st.subheader("🪪 Identificación")
        curp_input = st.text_input(
            "CURP",
            max_chars=18,
            placeholder="18 caracteres",
            help="Se usa para guardar tu historial y para relacionar tu diagnóstico "
                 "con los cursos que acredites. Puedes escribirla en minúsculas.",
            key="curp_input",
        )
        st.caption("Se valida el formato y el dígito verificador. No se almacena ningún otro dato personal.")

        # Estos tres campos son los que permiten agregar la demanda formativa por
        # segmento. Sin ellos el registro de brecha dice "faltan cursos" pero no
        # para quién, que es justo lo que el área necesita para planear.
        cseg1, cseg2, cseg3 = st.columns(3)
        with cseg1:
            nivel_educativo = st.selectbox("Nivel educativo", NIVELES_EDUCATIVOS,
                                           index=None, placeholder="Selecciona…",
                                           key="sel_nivel")
        with cseg2:
            funcion_docente = st.selectbox("Función", FUNCIONES_DOCENTES,
                                           index=None, placeholder="Selecciona…",
                                           key="sel_funcion")
        with cseg3:
            contexto_escuela = st.selectbox("Contexto de la escuela", CONTEXTOS_ESCUELA,
                                            index=None, placeholder="Selecciona…",
                                            key="sel_contexto")
        st.markdown("---")

        # Sección 1: Evaluación de competencias
        st.subheader("📊 Autodiagnóstico: retos de tu práctica docente")
        
        col1, col2 = st.columns(2)
        
        competencias_col1 = COMPETENCIAS_EVALUADAS[:5]
        competencias_col2 = COMPETENCIAS_EVALUADAS[5:]
        
        control_respuesta = None
        with col1:
            for pos, key in enumerate(competencias_col1, start=1):
                # El reactivo de control se intercala a media columna, con el
                # mismo formato que los demás bloques para que no se distinga.
                if pos == CONTROL_POSICION:
                    st.subheader(CONTROL_TITULO)
                    st.caption("🧭 Verificación de lectura del instrumento")
                    control_respuesta = st.radio(
                        "Selecciona tu nivel:",
                        options=CONTROL_OPCIONES,
                        key="radio_control",
                        label_visibility="collapsed"
                    )
                    st.divider()

                value = RUBRICA_DATA[key]
                st.subheader(value["titulo"])
                # Correspondencia con el marco que usa el DNF estatal.
                st.caption(f"🧭 Marco MEJOREDU — {etiqueta_funcion(key)}")
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
                # Correspondencia con el marco que usa el DNF estatal.
                st.caption(f"🧭 Marco MEJOREDU — {etiqueta_funcion(key)}")
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
        
        # Sección 3: DEMANDA FORMATIVA DECLARADA
        # Esta es la única pregunta del sistema que mira hacia adelante. Todo lo
        # demás (rúbrica, encuesta de satisfacción) pregunta por lo que ya pasó.
        # Es la materia prima para decidir qué cursos diseñar el próximo ciclo.
        st.markdown("---")
        st.subheader("🧰 ¿Qué herramientas quieres aprender?")
        st.write("*Para mejorar tu práctica educativa. Escríbelo con tus palabras: "
                 "de aquí salen las propuestas de cursos nuevos.*")
        herramientas_deseadas = st.text_area(
            label="Herramientas, métodos o temas que te gustaría dominar:",
            placeholder="""Ejemplo: Quiero aprender a usar inteligencia artificial para planear
mis clases, y herramientas para evaluar sin examen escrito. También me interesa
cómo trabajar con alumnos que tienen barreras de aprendizaje en grupos grandes.""",
            max_chars=600,
            height=110,
            help="No se limita al catálogo actual: justamente sirve para detectar lo que falta.",
            key="herramientas_deseadas",
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
                # La CURP se valida antes que nada: sin ella no hay dónde guardar.
                curp_ok, curp_msg = validar_curp(curp_input)
                if not curp_ok:
                    st.error(f"CURP: {curp_msg}")
                    st.stop()
                curp = curp_msg   # validar_curp devuelve la CURP ya normalizada

                if not nivel_educativo or not funcion_docente:
                    st.error("Selecciona tu nivel educativo y tu función: sin eso la "
                             "necesidad formativa no se puede agregar por segmento.")
                    st.stop()

                # Validar datos
                validar_datos_entrada(responses)

                with st.spinner("🧠 Analizando tu perfil y contexto con IA avanzada..."):
                    scores = []
                    for key in COMPETENCIAS_EVALUADAS:
                        selected = responses[key]
                        score = RUBRICA_DATA[key]["descriptores"].index(selected) + 1
                        scores.append(score)

                    # Reactivo de control. No condiciona el envío: se registra y
                    # se informa en los resultados para que quien lea el perfil
                    # sepa con qué nivel de atención fue contestado.
                    if control_respuesta is None:
                        control_num, control_ok = None, None
                    else:
                        control_num = CONTROL_OPCIONES.index(control_respuesta) + 1
                        control_ok = 1 if control_num == CONTROL_CORRECTA else 0

                    # El docente se identifica por CURP, no por un id fijo.
                    id_docente = obtener_o_crear_docente(
                        curp, contexto_personal,
                        nivel_educativo=nivel_educativo,
                        funcion=funcion_docente,
                        contexto_escuela=contexto_escuela)

                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()

                    # INSERT y no INSERT OR REPLACE: cada autodiagnóstico es un
                    # registro nuevo, para conservar la trayectoria del docente.
                    column_names = ', '.join(COMPETENCIAS_EVALUADAS)
                    placeholders = ', '.join(['?'] * len(COMPETENCIAS_EVALUADAS))
                    sql_insert = f"""
                        INSERT INTO Evaluaciones
                        (id_docente, {column_names}, contexto_personal, fuente,
                         herramientas_deseadas, control_respuesta, control_ok)
                        VALUES (?, {placeholders}, ?, ?, ?, ?, ?)
                    """
                    cursor.execute(sql_insert,
                                   [id_docente] + scores +
                                   [contexto_personal, 'autopercepcion',
                                    herramientas_deseadas, control_num, control_ok])

                    conn.commit()
                    conn.close()

                    # Generar recomendaciones CON CONTEXTO
                    st.session_state.recomendaciones = recomendar_cursos_diversificado(
                        id_docente, contexto_personal, 5,
                        herramientas_deseadas=herramientas_deseadas
                    )

                    # Registro de brecha de oferta: qué necesidades detectó el
                    # diagnóstico y cuáles de ellas el catálogo no pudo atender.
                    registrar_brecha(
                        id_docente,
                        st.session_state.get('_debiles', []),
                        st.session_state.get('_cubiertas', set()),
                        control_ok=control_ok,
                        id_evaluacion=id_evaluacion,
                    )

                    st.session_state.scores = scores
                    st.session_state.curp = curp
                    st.session_state.id_docente = id_docente
                    st.session_state.control_ok = control_ok
                    st.session_state.n_previas = max(0, len(historial_docente(id_docente)) - 1)
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

        st.markdown("---")
        if st.button("🧩 Brecha de oferta", use_container_width=True):
            st.session_state.page = 'brecha'
            st.rerun()
        if st.session_state.get('page') == 'brecha':
            if st.button("← Volver al diagnóstico", use_container_width=True):
                st.session_state.page = 'diagnostic'
                st.rerun()
        
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
    if st.session_state.page == 'brecha':
        show_brecha_page()
    elif st.session_state.page == 'diagnostic':
        show_diagnostic_page_con_contexto()  # Nueva función con contexto
    else:
        show_recommendations_page_diversificada()  # Función mejorada para mostrar resultados


if __name__ == "__main__":
    main()