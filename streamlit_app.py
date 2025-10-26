import streamlit as st
import os
import time
import requests
import tempfile
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. Configuración de la Página y Variables ---

# Configuración de la página de Streamlit (debe ser el primer comando de st)
st.set_page_config(
    page_title="Asistente de Manual Hyundai",
    page_icon="🚗",
    layout="centered"
)

# Cargar secretos de forma segura
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except KeyError:
    st.error("ERROR: Por favor, configura tus claves PINECONE_API_KEY y HUGGINGFACEHUB_API_TOKEN en el archivo .streamlit/secrets.toml")
    st.stop()

# --- Variables Globales de Configuración ---
PDF_URL = "https://kerner.hyundai.com.ec/documentos/manuales/manual-propietario-accent22.pdf"
INDEX_NAME = "hiunday-despliegue" # El índice que ya creamos
NAMESPACE = "manual_hiunday" # El nuevo namespace solicitado
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
MODEL_DIMENSION = 384 # Dimensión de all-MiniLM-L6-v2

# --- 2. Funciones Cacheadas (Optimización) ---
# Usamos @st.cache_resource para cargar modelos y conexiones una sola vez.

@st.cache_resource
def get_embeddings_model():
    """Carga el modelo de embeddings de Hugging Face."""
    st.info("Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'} # Usar CPU
    )

@st.cache_resource
def get_pinecone_index():
    """Conecta a Pinecone y obtiene el objeto Index."""
    st.info(f"Conectando al índice de Pinecone: '{INDEX_NAME}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Validar que el índice existe
    if INDEX_NAME not in pc.list_indexes().names():
        st.error(f"El índice '{INDEX_NAME}' no existe en Pinecone.")
        st.info("Creando índice... (esto puede tardar un momento)")
        # Si no existe, lo creamos (basado en el script original)
        from pinecone import ServerlessSpec # Importar aquí para que no falle si no se usa
        pc.create_index(
            name=INDEX_NAME,
            dimension=MODEL_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        st.success(f"Índice '{INDEX_NAME}' creado exitosamente.")

    return pc.Index(INDEX_NAME)

@st.cache_resource
def get_llm(_hf_token):
    """Carga el LLM de Hugging Face usando ChatHuggingFace."""
    st.info(f"Cargando LLM ({LLM_REPO_ID})...")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=_hf_token,
        task="conversational", # Importante: la tarea que sí funcionó
        model_kwargs={"stop": ["\nUser:", "\nSystem:", "</s>"]}
    )
    # Envolverlo en ChatHugginFace para compatibilidad
    return ChatHuggingFace(llm=llm_endpoint)

# --- 3. Funciones de Ingesta (Fase 1) ---

def ingest_pdf_from_url(url, index_obj, embeddings_model, namespace):
    """Descarga, procesa y carga un PDF desde una URL a Pinecone."""
    try:
        # 1. Descargar el PDF
        st.info(f"Descargando PDF desde {url}...")
        response = requests.get(url)
        response.raise_for_status() # Lanza error si la descarga falla

        # 2. Guardar en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        st.info(f"PDF guardado temporalmente en {temp_file_path}")

        # 3. Cargar el PDF temporal
        loader = PyPDFLoader(temp_file_path)
        documentos = loader.load()

        # 4. Dividir en Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documentos)
        st.info(f"PDF dividido en {len(chunks)} chunks.")

        if not chunks:
            st.error("No se pudieron extraer chunks del PDF.")
            return

        # 5. Borrar namespace anterior (para evitar duplicados)
        try:
            index_obj.delete(delete_all=True, namespace=namespace)
            st.info(f"Namespace '{namespace}' anterior borrado.")
        except Exception as e:
            st.warning(f"No se pudo borrar el namespace (puede que estuviera vacío): {e}")

        # 6. Realizar Upsert a Pinecone
        st.info(f"Iniciando upsert de {len(chunks)} chunks a Pinecone (namespace='{namespace}')...")
        PineconeVectorStore.from_documents(
            chunks,
            embeddings_model,
            index_name=index_obj.name,
            namespace=namespace
        )
        
        # 7. Sondeo (crucial)
        target_vector_count = len(chunks)
        current_vector_count = 0
        timeout = 120 # Esperar max 2 minutos
        start_time = time.time()
        
        with st.spinner(f"Esperando que Pinecone indexe {target_vector_count} vectores..."):
            while current_vector_count < target_vector_count and (time.time() - start_time) < timeout:
                time.sleep(5)
                try:
                    stats = index_obj.describe_index_stats()
                    current_vector_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
                    st.info(f"Progreso de indexación: {current_vector_count} / {target_vector_count}")
                except Exception as e:
                    st.info(f"Consultando stats... ({e})")

        if current_vector_count >= target_vector_count:
            st.success(f"¡Ingesta completada! {current_vector_count} vectores en namespace '{namespace}'.")
        else:
            st.warning(f"Tiempo de espera agotado. Pinecone solo indexó {current_vector_count} de {target_vector_count} vectores.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el PDF: {e}")
    except Exception as e:
        st.error(f"Error durante la ingesta: {e}")
    finally:
        # 8. Limpiar el archivo temporal
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            st.info("Archivo temporal eliminado.")

# --- 4. Funciones de RAG (Fase 2) ---

def get_rag_chain(embeddings, index_obj, llm):
    """Crea y devuelve la cadena RAG."""
    
    # 1. Crear el VectorStore (solo para apuntar)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE # ¡Apuntar al namespace correcto!
    )

    # 2. Crear el Retriever
    # Usamos k=3 para obtener más contexto del manual
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3, 
            "namespace": NAMESPACE # Forzar el namespace
        }
    )

    # 3. Definir el Prompt
    template = """
    Eres un asistente experto en el manual de propietario del Hyundai Accent.
    Responde la pregunta basándote *única y exclusivamente* en el siguiente contexto.
    Sé técnico, breve y directo.
    Si la respuesta no está en el contexto, di "No tengo información sobre eso en el manual."

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta útil:
    """
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    # 4. Crear la Cadena RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True # Para mostrar las fuentes
    )
    return qa_chain

# --- 5. Interfaz de Streamlit (Fase 2 y 3) ---

st.title("Asistente de Manual Hyundai Accent 🚗")
st.caption(f"Consultando el índice '{INDEX_NAME}' (namespace: '{NAMESPACE}')")

# --- Cargar Recursos Cacheados ---
try:
    embeddings_model = get_embeddings_model()
    index = get_pinecone_index()
    llm = get_llm(HF_TOKEN)
    rag_chain = get_rag_chain(embeddings_model, index, llm)
except Exception as e:
    st.error(f"Error al cargar los recursos de IA: {e}")
    st.stop()


# --- Botón de Ingesta en el Sidebar ---
with st.sidebar:
    st.subheader("Gestión de Datos")
    st.write(f"**PDF:** `{PDF_URL.split('/')[-1]}`")
    st.write(f"**Namespace:** `{NAMESPACE}`")
    if st.button("Re-Ingestar PDF"):
        with st.spinner("Realizando ingesta completa... Esto puede tardar varios minutos."):
            ingest_pdf_from_url(PDF_URL, index, embeddings_model, NAMESPACE)
        st.success("Ingesta finalizada. ¡Listo para chatear!")
        # Limpiamos caché de la cadena RAG si es necesario, aunque al estar
        # basada en los recursos cacheados, debería actualizarse sola.

# --- Lógica del Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿Cómo puedo ayudarte con el manual del Hyundai Accent?"}]

# Mostrar mensajes existentes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Ej: ¿Cómo reviso el aceite del motor?"):
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el manual..."):
            try:
                # 7. Ejecutar la cadena
                response = rag_chain.invoke({"query": prompt})
                
                full_response = response["result"]
                
                # Añadir las fuentes (Fase 3: Validación)
                sources = response.get("source_documents")
                if sources:
                    full_response += "\n\n---"
                    full_response += "\n\n**Fuentes Consultadas (fragmentos del manual):**\n"
                    unique_sources = set()
                    for doc in sources:
                        page = doc.metadata.get('page', 'N/A')
                        # Limpiar el texto para que se vea bien en markdown
                        source_text = doc.page_content.strip().replace("\n", " ")[:120] + "..."
                        unique_sources.add(f"- Pág. {page} (Fragmento: *\"{source_text}\"*)")
                    
                    full_response += "\n".join(list(unique_sources))

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Ocurrió un error al procesar tu solicitud: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

