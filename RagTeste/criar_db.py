# -*- coding: utf-8 -*-
# Workaround para problema de versao do SQLite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
#Usando HuggingFace GRATUITO - modelo multilingue para portugues
#O MODELO DE EMBEDING tem que ser igual ao modelo de IA que sera usado para a consulta

PASTA_BASE = "base"
PASTA_DB = "chroma_db"  # Pasta onde o banco vetorial sera salvo

def criar_db():
    #carregar documentos
    documentos = carregar_documentos()
    
    #dividir os documentos em chunks (blocos de texto)
    chunks = dividir_chunks(documentos)

    #vetorizar os chunks com o processo de embedding  (transformar o texto em vetores)
    vetorizar_chunks(chunks)
    

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE)
    documentos = carregador.load()
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, length_function=len)
    chunks = separador_documentos.split_documents(documentos)
    
    
    numero_de_chunks = len(chunks)
    print(f"Numero de chunks: {numero_de_chunks}")
    return chunks

def vetorizar_chunks(chunks):
    # Criar modelo de embeddings GRATUITO com HuggingFace
    # Modelo multilingue otimizado para portugues
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}  # Usa CPU (mude para 'cuda' se tiver GPU)
    )
    
    print("Criando banco de dados vetorial com ChromaDB...")
    
    # Criar banco vetorial com ChromaDB
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PASTA_DB  # Salva localmente
    )
    
    print(f"Banco de dados criado com sucesso em '{PASTA_DB}'!")
    print(f"Total de {len(chunks)} chunks vetorizados!")
    
    return db
 
criar_db()