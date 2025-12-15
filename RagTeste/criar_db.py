# -*- coding: utf-8 -*-
# Workaround para problema de versao do SQLite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import requests
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Carregar variaveis de ambiente
load_dotenv()

# Configuração PROCEMPA
# Endpoint da API de embedding da PROCEMPA
PROCEMPA_EMBEDDING_URL = os.getenv("PROCEMPA_EMBEDDING_URL", "https://nv-embed1b.k8s-gpu.procempa.com.br/v1/embeddings")

# Usando documentos de teste e banco separado
# Obter o diretório do script atual para caminhos relativos funcionarem
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_BASE = os.path.join(SCRIPT_DIR, "documentos_teste")  # caminho para os documentos de teste
PASTA_DB = os.path.join(SCRIPT_DIR, "chroma_db_procempa")  # banco de dados local

def criar_db():
    #carregar documentos
    documentos = carregar_documentos()
    
    #dividir os documentos em chunks (blocos de texto)
    chunks = dividir_chunks(documentos)

    #vetorizar os chunks com o processo de embedding  (transformar o texto em vetores)
    vetorizar_chunks(chunks)
    

def carregar_documentos():
   
    # usar DirectoryLoader com TextLoader para arquivos .txt no diretorio documentos_teste
    carregador = DirectoryLoader(
        PASTA_BASE,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documentos = carregador.load()
    print(f"Carregados {len(documentos)} documentos de '{PASTA_BASE}'")
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, length_function=len)
    chunks = separador_documentos.split_documents(documentos)
    
    # filtrar chunks duplicados /Era para parar de retornar chunks duplicados mas era por conta do banco duplicado, posso remover isso se quiser
    conteudos_unicos = set()
    chunks_filtrados = []

    for chunk in chunks:
        conteudo = chunk.page_content.strip()
        if conteudo not in conteudos_unicos:
            conteudos_unicos.add(conteudo)
            chunks_filtrados.append(chunk)

    numero_de_chunks_original = len(chunks)
    numero_de_chunks_filtrados = len(chunks_filtrados)
    print(f"Numero de chunks: {numero_de_chunks_original} (original) -> {numero_de_chunks_filtrados} (após filtrar duplicatas)")
    return chunks_filtrados

class ProcempaEmbeddings(Embeddings):
    """Classe customizada para usar embeddings da API PROCEMPA"""
    
    def __init__(self, api_url: str, verbose: bool = True):
        self.api_url = api_url
        self.verbose = verbose
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para uma lista de documentos"""
        embeddings = []
        
        #lotes para evitar sobrecarga
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._embed_batch(batch, input_type="passage") #_embed_batch é a função que gera os embeddings
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        #vetoriza/embeddings para uma query=pergunta
        return self._embed_batch([text], input_type="query")[0]
    
    def _embed_batch(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:# input_type="passage" para documentos, "query" para queries
        """Gera embeddings para um lote de textos"""
        # Formato da API NVIDIA NIM
        # Modelos assimétricos precisam do parâmetro input_type: "passage" ou "query"
        embeddings = []
        
        for text in texts:
            payload = {
                "input": text,
                "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",  # Modelo correto da PROCEMPA
                "input_type": input_type  # "passage" para documentos, "query" para queries
            }
            
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=60
                )
                response.raise_for_status()
                
                data = response.json()
                # A API NVIDIA retorna no formato {"data": [{"embedding": [...]}]}
                if "data" in data and len(data["data"]) > 0:
                    embedding = data["data"][0]["embedding"]
                    embeddings.append(embedding)
                elif "embedding" in data:
                    embeddings.append(data["embedding"])
                else:
                    # Tentar extrair diretamente
                    embeddings.append(data if isinstance(data, list) else [])
            
            except requests.exceptions.RequestException as e:
                print(f"ERRO ao chamar API PROCEMPA para texto: {text[:50]}...")
                print(f"Erro: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        print(f"Resposta da API: {e.response.text}")
                    except:
                        pass
                raise
        
        return embeddings


def vetorizar_chunks(chunks):
    #vetoriza chunks usando o modelo de embedding
    
    # cria modelo de embedding
    embeddings = ProcempaEmbeddings(
        api_url=PROCEMPA_EMBEDDING_URL
    )
    
    print("Criando banco de dados vetorial com ChromaDB usando embeddings...")
    
    # criar banco vetorial com ChromaDB
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PASTA_DB  # salva localmente
    )
    
    print(f"Banco de dados criado com sucesso em '{PASTA_DB}'!")
    print(f"Total de {len(chunks)} chunks vetorizados!")
    
    return db

if __name__ == "__main__":
    criar_db()