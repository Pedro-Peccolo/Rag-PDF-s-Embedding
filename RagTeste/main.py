# -*- coding: utf-8 -*-
# Workaround para problema de versao do SQLite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import List
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from criar_db import ProcempaEmbeddings
from retriever import buscar_com_scores
from dotenv import load_dotenv

# Baixar punkt tokenizer se necessário
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Carregar variaveis de ambiente
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

# Configurações do banco PROCEMPA
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_DB_PROCEMPA = os.path.join(SCRIPT_DIR, "chroma_db_procempa")
PROCEMPA_EMBEDDING_URL = os.getenv("PROCEMPA_EMBEDDING_URL", "https://nv-embed1b.k8s-gpu.procempa.com.br/v1/embeddings")
PROCEMPA_API_KEY = os.getenv("PROCEMPA_API_KEY", "")

def criar_embeddings_procempa(verbose=True):
    """Cria o modelo de embeddings PROCEMPA usado na criação do banco"""
    embeddings = ProcempaEmbeddings(
        api_url=PROCEMPA_EMBEDDING_URL,
        api_key=PROCEMPA_API_KEY,
        verbose=verbose
    )
    return embeddings


def carregar_banco_vetorial_procempa():
    """Carrega o banco vetorial ChromaDB PROCEMPA já criado"""
    embeddings = criar_embeddings_procempa(verbose=False)

    print("Carregando banco de dados vetorial...")

    # Verificar se o banco existe e tem documentos
    if not os.path.exists(PASTA_DB_PROCEMPA):
        print(f"❌ ERRO: Banco de dados não encontrado em '{PASTA_DB_PROCEMPA}'")
        print("Execute 'python3 criar_db.py' primeiro para criar o banco de dados.")
        return None

    db = Chroma(
        persist_directory=PASTA_DB_PROCEMPA,
        embedding_function=embeddings
    )

    # Verificar se há documentos no banco
    try:
        docs = db.get()
        num_docs = len(docs['documents']) if 'documents' in docs else 0

        if num_docs == 0:
            print(f"❌ ERRO: Banco de dados existe mas está vazio.")
            print("Execute 'python3 criar_db.py' primeiro para criar o banco de dados.")
            return None

        print(f"Banco carregado com sucesso! ({num_docs} documentos)")
    except Exception as e:
        print(f"❌ ERRO ao verificar banco de dados: {e}")
        return None

    return db

def criar_llm_nvidia():
    """Cria a LLM usando NVIDIA NIM (Llama)"""
    # Verificar se a API key da NVIDIA está configurada
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        print("❌ ERRO: NVIDIA_API_KEY não encontrada!")
        print("Configure a variável de ambiente NVIDIA_API_KEY no arquivo .env")
        print("Para obter uma chave:")
        print("1. Criar conta em: https://build.nvidia.com/")
        print("2. Gerar API key em: https://build.nvidia.com/explore/reasoning")
        return None

    try:
        # Configurar a API key da NVIDIA
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key

        # Usar modelo Llama disponível no NVIDIA NIM
        # Opções: "meta/llama-3.1-8b-instruct", "meta/llama-3.2-3b-instruct", etc.
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            temperature=0.7,  # Aumentado para ser menos conservador
            max_completion_tokens=1024,
            top_p=0.9  # Aumentado para mais diversidade
        )
        print("✅ LLM NVIDIA Llama inicializada com sucesso!")
        return llm
    except Exception as e:
        print(f"❌ ERRO ao criar LLM NVIDIA: {e}")
        print("Verifique se as credenciais da NVIDIA estão configuradas corretamente.")
        return None

class HybridRetriever(BaseRetriever):
    """Retriever híbrido que usa a lógica do retriever.py"""

    def __init__(self, db, alpha=0.7, k=3):
        super().__init__()
        object.__setattr__(self, 'db', db)
        object.__setattr__(self, 'alpha', alpha)
        object.__setattr__(self, 'k', k)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Busca híbrida usando a função do retriever.py"""
        # Usar a função buscar_com_scores do retriever.py
        docs_unicos, _ = buscar_com_scores(self.db, query, k=self.k, a=self.alpha)
        # Retornar apenas os documentos (não os scores)
        return [item['documento'] for item in docs_unicos]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Versão assíncrona"""
        return self._get_relevant_documents(query)

def criar_prompt_template():
    """Cria o template de prompt em português para o RAG - amigável e direto"""
    template = """Você é um sistema RAG (Retrieval-Augmented Generation) que responde perguntas baseado em documentos fornecidos.

O contexto abaixo contém informações extraídas de documentos. Sua função é apresentar essas informações de forma clara e amigável, SEM questionar, SEM corrigir, SEM adicionar conhecimento externo.

Contexto dos documentos:
{context}

Pergunta do usuário: {question}

Sua tarefa:
- Responda usando APENAS as informações do contexto acima
- Se o contexto contém a resposta, apresente-a de forma clara, direta e amigável
- Trate todas as informações do contexto como verdadeiras e confiáveis
- NÃO diga "desculpe", "não posso", "não há informações confiáveis" ou similar
- NÃO adicione conhecimento histórico, científico ou factual externo
- Se o contexto diz "X descobriu Y em Z", responda "X descobriu Y em Z" de forma natural
- Se não houver informação no contexto, diga apenas "Não há informação sobre isso nos documentos disponíveis"
- Seja prestativo, amigável e confiante nas informações do contexto

Responda em português brasileiro de forma natural:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return prompt

def criar_sistema_rag():
    """Cria o sistema RAG completo com busca híbrida e LLM NVIDIA"""
    # Carregar banco vetorial PROCEMPA
    db = carregar_banco_vetorial_procempa()
    if db is None:
        return None

    # Criar modelo de embeddings PROCEMPA
    embeddings = criar_embeddings_procempa(verbose=False)

    # Criar LLM NVIDIA
    llm = criar_llm_nvidia()
    if llm is None:
        return None

    # Criar retriever híbrido com alpha ótimo (0.7)
    hybrid_retriever = HybridRetriever(db, alpha=0.7, k=3)

    # Criar prompt template
    prompt = criar_prompt_template()

    # Criar cadeia de consulta RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

def fazer_pergunta_rag(qa_chain, pergunta):
    """Faz uma pergunta ao sistema RAG completo com lógica de score mínimo"""
    print(f"\n" + "="*80)
    print(f"Pergunta: {pergunta}")
    print("="*80)
    print("Processando com busca híbrida (70% embedding + 30% BM25) + LLM Llama...")
    print()

    try:
        resultado = qa_chain.invoke({"query": pergunta})

        print("RESPOSTA DO SISTEMA RAG:")
        print("-" * 50)
        print(resultado['result'])
        print()

        # Mostrar fontes utilizadas
        if resultado.get('source_documents'):
            print("FONTES UTILIZADAS:")
            print("-" * 50)
            fontes_vistas = set()
            contador = 1
            for doc in resultado['source_documents']:
                fonte = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
                pagina = doc.metadata.get('page', 'N/A')
                # Criar chave única para evitar duplicatas
                chave = (fonte, pagina)
                if chave not in fontes_vistas:
                    fontes_vistas.add(chave)
                    print(f"{contador}. {fonte} (página {pagina})")
                    contador += 1
        print()

    except Exception as e:
        print(f"❌ ERRO ao processar pergunta: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Função principal do sistema RAG completo"""
    print("="*80)
    print("🤖 SISTEMA RAG COMPLETO - CONSULTA DE DOCUMENTOS")
    print("="*80)
    print("Pipeline: Query → Retriever Híbrido (70% Embeddings + 30% BM25) → Contexto → LLM Llama")
    print("="*80)

    # Verificar configurações necessárias
    print("\nVerificando configurações...")

    # Verificar URL PROCEMPA (obrigatória)
    if not PROCEMPA_EMBEDDING_URL:
        print("❌ ERRO: PROCEMPA_EMBEDDING_URL não encontrada!")
        print("Configure a variável de ambiente PROCEMPA_EMBEDDING_URL no arquivo .env")
        return


    # Verificar banco de dados
    if not os.path.exists(PASTA_DB_PROCEMPA):
        print(f"❌ ERRO: Banco de dados não encontrado em '{PASTA_DB_PROCEMPA}'")
        print("Execute 'python3 criar_db.py' primeiro para criar o banco de dados.")
        return

    print("✅ Configurações validadas!")

    # Criar sistema RAG
    print("\nInicializando sistema RAG...")
    print("- Carregando banco vetorial...")
    print("- Inicializando retriever híbrido (alpha=0.7)...")
    print("- Conectando com LLM NVIDIA Llama...")

    qa_chain = criar_sistema_rag()

    if qa_chain is None:
        print("\n❌ Falha na inicialização do sistema RAG.")
        return

    print("✅ Sistema RAG pronto para uso!\n")

    # Loop de perguntas
    while True:
        print("-" * 80)
        pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")

        if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
            print("\n" + "="*80)
            print("👋 Encerrando sistema RAG. Até logo!")
            print("="*80)
            break

        if not pergunta.strip():
            print("❌ Por favor, digite uma pergunta válida.")
            continue

        # Processar pergunta com RAG completo
        fazer_pergunta_rag(qa_chain, pergunta)

if __name__ == "__main__":
    main()
