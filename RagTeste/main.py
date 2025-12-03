# -*- coding: utf-8 -*-
# Workaround para problema de versao do SQLite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv

# Carregar variaveis de ambiente do arquivo .env
load_dotenv()

PASTA_DB = "chroma_db"

def criar_embeddings():
    """Cria o mesmo modelo de embeddings usado na criacao do banco"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    return embeddings

def carregar_banco_vetorial():
    """Carrega o banco vetorial ChromaDB ja criado"""
    embeddings = criar_embeddings()
    
    print("Carregando banco de dados vetorial...")
    db = Chroma(
        persist_directory=PASTA_DB,
        embedding_function=embeddings
    )
    print(f"Banco carregado com sucesso!")
    return db

def criar_llm():
    """Cria a LLM usando HuggingFace Endpoint (GRATUITO)"""
    # Usando modelo disponivel na API gratuita do HuggingFace
    # mistralai/Mistral-7B-Instruct-v0.2 funciona bem com perguntas e respostas
    
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.3,
        max_new_tokens=512,
        timeout=120
    )
    return llm

class RetrieverSemDuplicatas(BaseRetriever):
    """Retriever customizado que remove duplicatas antes de retornar documentos"""
    
    def __init__(self, base_retriever: BaseRetriever, k: int = 3):
        super().__init__()
        self.base_retriever = base_retriever
        self.k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Busca documentos e remove duplicatas"""
        # Buscar mais documentos para garantir k únicos após remover duplicatas
        docs = self.base_retriever.get_relevant_documents(query)
        
        # Remover duplicatas baseado no conteúdo
        docs_unicos = []
        conteudos_vistos = set()
        
        for doc in docs:
            conteudo_normalizado = doc.page_content.strip()
            # Criar chave única: conteúdo + fonte + página
            chave = (
                conteudo_normalizado,
                doc.metadata.get('source', ''),
                doc.metadata.get('page', '')
            )
            
            if chave not in conteudos_vistos:
                conteudos_vistos.add(chave)
                docs_unicos.append(doc)
                
                # Parar quando tiver k chunks únicos
                if len(docs_unicos) >= self.k:
                    break
        
        return docs_unicos
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Versão assíncrona"""
        docs = await self.base_retriever.aget_relevant_documents(query)
        
        docs_unicos = []
        conteudos_vistos = set()
        
        for doc in docs:
            conteudo_normalizado = doc.page_content.strip()
            chave = (
                conteudo_normalizado,
                doc.metadata.get('source', ''),
                doc.metadata.get('page', '')
            )
            
            if chave not in conteudos_vistos:
                conteudos_vistos.add(chave)
                docs_unicos.append(doc)
                
                if len(docs_unicos) >= self.k:
                    break
        
        return docs_unicos

def criar_prompt_template():
    """Cria o template de prompt em portugues"""
    template = """Use as seguintes informacoes para responder a pergunta do usuario.
Se voce nao souber a resposta, apenas diga que nao sabe, nao tente inventar uma resposta.

Contexto: {context}

Pergunta: {question}

Resposta detalhada em portugues:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return prompt

def criar_sistema_rag():
    """Cria o sistema RAG completo com retriever que remove duplicatas"""
    # Carregar banco vetorial
    db = carregar_banco_vetorial()
    
    # Criar LLM
    llm = criar_llm()
    
    # Criar prompt
    prompt = criar_prompt_template()
    
    # Criar retriever base (busca mais para garantir k únicos após remover duplicatas)
    base_retriever = db.as_retriever(search_kwargs={"k": 6})  # Busca 6, retorna 3 únicos
    
    # Envolver com retriever customizado que remove duplicatas
    retriever_sem_duplicatas = RetrieverSemDuplicatas(base_retriever, k=3)
    
    # Criar cadeia de consulta RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_sem_duplicatas,  # Usa retriever sem duplicatas
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def fazer_pergunta(qa_chain, pergunta):
    """Faz uma pergunta ao sistema RAG"""
    print(f"\nPergunta: {pergunta}")
    print("Processando...\n")
    
    resultado = qa_chain.invoke({"query": pergunta})
    
    print(f"Resposta: {resultado['result']}\n")
    
    # Mostrar fontes (opcional) - removendo duplicatas
    if resultado.get('source_documents'):
        print("--- Fontes utilizadas ---")
        fontes_vistas = set()
        contador = 1
        for doc in resultado['source_documents']:
            fonte = doc.metadata.get('source', 'Desconhecido')
            pagina = doc.metadata.get('page', 'N/A')
            # Criar chave única para evitar duplicatas
            chave = (fonte, pagina)
            if chave not in fontes_vistas:
                fontes_vistas.add(chave)
                print(f"{contador}. {fonte} (página {pagina})")
                contador += 1
    
    return resultado

def main():
    """Funcao principal"""
    print("="*60)
    print("Sistema RAG - Consulta de Documentos")
    print("="*60)
    
    # Verificar se a API key do HuggingFace existe
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("\nERRO: Nao encontrei a API key do HuggingFace!")
        print("\nPara usar este sistema, voce precisa:")
        print("1. Criar conta gratuita em: https://huggingface.co")
        print("2. Gerar API token em: https://huggingface.co/settings/tokens")
        print("3. Criar arquivo .env na pasta do projeto com:")
        print("   HUGGINGFACEHUB_API_TOKEN=seu_token_aqui")
        return
    
    # Criar sistema RAG
    print("\nInicializando sistema RAG...")
    qa_chain = criar_sistema_rag()
    print("Sistema pronto!\n")
    
    # Loop de perguntas
    while True:
        print("-"*60)
        pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        
        if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
            print("\nEncerrando sistema. Ate logo!")
            break
        
        if not pergunta.strip():
            print("Por favor, digite uma pergunta valida.")
            continue
        
        try:
            fazer_pergunta(qa_chain, pergunta)
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
