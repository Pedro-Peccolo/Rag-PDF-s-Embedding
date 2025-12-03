# -*- coding: utf-8 -*-
# Workaround para problema de versao do SQLite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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

def buscar_informacao(db, pergunta, k=3):
    """Busca os trechos mais relevantes para a pergunta"""
    print(f"\nPergunta: {pergunta}")
    print("\nBuscando informacoes relevantes...\n")
    
    # Buscar documentos similares (buscar mais para garantir k únicos após remover duplicatas)
    docs = db.similarity_search(pergunta, k=k*2)  # Busca mais para ter opções
    
    # Remover duplicatas baseado no conteúdo do chunk
    docs_unicos = []
    conteudos_vistos = set()
    
    for doc in docs:
        conteudo_normalizado = doc.page_content.strip()
        # Criar uma chave única baseada no conteúdo + fonte + página
        chave = (conteudo_normalizado, 
                doc.metadata.get('source', ''), 
                doc.metadata.get('page', ''))
        
        if chave not in conteudos_vistos:
            conteudos_vistos.add(chave)
            docs_unicos.append(doc)
            
            # Parar quando tiver k chunks únicos
            if len(docs_unicos) >= k:
                break
    
    print("="*80)
    print("TRECHOS MAIS RELEVANTES ENCONTRADOS:")
    print("="*80)
    
    for i, doc in enumerate(docs_unicos, 1):
        fonte = doc.metadata.get('source', 'Desconhecido')
        pagina = doc.metadata.get('page', 'N/A')
        conteudo = doc.page_content.strip()
        
        print(f"\n[{i}] Fonte: {fonte} (Pagina {pagina})")
        print("-"*80)
        print(conteudo)
        print("-"*80)
    
    return docs_unicos

def main():
    """Funcao principal"""
    print("="*80)
    print("SISTEMA DE BUSCA EM DOCUMENTOS - VERSAO SIMPLES")
    print("="*80)
    print("\nEste sistema busca e retorna os trechos mais relevantes dos seus PDFs.")
    print("Nao usa LLM externa - funciona 100% local e offline!")
    print("="*80)
    
    # Carregar banco vetorial
    print("\nInicializando sistema...")
    db = carregar_banco_vetorial()
    print("\nSistema pronto!")
    
    # Loop de perguntas
    while True:
        print("\n" + "="*80)
        pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        
        if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
            print("\nEncerrando sistema. Ate logo!")
            break
        
        if not pergunta.strip():
            print("Por favor, digite uma pergunta valida.")
            continue
        
        try:
            buscar_informacao(db, pergunta, k=3)
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

