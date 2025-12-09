# -*- coding: utf-8 -*-
# Workaround para problema de versao do SQLite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from langchain_chroma import Chroma
from criar_db import ProcempaEmbeddings
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Baixar punkt tokenizer se necessário
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Carregar variaveis de ambiente
load_dotenv()

# Usar banco PROCEMPA
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_DB = os.path.join(SCRIPT_DIR, "chroma_db_procempa")
PROCEMPA_EMBEDDING_URL = os.getenv("PROCEMPA_EMBEDDING_URL", "https://nv-embed1b.k8s-gpu.procempa.com.br/v1/embeddings")
PROCEMPA_API_KEY = os.getenv("PROCEMPA_API_KEY", "")

def criar_embeddings(verbose=True):
    """Cria o modelo de embeddings PROCEMPA usado na criacao do banco"""
    embeddings = ProcempaEmbeddings(
        api_url=PROCEMPA_EMBEDDING_URL,
        api_key=PROCEMPA_API_KEY,
        verbose=verbose
    )
    return embeddings

def carregar_banco_vetorial():
    """Carrega o banco vetorial ChromaDB ja criado"""
    embeddings = criar_embeddings()

    print("Carregando banco de dados vetorial...")

    # Verificar se o banco existe e tem documentos
    if not os.path.exists(PASTA_DB):
        print(f"❌ ERRO: Banco de dados não encontrado em '{PASTA_DB}'")
        print("Execute 'python3 criar_db.py' primeiro para criar o banco de dados.")
        return None

    db = Chroma(
        persist_directory=PASTA_DB,
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

def calcular_similaridade(query_embedding, doc_embeddings, top_k=3):
    """Calcula scores de similaridade usando cosine similarity"""
    query_vec = np.array(query_embedding).reshape(1, -1)
    doc_matrix = np.array(doc_embeddings)
    scores = cosine_similarity(query_vec, doc_matrix)[0]

    resultados = []
    for i, score in enumerate(scores):
        resultados.append({
            'indice': i,
            'score': float(score)
        })

    resultados.sort(key=lambda x: x['score'], reverse=True)
    top_resultados = resultados[:top_k]

    df_scores = pd.DataFrame({
        'ranking': range(1, len(top_resultados) + 1),
        'indice_documento': [r['indice'] for r in top_resultados],
        'score_similaridade': [r['score'] for r in top_resultados]
    })

    return top_resultados, df_scores

def buscar_com_scores(db, pergunta, k=3, a=1.0):
    """Busca documentos e calcula scores híbridos (embedding + BM25)"""
    print(f"\nPergunta: {pergunta}")
    print(f"Calculando scores híbridos (a={a:.1f}: {a*100:.0f}% embedding, {(1-a)*100:.0f}% BM25)...\n")

    # Criar modelo de embeddings
    embeddings_model = criar_embeddings(verbose=False)

    # Obter embedding da pergunta
    query_emb = embeddings_model.embed_query(pergunta)
    print("Embedding da query gerado")

    # Buscar mais documentos para ter diversidade de chunks
    search_k = 50  # Buscar 50 chunks (permite múltiplos chunks por documento)
    docs = db.similarity_search(pergunta, k=search_k)

    # Gerar embeddings dos documentos (debug + re-ranking)
    print("Gerando embeddings...")
    doc_embeddings = []
    docs_validos = []

    for idx, doc in enumerate(docs):
        try:
            emb = embeddings_model.embed_documents([doc.page_content])[0]
            doc_embeddings.append(emb)
            docs_validos.append(doc)
        except Exception as e:
            continue

    print(f"{len(doc_embeddings)} embeddings gerados")
    if len(doc_embeddings) == 0:
        print("❌ Nenhum embedding válido gerado.")
        return [], pd.DataFrame()

    # Converter em matriz numpy
    doc_matrix = np.array(doc_embeddings)
    query_vec = np.array(query_emb).reshape(1, -1)

    # Calcular scores de embedding (cosine similarity)
    embedding_scores = cosine_similarity(query_vec, doc_matrix)[0]

    # Preparar dados para BM25
    corpus = [doc.page_content.strip() for doc in docs_validos]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

    # Criar modelo BM25
    bm25 = BM25Okapi(tokenized_corpus)

    # Tokenizar query para BM25
    tokenized_query = word_tokenize(pergunta.lower())

    # Calcular scores BM25
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalizar scores BM25 para o range [0, 1] (dividindo pelo máximo)
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
    bm25_scores_normalized = bm25_scores / max_bm25

    # Calcular scores híbridos: a * embedding + (1-a) * bm25
    hybrid_scores = a * embedding_scores + (1 - a) * bm25_scores_normalized

    # Criar lista com todos os resultados e scores híbridos
    todos_resultados = []
    for i, (hybrid_score, embedding_score, bm25_score, doc) in enumerate(zip(hybrid_scores, embedding_scores, bm25_scores_normalized, docs_validos)):
        conteudo = doc.page_content.strip()
        todos_resultados.append({
            'indice': i,
            'score': float(hybrid_score),
            'score_embedding': float(embedding_score),
            'score_bm25': float(bm25_score),
            'documento': doc,
            'conteudo': conteudo,
            'fonte': doc.metadata.get('source', ''),
            'pagina': doc.metadata.get('page', '')
        })

    # Ordenar pelo score híbrido (decrescente)
    todos_resultados.sort(key=lambda x: x['score'], reverse=True)

    # Estratégia simples: selecionar k trechos únicos pulando duplicados por conteúdo
    # Pula duplicados e continua na lista até encontrar k resultados únicos
    docs_unicos = []
    conteudos_vistos = set()

    K = k

    def normalize_text(s):
        """Normaliza texto para comparação: remove espaços extras e converte para minúsculas"""
        return ' '.join(s.lower().split())

    # Percorrer resultados ordenados por score (maior para menor)
    for idx, resultado in enumerate(todos_resultados):
        # Normalizar conteúdo para comparação
        conteudo_normalizado = normalize_text(resultado['conteudo'])

        # Se já vimos esse conteúdo, pula e continua para o próximo
        if conteudo_normalizado in conteudos_vistos:
            continue

        # Conteúdo único: adiciona
        conteudos_vistos.add(conteudo_normalizado)
        docs_unicos.append(resultado)

        # Para quando tiver exatamente K resultados únicos
        if len(docs_unicos) >= K:
            break

    # Se não encontrou K resultados únicos, informar
    if len(docs_unicos) < K:
        print(f"Apenas {len(docs_unicos)} resultados únicos encontrados")
    else:
        print(f"Encontrados {len(docs_unicos)} resultados únicos")

    # Mostrar resultados
    print(f"\nRESULTADOS HÍBRIDOS (a={a:.1f}):")
    print("="*50)

    for i, item in enumerate(docs_unicos, 1):
        score = item['score']
        score_emb = item['score_embedding']
        score_bm25 = item['score_bm25']
        fonte = os.path.basename(item['fonte']) if item['fonte'] else 'Desconhecido'
        pagina = item['pagina'] or 'N/A'
        conteudo = item['conteudo']

        print(f"\n[{i}] Score: {score:.4f} (Emb: {score_emb:.4f}, BM25: {score_bm25:.4f}) | Fonte: {fonte} (Página {pagina})")
        print("-"*80)
        print(conteudo)
        print("-"*80)

    # Criar df de scores apenas com os únicos
    df_scores = pd.DataFrame({
        'ranking': range(1, len(docs_unicos) + 1),
        'score_hibrido': [item['score'] for item in docs_unicos],
        'score_embedding': [item['score_embedding'] for item in docs_unicos],
        'score_bm25': [item['score_bm25'] for item in docs_unicos],
        'fonte': [os.path.basename(item['fonte']) if item['fonte'] else 'Desconhecido' for item in docs_unicos],
        'pagina': [item['pagina'] or 'N/A' for item in docs_unicos]
    })

    # Mostrar tabela final
    print(f"\nTABELA DE SCORES HÍBRIDOS (a={a:.1f}):")
    print("="*50)
    print(df_scores.to_string(index=False, float_format='%.4f'))

    return docs_unicos, df_scores

def main():
    """Funcao principal"""
    print("="*80)
    print("SISTEMA DE BUSCA HÍBRIDA EM DOCUMENTOS")
    print("="*80)
    print("\nEste sistema busca e retorna os trechos mais relevantes dos documentos.")
    print("Usa Hybrid Search: combinação de embeddings PROCEMPA e BM25.")
    print("Você pode ajustar o parâmetro 'a' (0.0=100% BM25, 1.0=100% embedding).")
    print("="*80)
    
    # Carregar banco vetorial
    print("\nInicializando sistema...")
    db = carregar_banco_vetorial()

    if db is None:
        print("\n❌ Sistema não pôde ser inicializado devido a problemas no banco de dados.")
        return

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

        # Permitir ao usuário escolher o parâmetro a
        try:
            a_input = input("Digite o parâmetro 'a' (0.0=100% BM25, 1.0=100% embedding) [padrão: 0.7]: ").strip()
            if a_input == "":
                a = 0.7
            else:
                a = float(a_input)
                if not (0.0 <= a <= 1.0):
                    print("Valor deve estar entre 0.0 e 1.0. Usando padrão 0.7.")
                    a = 0.7
        except ValueError:
            print("Valor inválido. Usando padrão 0.7.")
            a = 0.7

        try:
            buscar_com_scores(db, pergunta, k=3, a=a)
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

