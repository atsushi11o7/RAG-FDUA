import os
import sys
from pathlib import Path
import json
import pandas as pd
import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# リポジトリディレクトリの設定
repo_dir = Path(__file__).resolve().parents[2]
if repo_dir.as_posix() not in sys.path:
    sys.path.append(repo_dir.as_posix())

# 環境変数からAPIキーを取得
load_dotenv(repo_dir.joinpath(".env.local"))
openai_api_key = os.environ.get("OPENAI_API_KEY", None)
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in .env.local")

def cosine_similarity(vector1, vector2):
    """
    コサイン類似度を計算する関数

    Args:
        vector1 (List[float]): ベクトル1
        vector2 (List[float]): ベクトル2

    Returns:
        float: コサイン類似度
    """
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm1 = sum(a * a for a in vector1) ** 0.5
    norm2 = sum(b * b for b in vector2) ** 0.5
    return dot_product / (norm1 * norm2)


def retrieve_vector_store(
    query: str,
    vector_store_dir: str,
    top_k: int = 1,
    calculate_score: bool = False,
    collection_name: str = None
) -> List[Dict]:
    """
    JSONベクトルストアからクエリに基づいて検索

    Args:
        query (str): 検索クエリ
        vector_store_dir (str): JSONベクトルストアの保存先ディレクトリ
        collection_name (str): JSONベクトルストアのコレクション名
        top_k (int): 上位何件の結果を返すか

    Returns:
        List[Dict]: 検索結果のリスト（コンテンツ、ファイル名、類似度スコアを含む）
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    if collection_name == None:
        vectorstore = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma(
            persist_directory=vector_store_dir,
            collection_name=collection_name,
            embedding_function=embeddings
        )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    results = retriever.get_relevant_documents(query)

    if calculate_score:
        # クエリの埋め込みを計算
        query_embedding = embeddings.embed_query(query)

    scored_results = []
    for result in results:
        document_embedding = embeddings.embed_query(result.page_content)
        similarity_score = "-"
        if calculate_score:
            similarity_score = str(cosine_similarity(query_embedding, document_embedding))
        scored_results.append({
            "content": result.page_content,
            "file_name": result.metadata.get('file_name', 'unknown'),
            "similarity": similarity_score
        })

    return scored_results