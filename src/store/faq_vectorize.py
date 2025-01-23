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


def chunk_json_content(
    json_data: Dict[str, List[str]],
    max_tokens: int = 1024,
    overlap: int = 100
) -> List[Document]:
    """
    JSONファイルの内容をチャンク化してドキュメント化。

    Args:
        json_data (Dict[str, List[str]]): JSONデータ。
        max_tokens (int): チャンクの最大トークン数。
        overlap (int): チャンク間の重複トークン数。

    Returns:
        List[Document]: チャンク化されたドキュメントリスト。
    """
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for file_name, content_list in json_data.items():
        # コンテンツを1つのテキストとして結合
        combined_text = "\n".join(content_list)
        text_chunks = splitter.split_text(combined_text)

        for chunk in text_chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={'file_name': file_name}
            ))

    return documents


def process_and_build_json_vector_store(
    json_file: str,
    vector_store_dir: str,
    collection_name: str,
    max_tokens_per_chunk: int = 1024,
    overlap_tokens: int = 100
):
    """
    JSONファイルをチャンク化してベクトルストアを構築。

    Args:
        json_file (str): JSONファイルのパス。
        embeddings (OpenAIEmbeddings): OpenAIのエンベディングインスタンス。
        vector_store_dir (str): ベクトルストアの保存先ディレクトリ。
        collection_name (str): コレクション名。
        max_tokens_per_chunk (int): 各チャンクの最大トークン数。
        overlap_tokens (int): チャンク間の重複トークン数。
    """
    
    # JSONファイルを読み込み
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # エンベディングの初期化
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    # チャンク化
    documents = chunk_json_content(json_data, max_tokens=max_tokens_per_chunk, overlap=overlap_tokens)

    # ベクトルストアを構築
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=vector_store_dir,
        collection_name=collection_name
    )
    vectorstore.persist()
    logging.info("[INFO] JSONデータのベクトルストアの構築完了")


def chunk_documents_by_page(df: pd.DataFrame) -> List[Document]:
    """
    `page_number` ごとにデータをまとめてドキュメントリストを返す

    Args:
        df (pd.DataFrame): CSVデータのDataFrame

    Returns:
        List[Document]: ドキュメントリスト
        List[Document]: ドキュメントリスト
    """
    q_documents = []
    a_documents = []
    grouped = df.groupby(['page_number'])

    for page_number, group in grouped:
        q_text = "\n\n".join(
            [row['faq_question'] for _, row in group.iterrows()]
        )
        a_text = "\n\n".join(
            [row['faq_answer'] for _, row in group.iterrows()]
        )


        # ドキュメントを作成
        q_documents.append(Document(
            page_content=q_text,
            metadata={'page_number': page_number}
        ))
        a_documents.append(Document(
            page_content=a_text,
            metadata={'page_number': page_number}
        ))

    return q_documents, a_documents


def build_bi_vector_stores(
    csv_file: str,
    output_dir: str
):
    """
    CSVファイルごとにベクトルストアを構築し、双方向ストアを作成

    Args:
        csv_file (str): CSVファイルのパス
        output_dir (str): ベクトルストアを保存するディレクトリ
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    # ファイル名（企業名など）を取得
    file_name = os.path.splitext(os.path.basename(csv_file))[0]

    # CSVを読み込む
    df = pd.read_csv(csv_file)
    if df.empty or not {'faq_question', 'faq_answer', 'page_number'}.issubset(df.columns):
        logging.warning(f"[WARN] {file_name} のデータが不正です。スキップします。")
        return

    # ドキュメントをチャンク化
    q_documents, a_documents = chunk_documents_by_page(df)

    # ベクトルストアのディレクトリを作成
    store_dir = os.path.join(output_dir, file_name)
    os.makedirs(store_dir, exist_ok=True)

    # 質問→回答のベクトルストア
    question_documents = [
        Document(
            page_content=doc.page_content,
            metadata={
                'faq_answer': "\n\n".join(
                    df[df['page_number'] == doc.metadata['page_number']]['faq_answer'].tolist()
                )
            }
        )
        for doc in q_documents
    ]
    question_store_dir = os.path.join(store_dir, "faq_question")
    question_vectorstore = Chroma.from_documents(
        documents=question_documents,
        embedding=embeddings,
        persist_directory=question_store_dir
    )
    question_vectorstore.persist()

    # 回答→質問のベクトルストア
    answer_documents = [
        Document(
            page_content=doc.page_content,
            metadata={
                'faq_question': "\n\n".join(
                    df[df['page_number'] == doc.metadata['page_number']]['faq_question'].tolist()
                )
            }
        )
        for doc in a_documents
    ]
    answer_store_dir = os.path.join(store_dir, "faq_answer")
    answer_vectorstore = Chroma.from_documents(
        documents=answer_documents,
        embedding=embeddings,
        persist_directory=answer_store_dir
    )
    answer_vectorstore.persist()

    logging.info(f"[INFO] {file_name} の双方向ベクトルストアを構築しました。")
