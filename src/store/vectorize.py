import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import pandas as pd
import tiktoken

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging  # ロギングのインポート

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("vectorization.log"),
        logging.StreamHandler()
    ]
)

# リポジトリディレクトリの設定
repo_dir = Path(__file__).resolve().parents[2]
if repo_dir.as_posix() not in sys.path:
    sys.path.append(repo_dir.as_posix())

def count_tokens(text: str, encoding_name: str = 'cl100k_base') -> int:
    """
    テキストのトークン数をカウントする。

    Args:
        text (str): トークン数を計算するテキスト。
        encoding_name (str): 使用するエンコーディングの名前。

    Returns:
        int: トークン数。
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

# ドキュメントをチャンクに分割する関数
def split_into_chunks(text: str, max_tokens: int = 1024, overlap: int = 100) -> List[str]:
    """
    テキストを指定したトークン数ごとにチャンクに分割する。中途半端な箇所で分割しない。

    Args:
        text (str): 分割するテキスト。
        max_tokens (int): 各チャンクの最大トークン数。
        overlap (int): チャンク間の重複トークン数。

    Returns:
        List[str]: 分割されたテキストのリスト。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_batch(batch: List[Document], embeddings: OpenAIEmbeddings, vector_store_dir: str, collection_name: str):
    """
    バッチごとにドキュメントをベクトルストアに追加し、保存する。

    Args:
        batch (List[Document]): 処理するドキュメントのバッチ。
        embeddings (OpenAIEmbeddings): エンベディングのインスタンス。
        vector_store_dir (str): ベクトルストアの保存先ディレクトリ。
        collection_name (str): Chromaのコレクション名。

    Returns:
        Chroma: 更新されたベクトルストアのインスタンス。
    """
    try:
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=vector_store_dir,
            collection_name=collection_name
        )
        vectorstore.persist()
        logging.info(f"[INFO] バッチを処理し、{vector_store_dir} に保存しました。")
    except Exception as e:
        logging.error(f"[ERROR] バッチの処理中にエラーが発生しました: {e}")

def vectorize_faqs_from_csv(
    csv_dir: str,
    vector_store_dir: str = "vectorstore",
    collection_name: str = "faq_pages",
    batch_size: int = 100,
    delay_seconds: int = 60,
    max_tokens_per_chunk: int = 1024,
    overlap_tokens: int = 100
) -> Chroma:
    """
    複数のCSVファイルからFAQデータを読み込み、ページごとにベクトル化してChromaに保存。

    Args:
        csv_dir (str): CSVファイルが保存されているディレクトリのパス。
        vector_store_dir (str): ベクトルストアを保存するディレクトリのパス。
        collection_name (str): Chromaのコレクション名。
        batch_size (int): 各バッチで処理するドキュメントの数。
        delay_seconds (int): バッチ間の遅延時間（秒）。
        max_tokens_per_chunk (int): 各ドキュメントチャンクの最大トークン数。
        overlap_tokens (int): チャンク間の重複トークン数。

    Returns:
        Chroma: ベクトルストアのリトリーバーオブジェクト。
    """
    load_dotenv(repo_dir.joinpath(".env.local"))

    # 環境変数からAPIキーを取得
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in .env.local")

    # エンベディングの初期化
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    # CSVファイルの一覧取得
    csv_files = [
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith('.csv')
    ]

    if not csv_files:
        raise ValueError("CSVファイルが見つかりません。")

    documents = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logging.warning(f"[WARN] ファイル {csv_file} の読み込みに失敗しました: {e}")
            continue

        # 必要なカラムが存在するか確認
        required_columns = {'file_name', 'page_number', 'faq_question', 'faq_answer'}
        if not required_columns.issubset(df.columns):
            logging.warning(f"[WARN] ファイル {csv_file} に必要なカラムが不足しています。")
            continue

        # page_number ごとにグループ化
        grouped = df.groupby(['file_name', 'page_number'])

        for (file_name, page_number), group in grouped:
            # 各質問と回答を結合
            qa_pairs = []
            for _, row in group.iterrows():
                question = row['faq_question']
                answer = row['faq_answer']
                qa_pairs.append(f"質問: {question}\n回答: {answer}")

            combined_text = "\n\n".join(qa_pairs)

            # チャンク化
            chunks = split_into_chunks(
                text=combined_text,
                max_tokens=max_tokens_per_chunk,
                overlap=overlap_tokens
            )

            for chunk in chunks:
                # メタデータのクレンジング
                clean_file_name = file_name if isinstance(file_name, (str, int, float, bool)) else str(file_name)
                clean_page_number = page_number if isinstance(page_number, (str, int, float, bool)) else int(page_number)

                document = Document(
                    page_content=chunk,
                    metadata={
                        'file_name': clean_file_name,
                        'page_number': clean_page_number
                    }
                )
                documents.append(document)

    if not documents:
        raise ValueError("ベクトル化するドキュメントが見つかりませんでした。")

    # チャンクをバッチに分割して処理
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            process_batch(batch, embeddings, vector_store_dir, collection_name)
            logging.info(f"[INFO] バッチ {i//batch_size + 1} を処理しました。")
            time.sleep(delay_seconds)
        except Exception as e:
            logging.error(f"[ERROR] バッチ {i//batch_size + 1} の処理中にエラーが発生しました: {e}")
            time.sleep(delay_seconds)
            # 必要に応じて追加のエラーハンドリングを実装可能

    logging.info(f"[INFO] すべてのドキュメントがベクトル化され、保存されました。")

    # 最後にベクトルストアをロードしてリトリーバーを返す
    vectorstore = Chroma(
        persist_directory=vector_store_dir,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    return vectorstore.as_retriever()