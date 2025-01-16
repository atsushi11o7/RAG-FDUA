import os
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader

def load_pdfs(pdf_file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    複数のPDFファイルを読み込み、ファイルごとに以下の情報をまとめたリストを返す関数
      [
        {
          "file_name": <ファイル名>,
          "first_page": Document (最初のページ),
          "other_pages": [Document, Document, ...] (2ページ目以降のリスト)
        },
        ...
      ]

    Args:
        pdf_file_paths (List[str]): PDFファイルのパスのリスト

    Returns:
        results (List[Dict[str, Any]]): PDFファイルごとのデータ
    """
    results = []

    for file_path in pdf_file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()  # 1ページ = 1 Document のリスト

        # 空ファイルの場合スキップ
        if not pdf_docs:
            continue

        # 最初のページ
        pdf_docs[0].metadata["file_name"] = file_name
        pdf_docs[0].metadata["page_number"] = 1
        first_page = pdf_docs[0]

        # 2ページ目以降
        other_pages = []
        if len(pdf_docs) > 1:
            for i, doc in enumerate(pdf_docs[1:], start=2):
                doc.metadata["file_name"] = file_name
                doc.metadata["page_number"] = i
                other_pages.append(doc)

        results.append({
            "file_name": file_name,
            "first_page": first_page,      # Document
            "other_pages": other_pages,    # List[Document]
        })
    
    return results