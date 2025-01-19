import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import pandas as pd
import re

from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

repo_dir =  Path(__file__).resolve().parents[2]
if repo_dir.as_posix() not in sys.path:
    sys.path.append(repo_dir.as_posix())


def format_texts_with_gpt4o(
    pdf_data_list: List[Dict[str, Any]],
    output_dir: str = "formatted_texts"
):
    """
    抽出済みのPDFデータリストをGPT-4Oで整形し、
    ファイルごとにテキストファイルを作成して保存する
    各ページごとに見出しを入れて出力する
    
    Args:
        pdf_data_list: load_pdfs_use_pypdfium2 で得たリスト
        output_dir: 出力先ディレクトリ
    """
    # 整形用のプロンプトチェーンを定義
    formatting_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """あなたはテキスト整形の専門家です。
            ユーザーから与えられたテキストを整形してください。
            与えられたテキストはPDFから抽出したため、
            文字化けが発生している場合がありますが、適宜補完してください。
            元のPDFに表が含まれている場合がありますが、
            テキストではその情報が抜け落ちているため、表と思われる不自然な改行がある場合は表の形に直してください。
            表の形式はマークダウン形式にしてください。
            また、整形したテキスト以外は応答に含めないでください。"""
        ),
        ("user", "{text}")
    ])

    load_dotenv(repo_dir.joinpath(".env.local"))

    # 環境変数からAPIキーを取得
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in .env.local")

    # GPT-4O 使用
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
        temperature=0)

    formatting_chain = LLMChain(
        prompt=formatting_prompt,
        llm=llm
    )

    os.makedirs(output_dir, exist_ok=True)

    for item in pdf_data_list:
        file_name = item["file_name"]
        pdf_docs = [item["first_page"]] + item["other_pages"]

        output_txt_path = os.path.join(output_dir, f"{file_name}_formatted.txt")

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for idx, doc in enumerate(pdf_docs, start=1):
                input_text = doc.page_content

                response_str = formatting_chain.run({"text": input_text})

                f.write(f"=== {file_name} - Page {idx} ===\n")
                f.write(response_str.strip() + "\n\n")

        print(f"[INFO] Formatted: {file_name} -> {output_txt_path}")


def summarize_formatted_texts_with_gpt4o(
    formatted_texts_dir: str,
    summary_output_dir: str = "summaries",
):
    """
    フォーマット済みのテキストファイルを読み込み、ページごとに要約を生成して保存します。
    
    Args:
        formatted_texts_dir (str): フォーマット済みテキストファイルが保存されているディレクトリのパス。
        summary_output_dir (str): 要約を保存するディレクトリのパス。
        max_retries (int): リトライの最大回数。
        backoff_factor (float): リトライ間の待機時間の増加率。
        sleep_time (float): バッチ間の待機時間（秒）。
    
    Raises:
        ValueError: 必要な環境変数が設定されていない場合。
    """
    load_dotenv(repo_dir.joinpath(".env.local"))

    # 環境変数からAPIキーを取得
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in .env.local")

    # 要約用のプロンプトテンプレートを定義
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """あなたは優秀なFAQ作成者です。
            FAQの質問・回答は、前提条件を含めるなど可能な限り詳細な文面にしてください。
            ユーザーから与えられた内容を網羅する質問と回答を可能な限り大量に作成してください。
            ただし、質問は以下のような内容を想定しており、具体的な名称や数値、ポリシー等、明確な回答を作成できる内容以外は無視してください。
            ・xxxグループの2025年3月期の受注高の計画は前期比何倍か
            ・2023年でxxxコーポレーションの一人当たりの年間消費量が最も多い国はどこか
            ・2023年度のxxx会社の海外事業において、コア営業利益が2番目に高い地域に含まれる国として記載がある国名を全て教えてください
            ・2024年3月期のxxx社のセグメント別売上高の中で、2番目に売上高が大きいセグメントの販売先を答えよ
            ・xxxグループの2023年度の従業員の平均年収は約何万円でしょうか
            ・xxx会社において、2023年度企業市民活動の費用が高いのは、北米と社会福祉分野のどちらか
            ・xxx社の社内ベンチャー制度の名前は
            ・xxxコーポレーションの経営理念は何ですか
            ・xxxの事業所の数は全部で何拠点ですか
            ・xxxの事業所をすべて挙げてください？
            ・xxx社の最も小さい天然水の森の面積は約何ヘクタールですか？
            回答も詳細に作成し、"2022年のxxx社の売上高は18兆7,148億円です。前年比78.4％増の売り上げを記録しています" のように具体的にしてください。
            また、ユーザーから与えられた内容以外は応答に含めないでください。
            ユーザーへの応答は下記のような形式のJSON文字列としてください。
            {{"faqs": [{{"question": 質問, "answer": 回答}},...]}}"""
        ),
        ("user", "{text}")
    ])

    # GPT-4oの設定
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
        temperature=0,
    )

    # LLMChainの作成
    summary_chain = LLMChain(
        prompt=summary_prompt,
        llm=llm
    )

    # 要約の保存先ディレクトリを作成
    os.makedirs(summary_output_dir, exist_ok=True)

    # フォーマット済みテキストファイルの一覧取得
    file_paths = [
        os.path.join(formatted_texts_dir, f)
        for f in os.listdir(formatted_texts_dir)
        if f.endswith('.txt')
    ]

    if not file_paths:
        raise ValueError(f"指定されたディレクトリ '{formatted_texts_dir}' にテキストファイルが存在しません。")

    for file_path in file_paths:
        file_name = os.path.basename(file_path).split('.')[0]
        output_summary_path = os.path.join(summary_output_dir, f"{file_name}_summarized.txt")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # ページごとに分割
        pages = content.split('=== ')[1:]  # 最初の要素は空文字列となるためスキップ

        with open(output_summary_path, 'w', encoding='utf-8', newline='') as csv_file:
            csv_writer = pd.DataFrame(columns=["file_name", "page_number", "faq_question", "faq_answer"])

        all_faqs = []
            
        for page in pages:
            try:
                header, page_content = page.split(' ===\n', 1)
                # ヘッダーからページ番号を抽出
                _, page_number_str = header.rsplit(' - Page ', 1)
                page_number = int(page_number_str)
            except ValueError as e:
                print(f"[WARN] ファイル '{file_name}' のページヘッダーの解析に失敗しました: {e}")
                continue

            summary = summary_chain.run({"text": page_content})

            try:
                # JSON文字列を辞書に変換
                summary_data = json.loads(summary)
                faqs = summary_data.get("faqs", [])
            except json.JSONDecodeError as e:
                print(f"[WARN] 無効なJSONレスポンスをスキップします: {e}")
                print(f"レスポンス内容: {summary}")
                continue

            if not faqs:
                print(f"[INFO] ファイル '{file_name}' のページ {page_number} の要約が空です")
                continue

            for faq in faqs:
                faq["file_name"] = file_name
                faq["page_number"] = page_number

            all_faqs.extend(faqs)

        if all_faqs:
            faq_df = pd.DataFrame(all_faqs)
            faq_df = faq_df[["file_name", "page_number", "question", "answer"]]
            faq_df.rename(columns={"question": "faq_question", "answer": "faq_answer"}, inplace=True)
            faq_df.to_csv(output_summary_path, index=False, encoding='utf-8-sig')

        print(f"[INFO] 要約ファイルを作成しました: {output_summary_path}")


def summarize_texts_with_gpt4o(
    pdf_data_list: List[Dict[str, Any]],
    summary_output_dir: str = "summaries",
):
    """
    抽出済みのPDFデータリストを読み込み、ページごとに要約を生成して保存します。

    Args:
        pdf_data_list (List[Dict[str, Any]]): PDFデータリスト（ファイル名と各ページの内容を含む）。
        summary_output_dir (str): 要約を保存するディレクトリのパス。

    Raises:
        ValueError: 必要な環境変数が設定されていない場合。
    """
    load_dotenv(repo_dir.joinpath(".env.local"))

    # 環境変数からAPIキーを取得
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in .env.local")

    # 要約用のプロンプトテンプレートを定義
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """あなたは優秀なFAQ作成者です。
            FAQの質問・回答は、前提条件を含めるなど可能な限り詳細な文面にしてください。
            ユーザーから与えられた内容を網羅する質問と回答を可能な限り大量に作成してください。
            ただし、質問は以下のような内容を想定しており、具体的な名称や数値、ポリシー等、明確な回答を作成できる内容以外は無視してください。
            以下は質問の例です。
            ・xxxグループの2025年3月期の受注高の計画は前期比何倍か
            ・2023年でxxxコーポレーションの一人当たりの年間消費量が最も多い国はどこか
            ・2023年度のxxx会社の海外事業において、コア営業利益が2番目に高い地域に含まれる国として記載がある国名を全て教えてください
            ・2024年3月期のxxx社のセグメント別売上高の中で、2番目に売上高が大きいセグメントの販売先を答えよ
            ・xxxグループの2023年度の従業員の平均年収は約何万円でしょうか
            ・xxx会社において、2023年度企業市民活動の費用が高いのは、北米と社会福祉分野のどちらか
            ・xxx社の社内ベンチャー制度の名前は
            ・xxxコーポレーションの経営理念は何ですか
            ・xxxの事業所の数は全部で何拠点ですか
            ・xxxの事業所をすべて挙げてください？
            ・xxx社の最も小さい天然水の森の面積は約何ヘクタールですか？
            回答も詳細に作成し、"2022年のxxx社の売上高はx兆x,xxx億円です。前年比xx.x％増の売り上げを記録しています" のように具体的にしてください。
            また、ユーザーから与えられた内容以外は応答に含めないでください。
            与えられたテキストはPDFから抽出しており、整形していないことを考慮してください。
            表と思われる箇所の情報は数値データを含んでいるため、質問と回答を大量に作成できる可能性があります
            ユーザーへの応答は、下記のようなJSON文字列としてください。
            {{"faqs": [{{"question": 質問, "answer": 回答}},...]}}"""
        ),
        ("user", "{text}")
    ])

    # GPT-4Oの設定
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
        temperature=0,
    )

    # LLMChainの作成
    summary_chain = LLMChain(
        prompt=summary_prompt,
        llm=llm
    )

    # 要約の保存先ディレクトリを作成
    os.makedirs(summary_output_dir, exist_ok=True)

    for item in pdf_data_list:
        file_name = item["file_name"]
        pdf_docs = [item["first_page"]] + item["other_pages"]

        output_summary_path = os.path.join(summary_output_dir, f"{file_name}_summarized.csv")

        all_faqs = []

        for idx, doc in enumerate(pdf_docs, start=1):
            input_text = doc.page_content

            try:
                summary = summary_chain.run({"text": input_text})
                summary = summary.strip()
                summary = summary.replace('\n', '').replace('\r', '').replace('\t', '').replace('\s', '')
                summary = re.sub(r'```json\s*', '', summary)
                summary = re.sub(r'```', '', summary)


                # JSON文字列を辞書に変換
                summary_data = json.loads(summary)
                faqs = summary_data.get("faqs", [])

            except json.JSONDecodeError as e:
                print(f"[WARN] ファイル '{file_name}' のページ {idx} のレスポンスが無効なJSON形式です: {e}")
                print(f"レスポンス内容: {summary}")
                continue
            except Exception as e:
                print(f"[ERROR] ファイル '{file_name}' のページ {idx} の要約生成に失敗しました: {e}")
                continue

            print(f"ファイル '{file_name}' のページ {idx} のレスポンス: {summary}")

            if not faqs:
                print(f"[INFO] ファイル '{file_name}' のページ {idx} の要約が空です")
                continue

            for faq in faqs:
                faq["file_name"] = file_name
                faq["page_number"] = idx

            all_faqs.extend(faqs)

        if all_faqs:
            faq_df = pd.DataFrame(all_faqs)
            faq_df = faq_df[["file_name", "page_number", "question", "answer"]]
            faq_df.rename(columns={"question": "faq_question", "answer": "faq_answer"}, inplace=True)
            faq_df.to_csv(output_summary_path, index=False, encoding='utf-8-sig')

        print(f"[INFO] 要約ファイルを作成しました: {output_summary_path}")