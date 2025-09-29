def get_filename_without_extension(file_path):
    # Extract the filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    return filename_without_extension


import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import langchain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

langchain.debug = False

import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from langchain_community.chat_models import ChatOpenAI

from database.DBEntity import *

items = [
    "Engaging", "Controversial", "Rigorous", "Innovative", "Accessible", "Methodical", "Concise", "Persuasive",
    "Comprehensive", "Insightful", "Relevant", "Objective", "Replicable", "Structured", "Coherent", "Original",
    "Balanced", "Authoritative", "Impactful", "Interdisciplinary", "Well-sourced", "Technical", "Provocative",
    "Hypothesis-driven", "Ethical", "Difficult to understand", "Exciting", "Not well written", "Theoretical", "To the point",
    "Disengaging", "Uncontroversial", "Lax", "Conventional", "Inaccessible", "Haphazard", "Verbose", "Unconvincing",
    "Superficial", "Uninsightful", "Irrelevant", "Subjective", "Non-replicable", "Unstructured", "Incoherent", "Derivative",
    "Unbalanced", "Unreliable", "Inconsequential", "Narrow", "Poorly-sourced", "Nontechnical", "Unprovocative",
    "Speculation-driven", "Unethical", "Easy to understand", "Dull", "Well written", "Empirical", "Circumlocutory"
]

class GPT_Paper_Response_Fixer(BaseModel):
    IMPACT: float = Field(
        description="The predicted acadmic impact value range from 0 to 1. Results are rounded to two decimal places. e.g., 0.46")


lpqa_parser = PydanticOutputParser(pydantic_object=GPT_Paper_Response_Fixer)


def parse_scores(content):
    # 解析评分内容
    try:
        scores = [int(line.split()[1]) for line in content.split('\n')]
        # 计算均值
        mean_score = sum(scores) / len(scores)
        return mean_score
    except Exception as e:
        print(e)
        return 0

def paper_rating(abstract):
    '''
    This is the original prompt used in the paper "Can ChatGPT be used to predict citation counts, readership, and social media interaction? An exploration among 2222 scientific abstracts"
    Resulting poor performance with NDCG@20 below 0.1.
    '''

    prompt = f"Please rate the following abstract on each of the 60 items from 0 = Not at all to 100 = Very much. Only provide the numbers. For example:\n\n"
    prompt += "1. 65\n2. 50\n3. 5\n4. 95\n5. …\n\n"
    prompt += f"This is the abstract:\n{abstract}\n\n"
    prompt += "These are the items:\n" + "\n".join([f"{i + 1}. {item}" for i, item in enumerate(items)])
    prompt_template = prompt

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0) # Lite Speed
    rst = llm.invoke(prompt_template)
    content = rst.content

    return parse_scores(content)






def chat_qianfan(prompt):
    import qianfan
    # This is used to calculat 'LLaMA-3-generated' in Tab.2. You have to regist qianfan or try another LLM Inference API provider.
    chat_comp = qianfan.ChatCompletion()

    # 指定特定模型
    resp = chat_comp.do(model="Meta-Llama-3-8B", messages=[{
        "role": "user",
        "content": prompt
    }])

    print(resp["body"])
    return resp["body"].get('result')
def paper_rating_improved(row):
    title = row['title']
    abstract = row['abstract']
    prompt = f'''Based on the following information, predict the academic impact of this paper as a single number between 0 and 1. Output only the number, with no additional text:
    Title: {title}
    Abstract: {abstract}
    ONLY output a single number representing the future academic impact between 0 and 1. e.g., 0.69'''

    try:

        impact = float(chat_qianfan(prompt))
        time.sleep(0.5)
    except Exception as e:
        print(e)
        return None

    return impact

def main():
    # 读取数据
    data = pd.read_csv(r'NAID\NAID_test_extrainfo.csv')

    scores = []


    with ThreadPoolExecutor(max_workers=1) as executor:  # 这里将线程数设置为10

        futures = [executor.submit(paper_rating_improved, row) for _, row in data.iterrows()]


        for future in tqdm(as_completed(futures)):
            score = future.result()
            if score:
                scores.append(score)
            else:
                scores.append(-1)

    data['average_score'] = scores
    data = data[data['average_score'] >= 0]

    columns_to_save = ['id', 'cites', 'TNCSI', 'TNCSI_SP', 'abstract', 'average_score']
    data[columns_to_save].to_csv(r'gpt_predict_llama-3.csv', index=False)


import pandas as pd
from sklearn.metrics import ndcg_score


def calculate_ndcg(file_path):

    data = pd.read_csv(file_path)


    if 'average_score' not in data.columns or 'cites' not in data.columns:
        return "The required columns are not in the dataframe."


    y_true = data['TNCSI_SP'].to_numpy()
    y_score = data['average_score'].to_numpy()

    # Reshape data for ndcg calculation (1, -1) as ndcg expects at least 2D arrays
    y_true = y_true.reshape(1, -1)
    y_score = y_score.reshape(1, -1)


    ndcg = ndcg_score(y_true, y_score,k=20)

    return ndcg


#
if __name__ == "__main__":
    main()
    ndcg_value = calculate_ndcg('gpt_predict_improved.csv')
    print(f"The NDCG value is: {ndcg_value}")
