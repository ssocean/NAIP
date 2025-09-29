import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from retry import retry
import Levenshtein
from tqdm import tqdm

API_SECRET_KEY = "sk-xxx"
BASE_URL = "xxx"
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

def normalized_edit_distance(str1, str2):
     
    str1 = str1.strip().lower()
    str2 = str2.strip().lower()

     
    edit_distance = Levenshtein.distance(str1, str2)

     
    max_length = max(len(str1), len(str2))

     
    normalized_distance = edit_distance / max_length if max_length != 0 else 0
    return normalized_distance
usr_prompts = ["Given the following title and abstract of the research paper, identify the core task or problem being addressed in few words. You MUST respond with the keyphrase ONLY in this format: xxx",
               "Based on the given title and abstract, what is the main focus or task of the research? Summarize it in a few words. You MUST respond with the keyphrase ONLY in this format: xxx",
               "Analyze the title and abstract provided to identify the central task or topic of the paper, which will be used as a keyword for searching related academic papers on Google Scholar. Avoid terms that are either too broad (such as 'deep learning' or 'computer vision') or too specific (such as certain model names, unless widely recognized.). You MUST respond with the keyword ONLY in this format: xxx"
               ]
@retry(delay=2)
def get_chatgpt_field(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True, model="gpt-3.5-turbo-0125", temperature=0):
    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic key phrase from paper's title and "
            "abstract. Ensure that the topic key phrase precisely defines the research area within the article. For effective academic searching, such as on Google Scholar, the field should be specifically targeted rather than broadly categorized. For instance, use 'image classification' instead of the general 'computer vision' to enhance relevance and searchability of related literature.")
    if not usr_prompt:
        usr_prompt = ("Analyze the title and abstract provided to identify the central topic of the paper, which will be used as a keyword for searching related academic papers on Google Scholar. Avoid terms that are either too broad (such as 'deep learning' or 'computer vision') or too specific (such as obscure model names, unless widely recognized). Focus on a keyword that reflects the innovative aspect or core methodology of the study. You MUST respond with the keyword ONLY in this format: xxx")

    messages = [SystemMessage(content=sys_content)]

    extra_abs_content = '''
    Given Title: Large Selective Kernel Network for Remote Sensing Object Detection
    Given Abstract: Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, which can vary for different objects. This paper considers these priors and proposes the lightweight Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To our knowledge, large and selective kernel mechanisms have not been previously explored in remote sensing object detection. Without bells and whistles, our lightweight LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46% mAP), DOTA-v1.0 (81.85% mAP), and FAIR1M-v1.0 (47.87% mAP).''' if abstract else ''
    if extra_prompt:
        messages += [HumanMessage(content=f'''{usr_prompt}\n\n{extra_abs_content}'''), AIMessage(content='remote sensing object detection')]

    content = f'''{usr_prompt}\n
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'
    messages.append(HumanMessage(content=content))

    chat = ChatOpenAI(model=model, temperature=temperature)

    return chat.batch([messages])[0].content
import csv
from multiprocessing import Pool
prompt = "Identify the research field from the given title and abstract. You MUST respond with the keyword ONLY in this format: xxx"
def process_row(row):
    title, abs, GT_kwd = row[0], row[1], row[2]
    pred_kwd = get_chatgpt_field(title, abs, usr_prompt=prompt) # This should be replaced with the actual prediction logic
    # Assuming normalized_edit_distance is defined elsewhere

    ned = normalized_edit_distance(GT_kwd, pred_kwd)
    print(f'GT:{GT_kwd} \t Pred:{pred_kwd} \t Ned:{ned}')
    return ned

def main():
    with open(r'TKPD.csv','r', newline='', encoding='gbk') as input_csvfile:
        reader = csv.reader(input_csvfile)
        rows = [row for row in reader]
    print(len(rows))
    with Pool(12) as p:
        results = p.map(process_row, rows)

    average_distance = sum(results) / len(results) if results else 0
    print(f"{prompt}: {average_distance}")

if __name__ == '__main__':
    main()
