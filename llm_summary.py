import pathlib
import tarfile
import textwrap
import os

import requests
import google.generativeai as genai
import re

def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """

    try:
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False

def download_arxiv_latex(arxiv_id):
    # E.g., arxiv id = '2104.13922'
    source_link = "https://arxiv.org/e-print/" + arxiv_id
    response = requests.get(source_link)
    filename = arxiv_id + ".tar.gz"
    with open(filename, "wb") as f:
        f.write(response.content)

    output_path = f"./{arxiv_id}"

    if not os.path.exists(output_path):
        untar(filename, output_path)
        # delete all non .tex files in all the root folders and subfolders to save space
        # and put all tex files into a long text
        # Traverse recursively  
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if not file.endswith(".tex"):
                    os.remove(os.path.join(root, file))
                    
        # also remove the tar.gz file
        os.remove(filename)
    
    # Traverse recursively to put all tex files into a long text
    all_content = ""
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith(".tex"):
                # put them into a long text
                with open(os.path.join(root, file), "r") as f:
                    content = f.read()
                    # all_content += "==== File: " + file + " ====\n" + content
                    all_content += content

    return all_content

genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

model = genai.GenerativeModel('gemini-pro') 

def get_arxiv_summary(arxiv_link):
    # Load arXiv papers and generate a summary.
    # Download a paper from arXiv, extract its text, and generate a summary using the model.
    # Download the paper.
    # arxiv_link = 'https://arxiv.org/abs/2104.13922'
    # arxiv_link = "https://arxiv.org/abs/2309.17453"
    arxiv_id = arxiv_link.split('/')[-1]
    paper = download_arxiv_latex(arxiv_id)

    # Extract the abstract and introduction from paper.
    # The title is within \XXXtitle{...}, where XXX can be any string, e.g., \icmltitle{...}, etc
    # Using regular expression to extract the content.
    title = re.search(r'\\(.*?)title{(.*?)}', paper).group(2).replace('\\\\', '')
    
    # The abstract starts with \begin{abstract} and ends with \end{abstract} and may span multiple lines 
    # Using regular expression to extract the content.
    abstract = re.search(r'\\begin{abstract}(.*?)\\end{abstract}', paper, re.DOTALL).group(1)

    # The introduction starts with \section{Introduction} and ends when another section starts with \section{...} 
    # It may span multiple lines.
    # Using regular expression to extract the content.
    introduction = re.search(r'\\section{Introduction}(.*?)\\section{', paper, re.DOTALL).group(1)

    prompt = '''
    Generate a natural language summary of the following paper. 
    Please be concise and emphasize the contribution of the paper, i.e., how it is different from existing research works. 
    Please use 3-5 sentences to summarize the paper.

    Title: {title}
    Abstract: {abstract}
    Introduction: {introduction}
    '''

    response = model.generate_content(prompt.format(title=title, abstract=abstract, introduction=introduction))
    return response.text
