import json
import pathlib
import tarfile
import textwrap
import os
from typing import List

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
    output_path = f"./{arxiv_id}"

    if not os.path.exists(output_path):
        source_link = "https://arxiv.org/e-print/" + arxiv_id
        response = requests.get(source_link)
        filename = arxiv_id + ".tar.gz"
        with open(filename, "wb") as f:
            f.write(response.content)

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

def get_arxiv_summary(arxiv_info, reference_idea=None):
    # Load arXiv papers and generate a summary.
    # Download a paper from arXiv, extract its text, and generate a summary using the model.
    # Download the paper.
    # arxiv_link = 'https://arxiv.org/abs/2104.13922'
    # arxiv_link = "https://arxiv.org/abs/2309.17453"
    paper = download_arxiv_latex(arxiv_info["arxiv_id"])
    
    # Placeholder for introduction retrieval
    paper += r"\section{dummy}"

    title = arxiv_info["title"] 
    abstract = arxiv_info["abstract"]

    # The introduction starts with \section{Introduction} and ends when another section starts with \section{...} 
    # It may span multiple lines.
    # Using regular expression to extract the content.
    try:
        introduction = re.search(r'\\section{Introduction}(.*?)\\section{', paper, re.DOTALL).group(1)
    except:
        introduction = ""

    # Then extract each section / subsection to get an idea on what's going on in details. 

    sections = []

    for m in re.finditer(r'\\section{(.*?)}(.*?)\\section{', paper, re.DOTALL):
        sec_title = m.group(1)
        sec_content = m.group(2)

        if sec_title == "Introduction":
            continue
        sections.append(dict(title=sec_title, content=sec_content))

    prompt = '''
    Generate a natural language summary of the following paper, given the title, abstract, introduction and start/end part of each sections. 

    1. First, list 1-2 bullet points to summarize the abstract and introduction. Do not simply copy from the abstract and/or introduction. 
    2. Second, list 1 bullet point for methodology innovation. Please be concise and emphasize the contribution of the paper, i.e., how it is different from existing research works. 
    3. Finally, list 1-2 bullet points to summarize its experimental results. If you cannot find any experimental results, just say "no experiments".  
    '''

    if reference_idea is not None:
        prompt += "4. Also compare the paper with a reference idea and summarize how the reference idea is different from the paper. Reference idea: " + reference_idea + "\n"

    input_data = '''

    Title: {title}
    Abstract: {abstract}
    Introduction: {introduction}
    Sections: {sections}
    '''

    section_str = ""
    for sec in sections:
        section_str += "Section title: " + sec['title'] + "\n"
        section_str += "Section content: "

        content = sec["content"]
        if len(content) < 3000:
            section_str += content
        else:
            section_str += content[:1500] + "\n\n [Content omitted]\n\n" + content[-1500:]

        section_str += "\n" 

    for i in range(2):
        try:
            response = model.generate_content(
                prompt + input_data.format(title=title, abstract=abstract, introduction=introduction, sections=section_str))
            ret = response.text

            if title == "":
                ret = "[No title]" + ret
            if abstract == "":
                ret = "[No abstract]" + ret
            if introduction == "":
                ret = "[No introduction]" + ret

            return ret 
        except:
            pass

    return "Error"

def summarize_keywords(comments : List[str]):
    # Given comments, call the model to summarize the comments into a few keywords for arXiv search.
    prompt = '''
    Generate a few keywords to summarize the following comments. Please return the keywords in json format (e.g., ["keyword1", "keyword2", "keyword3"]).  
    
    Comments: 
    {comments} 
    ''' 

    for i in range(3):
        response = model.generate_content(prompt.format(comments="\n".join(comments)))
        try:
            items = response.text.split("\n")
            return json.loads("\n".join(items[1:-1]))
        except:
            print("Error in summarizing keywords. Retrying..")
            pass 

    return "Summary error!"


if __name__ == "__main__":
    arxiv_link = '2309.17453'
    summary = get_arxiv_summary(arxiv_link)
    print(summary)
