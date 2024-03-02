import json
import pathlib
import tarfile
import textwrap
import os
from typing import List

import requests
import google.generativeai as genai
import re

from arxiv_utils import ArXiv

def shorten_section(title, content, max_length=3000):
    section_str = "Section title: " + title + "\n"
    section_str += "Section content: "

    if len(content) < max_length:
        section_str += content
    else:
        section_str += content[:max_length//2] + "\n\n [Content omitted]\n\n" + content[-max_length//2:]

    section_str += "\n" 
    return section_str

class ModelInterface:
    def __init__(self):
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

        # for m in genai.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(m.name)

        self.model = genai.GenerativeModel('gemini-pro') 

    def call_model(self, prompt, post_process=None, max_retry=3):
        for i in range(max_retry):
            try:
                response = self.model.generate_content(prompt)
                ret = response.text
                if post_process is not None:
                    ret = post_process(ret)
                return ret
            except:
                pass

        return "Error"

    def get_summary(self, paper : ArXiv, reference_idea=None):
        # Summarization of each section. 
        prompt = '''
        Generate a summary of the following section. The summary should be 2-3 sentences, be concise and informative. 
        '''
        if reference_idea is not None:
            prompt += "Also compare the paper with a reference idea. Summarize how the reference idea is different from the paragraph, if the reference idea is relevant. Reference idea: " + reference_idea + "\n"

        input_data = '''

        Title: {section_title}
        Content: {content}
        '''

        sections = paper.sections

        results = dict()
        for sec_title, content in sections.items():
            input_all = prompt + input_data.format(section_title=sec_title, content=content)
            # print(input_all)
            output = self.call_model(input_all)
            results[sec_title] = output
            
        return results

    def summarize_keywords(self, comments : List[str]) -> List[str]:
        # Given comments, call the model to summarize the comments into a few keywords for arXiv search.
        prompt = '''
        Generate a few keywords to summarize the following comments. Please return the keywords in json format (e.g., ["keyword1", "keyword2", "keyword3"]).  
        
        Comments: 
        {comments} 
        ''' 
        def post_process(ret):
            return json.loads("\n".join(ret.split("\n")[1:-1]))

        final_input = prompt.format(comments="\n".join(comments))
        
        return self.call_model(final_input, post_process=post_process)

"""
prompt_summarize_paper = '''
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

for sec in sections:
    section_str += shorten_section(sec["title"], sec["content"], max_length=3000)

"""

if __name__ == "__main__":
    arxiv_link = 'https://arxiv.org/pdf/2402.18510.pdf'
    paper = ArXiv(arxiv_link, download=True)
    model = ModelInterface()
    summary = model.get_summary(paper)
    print(summary)
