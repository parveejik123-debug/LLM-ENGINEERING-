import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from openai import OpenAI
import requests


knowledge = {}
file_path = glob.glob("week5/knowledge-base/employees/*")
for file_name in file_path:
    name = Path(file_name).stem.split()[-1]
    with open(file_name,"r",encoding='utf-8') as f:
        knowledge[name.lower()] = f.read()


def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    relevant_context = []
    words = text.lower().split()
    for word in words:
        if word in knowledge:
            relevant_context.append(knowledge[word])
    return relevant_context

def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result

# print(additional_context("Park"))

SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""

# requests.get("http://localhost:11434").content

# !ollama pull llama3.2



MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')


def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = ollama.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
