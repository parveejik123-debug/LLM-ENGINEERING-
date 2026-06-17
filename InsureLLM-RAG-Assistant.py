import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import gradio as gr


MODEL = "llama3.2"
db_name = "vector_db"

folders = glob.glob("week5/knowledge-base/*")


documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder,glob="**/*.md",loader_cls=TextLoader,loader_kwargs={'encoding':'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata['doc_type'] = doc_type
        documents.append(doc)

# print(f"Loaded {len(documents)} Documents")
# print(documents[1])

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# print(f"Total number of chunks created are : {len(chunks)}")
# print(f"First chunk is : {(chunks[0])}")

embaddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-V2')
if os.path.exists(db_name):
    Chroma(persist_directory=db_name,embedding_function=embaddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks,embedding=embaddings,persist_directory=db_name)
# print(f"Vectorstore created with {vectorstore._collection.count()} documents")

vectorstore = Chroma(persist_directory=db_name, embedding_function=embaddings)

retriever = vectorstore.as_retriever()
llm = ChatOllama(temperature=0, model=MODEL)

SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

def answer_question(question, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

# print(answer_question("Who is Averi Lancaster?", []))

gr.ChatInterface(
    fn=answer_question,
    type="messages"
).launch(inbrowser=True)