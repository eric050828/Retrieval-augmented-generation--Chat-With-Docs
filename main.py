import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_TOKEN")
os.environ['OPENAI_API_KEY'] = api_key

# Step 1. Load
'''
Specify a DocumentLoader to load in your unstructured data as Documents.
將要載入到非結構化資料中的 指定 DocumentLoader 為 Documents 。

A Document is a dict with text (page_content) and metadata.
A Document 是帶有文字 （ page_content ） 和 metadata 的字典。
'''
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()


# Step 2. Split
'''
Split the Document into chunks for embedding and vector storage.
將其 Document 拆分為塊以進行嵌入和向量存儲。
'''
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


# Step 3. Store
'''
To be able to look up our document splits, we first need to store them where we can later look them up.
為了能夠查找我們的文檔拆分，我們首先需要將它們存儲在以後可以查找的地方。

The most common way to do this is to embed the contents of each document split.
執行此操作的最常見方法是嵌入每個文檔拆分的內容。

We store the embedding and splits in a vectorstore.
我們將嵌入和拆分存儲在向量存儲中。
'''
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings())


# RAG prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")


# LLM
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="text-davinci-003", temperature=0)


# RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
question = "What are the approaches to Task Decomposition?"
result = qa_chain({"query": question})
result["result"]

"""
# Step 4. Retrieve
'''
Retrieve relevant splits for any question using similarity search.
使用相似性搜索檢索任何問題的相關拆分。

This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
這隻是“top K”檢索(RAG-Sequence)，我們根據嵌入與查詢的相似性來選擇文檔。
'''
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)
# ouput: 4


# Step 5. Generate
'''
Distill the retrieved documents into an answer using an LLM/Chat model (e.g., gpt-3.5-turbo).
使用LLM /聊天模型將檢索到的文檔提取為答案（例如， gpt-3.5-turbo ）。

We use the Runnable protocol to define the chain.
我們使用 Runnable 協議來定義鏈。

Runnable protocol pipes together components in a transparent way.
可運行的協定以透明的方式將元件管道連接在一起。

We used a prompt for RAG that is checked into the LangChain prompt hub (here).
我們使用了一個RAG提示，該提示已簽入LangChain 提示中心（此處）。
'''
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
)

rag_chain.invoke("What is Task Decomposition?")
# AIMessage(content='Task decomposition is the process of breaking down a task into smaller subgoals or steps.
#                    It can be done using simple prompting, task-specific instructions, or human inputs.')
"""
