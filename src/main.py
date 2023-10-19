import os, sys
from dotenv import load_dotenv

load_dotenv("config/.env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


# Step 1. Load
'''
載入文件(僅支援PDF)
'''
from langchain.document_loaders import PyPDFLoader
file_path = "assets/"+input("輸入檔名(assets/XXX.pdf>)：\n")
loader = PyPDFLoader(file_path)
data = loader.load()


# Step 2. Split
'''
選擇splitter，將文件拆分成多個chunk
'''
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


# Step 3. Store
'''
將embedding和splits儲存至vectorstore
'''
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits,
                                    embedding=OpenAIEmbeddings())


# Step 4. Retrieval QA
## query
query = input("輸入訊息(直接enter可結束程式)：\n")
if not query: sys.exit()

## prompt
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate

input_variables = ['question', 'context']
messages = [
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=input_variables,
            template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Respond in the language of the question.\nQuestion: {question} \nContext: {context} \nAnswer:"
        )
    )
]
prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)

##LLM
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 temperature=0)


# Step 5. Generate
# ## RAG prompt template
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")


## QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
result = qa_chain({"query": query})

# Output
print(result["result"])
