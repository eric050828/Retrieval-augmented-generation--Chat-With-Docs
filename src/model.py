from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pathlib import Path

class Model:
    def __init__(self, ):
        load_dotenv("../config/.env")
        self.ai_model ="gpt-3.5-turbo"
        self.LLM = ChatOpenAI(model_name=self.ai_model, 
                              temperature=0,)
        self.VECTORSTORE_DIRECTORY = "vectordb"
        self.EMBEDDING = OpenAIEmbeddings()
        self.vectordb = Chroma(self.VECTORSTORE_DIRECTORY, self.EMBEDDING)
        # prompt
        self.prompt = self.generatePrompt()

    def generatePrompt(self,):
        input_variables = ["question", "context"]
        prompt_template = "You are a very powerful assistant for question-answering tasks. Please use the retrieved context pieces to answer the questions, step-by-step reasoning and inference to provide deeper and more accurate information. If you don't know the answer, simply say you don't know to avoid uncertain or inaccurate results. Ensure that your responses are based on reliable sources, comprehensive data retrieval, and known information.\nQuestion: {question} \nContext: {context} \nAnswer:"
        messages = [
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=input_variables,
                    template=prompt_template
                )
            )
        ]
        prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)
        return prompt

    def fileIsExisted(self, file_path: str):
        """檢查檔案是否已被儲存在vectorDB中"""
        retrieved_values = self.vectordb.get(where={"source":file_path})
        # print(retrieved_values)
        for metadata in retrieved_values["metadatas"]:
            if file_path == metadata["source"]:
                print("This file has been saved.")
                return True
        print("This file has not been saved yet.")
        return False
        

    def load(
        self, 
        file_type: str, 
        file_path: str, 
        configs: tuple, 
    ):
        # Step 1. Load
        """載入檔案"""
        print("Loading...")
        if file_type == ".pdf":
            can_extract_images = configs
            loader = PyPDFLoader(file_path, extract_images=can_extract_images)

        elif file_type == ".txt":
            loader = TextLoader(file_path)

        data = loader.load()
        print("done\n")
        return data
    
    def split(
        self, 
        data
    ):
        # Step 2. Split
        """選擇splitter，將文件拆分成多個chunk"""
        print("Splitting...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        print("done\n")
        return all_splits
    
    def store(
        self, 
        all_splits
    ):
        # Step 3. Store
        """將embedding和splits儲存至vectordb"""
        print("Storing...")
        self.vectordb = Chroma.from_documents(
            documents=all_splits,
            embedding=self.EMBEDDING,
            persist_directory=self.VECTORSTORE_DIRECTORY,
        )
        self.vectordb.persist()
        print("done\n")
        return
    
    def retrieval(
        self, 
        file_path: str
    ):
        """建立QA Chain，並檔案中檢索出context"""
        print("Retrieving...")
        # Step 4. Retrieval QA
        ## QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.LLM,
            retriever=self.vectordb.as_retriever(search_kwargs={"filter": {"source":file_path}}),
            chain_type_kwargs={"prompt": self.prompt}
        )
        print("done\n")
        return 
    
    def generate(
        self, 
        query: str
    ) -> str:
        """根據query和context生成回應"""
        print("Generating...")
        # Step 5. Generate
        result = self.qa_chain({"query": query})
        print("done\n")
        return result["result"]

    def storeFileIntoDataBase(
        self, 
        file_path: str, 
        configs: tuple,
    ) -> bool:
        """將檔案存入資料庫中，步驟包含
           1.load 
           2.split 
           3.store
        """
        if not self.fileIsExisted(file_path):
            file_type = Path(file_path).suffix
            data = self.load(file_type, file_path, configs)
            # print("data:",data)
            all_splits = self.split(data)
            # print("Chunks:",all_splits)
            self.store(all_splits)
            return True
        return False

    def generateResponse(
        self, 
        file_path: str, 
        query: str, 
    ) -> str:
        self.retrieval(file_path)
        response = self.generate(query)
        return response