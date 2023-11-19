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


class Model:
    def __init__(self, ):
        load_dotenv("../config/.env")
        self.LLM = ChatOpenAI(model_name="gpt-3.5-turbo", 
                              temperature=0,)
        self.VECTORSTORE_DIRECTORY = "vectordb"

    def loadText(self, data):
        with open("assets/temp.txt","w") as f:
            f.write(data)
        loader = TextLoader("assets/temp.txt")
        data = loader.load()
        return data

    def loadPdf(self, file_paths:list, can_extract_images:bool):
        # Step 1. Load
        '''
        載入文件(僅支援PDF)
        '''
        # file_path = "assets/"+file_name
        print("file_path: ", file_paths)
        assert len(file_paths) == 1, "Only one file can be uploaded at a time."
        loader = PyPDFLoader(file_paths[0], extract_images=can_extract_images)
        data = loader.load()
        return data
    
    def split(self, data):
        # Step 2. Split
        '''
        選擇splitter，將文件拆分成多個chunk
        '''
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        return all_splits
    
    def store(self, all_splits):
        # Step 3. Store
        '''
        將embedding和splits儲存至vectordb
        '''

        self.vectordb = Chroma.from_documents(
            documents=all_splits,
            embedding=OpenAIEmbeddings(),
            persist_directory=self.VECTORSTORE_DIRECTORY,
        )
        self.vectordb.persist()
        return
    
    def retrival(self,):
        # Step 4. Retrieval QA

        ## prompt
        input_variables = ['question', 'context']
        prompt_template = "You are a very powerful assistant for question-answering tasks. Please use the retrieved context pieces to answer the question and step-by-step reasoning to provide more in-depth and accurate information. If you don't know the answer, simply say you don't know to avoid uncertainty or inaccurate results. Ensure that your responses are based on reliable sources and comprehensive information retrieval.\nQuestion: {question} \nContext: {context} \nAnswer:"
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
    
    def generate(self, prompt, query):
        # Step 5. Generate
        ## QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.LLM,
            retriever=self.vectordb.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        result = qa_chain({"query": query})

        # Output
        return result["result"]
    
    def generateResponse(self, file_type, query, **kwargs):
        loaders={"text":self.loadText, "pdf":self.loadPdf,}
        data = loaders[file_type](**kwargs)

        all_splits = self.split(data)

        self.store(all_splits)

        prompt = self.retrival()

        response = self.generate(prompt, query)
        return response