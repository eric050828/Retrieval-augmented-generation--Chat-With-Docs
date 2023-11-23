import os
from pathlib import Path

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI


class RAGModel:
    """This model is used for communicating with the vectorDB."""
    def __init__(
        self, 
        api_key: str
    ) -> None:
        self.set_api_key(api_key)
        self.ai_model ="gpt-3.5-turbo"
        self.temperature = 0
        self.LLM = ChatOpenAI(model_name=self.ai_model, 
                              temperature=self.temperature,)
        self.prompt = self.generate_prompt()
        self.VECTORSTORE_DIRECTORY = "vectordb"
        self.EMBEDDING = OpenAIEmbeddings()
    

    def generate_prompt(
        self, 
    ) -> ChatPromptTemplate:
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


    def retrieve(
        self, 
        name: str
    ):
        """Retrieve data from the specified collection_name and create a QA_Chain."""
        vectordb = Chroma(
            collection_name=name, 
            persist_directory=self.VECTORSTORE_DIRECTORY,
            embedding_function=self.EMBEDDING)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.LLM,
            retriever=vectordb.as_retriever(),
            chain_type_kwargs={"prompt": self.prompt}
        )
        return qa_chain


    def generate_response(
        self, 
        collection_name: str, 
        query: str
    ) -> str:
        """Generate a response using LLM based on the user's query and retrieved data."""
        qa_chain = self.retrieve(collection_name)
        response = qa_chain({"query": query})
        return response["result"]


    def set_api_key(
        self, 
        api_key: str
    ) -> None:
        """Set up api key from the user."""
        os.environ["OPENAI_API_KEY"] = api_key
        

class DatabaseModel:
    """This model is used for generating responses."""
    def __init__(
        self, 
        api_key: str
    ) -> None:
        self.set_api_key(api_key)
        self.VECTORSTORE_DIRECTORY = "vectordb"
        self.chunk_size = 4000
        self.EMBEDDING = OpenAIEmbeddings()


    def load(
        self, 
        loader
    ):
        """Loading file."""
        data = loader.load()
        return data


    def split(
        self, 
        data
    ):
        """Split the text into several chunks using a splliter."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=0)
        chunks = text_splitter.split_documents(data)
        return chunks


    def store(
        self, 
        name: str, 
        chunks
    ) -> None:
        """Store embeddings and chunks into the vectorDB."""
        vectordb = Chroma.from_documents(
            collection_name=name, 
            documents=chunks, 
            embedding=self.EMBEDDING, 
            persist_directory=self.VECTORSTORE_DIRECTORY, 
        )
        vectordb.persist()


    def set_api_key(
        self, 
        api_key: str
    ) -> None:
        """Set up api key from the user."""
        os.environ["OPENAI_API_KEY"] = api_key