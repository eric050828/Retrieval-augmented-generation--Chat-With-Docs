# controller.py
from model import Model

class Controller:
    def __init__(self):
        self.model = Model()
    
    def getResponse(
        self, 
        file_path: str, 
        query: str, 
    ) -> str:
        """取得回應"""
        return self.model.generateResponse(file_path, query)
    
    def uploadFile(
        self, 
        file_path: str, 
        configs: str,
    ) -> bool:
        """存入檔案"""
        return self.model.storeFileIntoDataBase(file_path, configs)