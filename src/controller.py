# controller.py
from model import Model

class Controller:
    def __init__(self):
        self.bot = Model()
    
    def getResponse(self, file_type, query, **kwargs):
        return self.bot.generateResponse(file_type, query, **kwargs)