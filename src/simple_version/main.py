from view import MainInterface
from controller import MainController

if __name__ == "__main__":
    ui = MainInterface(MainController())
    ui.render()