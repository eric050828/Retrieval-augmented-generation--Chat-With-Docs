from view import MainInterface
from controller import Controller

if __name__ == "__main__":
    ui = MainInterface(Controller())
    ui.render()