# view.py
import gradio as gr
from controller import Controller

class MainInterface:
    def __init__(
        self, 
        controller:Controller
    ) -> None:
        self.controller = controller

    def render(self):
        settings_tab = Settings(self).page  # 設定頁

        self.tabs = [Tab(self.controller).page, Tab(self.controller).page, settings_tab]
        self.tab_names = ["New Chat 1", "New Chat 2", "Settings"]
        self.interface = gr.TabbedInterface(
            self.tabs,
            self.tab_names,
            title="Chat with Docs!"
        )
        self.interface.launch()



class Tab:
    def __init__(
        self, 
        controller: Controller
    ):
        self.controller = controller
        self.render()
        
    def render(self,):
        with gr.Blocks() as self.page:
            # upload
            file_path = gr.FileExplorer(label="Choose a file", glob="assets/*",file_count="single",visible=False)
            files = gr.File(file_count=1, type="file")
            with gr.Accordion("Advanced Options...", open=False):
                gr.Markdown("### PDF")
                extract_images = gr.Checkbox(label="Extract images from PDF")
                
            upload_btn = gr.Button("Upload File!")
            output = gr.Markdown(show_label=False, value="## Waiting for file upload...")
            inputs = [files, extract_images]
            outputs = [output]
            upload_btn.click(fn=self.uploadFile, inputs=inputs, outputs=outputs)
            # QA
            self.chat = gr.Chatbot(height=900, bubble_full_width=False)
            state = gr.State()
            with gr.Row():
                with gr.Column():
                    question = gr.Textbox(label="Question", placeholder="Enter the message", type="text", max_lines=10)
                    with gr.Row():
                        submit_btn = gr.Button("Submit")
                        undo_btn = gr.Button("Undo")
                        clear_btn = gr.Button("Clear")
            inputs = [files, question, state,]
            outputs = [self.chat, state]
            submit_btn.click(fn=self.getResponse, 
                             inputs=inputs, 
                             outputs=outputs, 
                             scroll_to_output=True, )
            undo_btn.click(fn=self.undo, 
                           inputs=state)
            clear_btn.click(fn=self.clear,
                            inputs=state)

    def undo(self, history):
        history.pop()
        return
    
    def clear(self, history):
        history = gr.State()
        return

    def uploadFile(
        self, 
        files, 
        *configs
    ) -> str:
        """上傳檔案"""
        self.controller.uploadFile(files[0].name, configs)
        self.chat.visible = True
        return "## File upload completed!"

    def getResponse(
        self, 
        files, 
        query: str, 
        history, 
    ) -> str:
        history = history or []
        response = self.controller.getResponse(files[0].name, query, )
        history.append((query, response))
        return history, history

class Settings:
    def __init__(
        self, 
        interface: MainInterface
    ) -> None:
        self.interface = interface
        self.controller = self.interface.controller
        self.render()

    def render(self):
        with gr.Blocks() as self.page:
            # 新增新聊天室
            with gr.Row():
                tab_name = gr.Textbox(label="New Chat", placeholder="Enter the chat name", scale=2)
                create_btn = gr.Button("Create a new chat!", scale=1, interactive=False)
                create_btn.click(fn=self.newTab, inputs=[tab_name], outputs=[])
            
            # 設定AI模型
            with gr.Row():
                choises = [
                    ("gpt-3.5-turbo","gpt-3.5-turbo"), 
                    ("gpt-4","gpt-4"), 
                ]
                radio = gr.Radio(label="Choose a AI model", choices=choises, value="gpt-3.5-turbo")
                radio.change(fn=self.changeModel, inputs=radio, outputs=None)
    
    def newTab(
        self, 
        name: str,
    ) -> None:
        """建立新聊天室(Broken)"""
        # new_tab = Tab(self.controller).page
        # self.tabs.insert(-1, new_tab)
        self.tabs = [Tab(self.controller).page] + self.tabs
        self.names.append(name)

    def changeModel(
        self, 
        model_name: str
    ) -> None:
        """更改AI模型"""
        self.controller.setAiModel(model_name)


