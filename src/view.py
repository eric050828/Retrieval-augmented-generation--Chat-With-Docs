# view.py
import gradio as gr
from controller import Controller


class Tab:
    def __init__(
        self, 
        controller: Controller
    ):
        self.controller = controller
        with gr.Blocks() as self.page:
            # upload
            file_path = gr.FileExplorer(label="Choose a file", glob="assets/*",file_count="single",)
            extract_images = gr.Checkbox(label="Extract images from PDF")
            submit_btn = gr.Button("Upload File!")
            output = gr.Textbox(show_label=False, value="Waiting for upload file...")
            inputs = [file_path, extract_images]
            outputs = [output]
            submit_btn.click(fn=self.uploadFile, inputs=inputs, outputs=outputs)
            # QA
            self.chat = gr.Chatbot(bubble_full_width=False)
            state = gr.State()
            with gr.Row():
                with gr.Column():
                    question = gr.Textbox(label="Question", placeholder="Enter the message", type="text", max_lines=10)
                    with gr.Row():
                        submit_btn = gr.Button("Submit")
                        undo_btn = gr.Button("Undo")
                        clear_btn = gr.Button("Clear")
            inputs = [file_path, question, state,]
            outputs = [self.chat, state]
            submit_btn.click(fn=self.getResponse, 
                             inputs=inputs, 
                             outputs=outputs,)
            undo_btn.click(fn=self.undo, 
                           inputs=state)
            clear_btn.click(fn=self.clear,
                            inputs=state)

    def undo(self,history):
        history.pop()
        return
    
    def clear(self, history):
        history = gr.State()
        return

    def uploadFile(
        self, 
        file_path: str, 
        *configs
    ) -> str:
        """上傳檔案"""
        self.controller.uploadFile(file_path, configs)
        self.chat.visible = True
        return "Upload completed!"

    def getResponse(
        self, 
        file_path: str, 
        query: str, 
        history, 
    ) -> str:
        history = history or []
        response = self.controller.getResponse(file_path, query, )
        history.append((query, response))
        return history, history

class MainInterface:
    def __init__(
        self, 
        controller:Controller
    ) -> None:
        self.controller = controller

    def newTab(
        self, 
        name: str,
    ):
        """建立新聊天室(Broken)"""
        new_tab = Tab(self.controller)
        self.tabs.insert(-1,new_tab.page)
        self.names.append(name)

    def render(self):
        with gr.Blocks() as setting_tab:
            with gr.Column():
                tab_name = gr.Textbox()
            with gr.Column():
                create_btn = gr.Button("Create a new tab")
            create_btn.click(fn=self.newTab,inputs=[tab_name],outputs=[])
        
        self.tabs = [Tab(self.controller).page, setting_tab]
        self.tab_names = ["New Chat", "Settings", ]
        self.interface = gr.TabbedInterface(
            self.tabs,
            self.tab_names,
            title="Chat with Docs!"
        )
        self.interface.launch()
    
