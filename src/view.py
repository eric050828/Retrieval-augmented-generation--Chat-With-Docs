# view.py
import gradio as gr
from controller import Controller


class MainInterface:
    def __init__(self, controller:Controller) -> None:
        self.controller = controller

    def textGetResponse(self, query, data, history):
        history = history or []
        kwargs = {"data":data}
        response = self.controller.getResponse("text", query, **kwargs)
        history.append((query, response))
        return history, history
    
    def pdfGetResponse(self, query, file_paths, can_extract_images, history):
        history = history or []
        kwargs = {"file_paths":file_paths, 
                  "can_extract_images":can_extract_images,}
        response = self.controller.getResponse("pdf", query, **kwargs)
        history.append((query, response))
        return history, history

    def render(self):
        with gr.Blocks() as text_loader:
            with gr.Row():
                with gr.Column():
                    data = gr.Textbox(label="Reference data", placeholder="type somthing...", lines=5, max_lines=20)

                    question = gr.Textbox(label="Question", placeholder="Enter the message", type="text", max_lines=10)

                    submit_btn = gr.Button("Submit")
                with gr.Column():
                    chat = gr.Chatbot(height=600)
                    state = gr.State()

            inputs = [question, data, state]
            outputs = [chat, state]
            submit_btn.click(fn=self.textGetResponse, 
                             inputs=inputs, 
                             outputs=outputs,)

        with gr.Blocks() as pdf_loader:
            with gr.Row():
                with gr.Column():
                    file_paths = gr.FileExplorer(label="Upload PDF file", glob="assets/*",)
                    
                    question = gr.Textbox(label="Question", placeholder="Enter the message", type="text", max_lines=10)

                    extract_images = gr.Checkbox(label="Extract images from PDF", default=False)

                    submit_btn = gr.Button("Submit")
                with gr.Column():
                    chat = gr.Chatbot(height=600)
                    state = gr.State()

            inputs = [question, file_paths, extract_images, state]
            outputs = [chat, state]
            submit_btn.click(fn=self.pdfGetResponse, 
                             inputs=inputs, 
                             outputs=outputs,)

        tabs = gr.TabbedInterface(
            [text_loader, pdf_loader],  # tabs
            ["Text", "PDF"],  # name
            title=""
        )
        # Gradio UI 介面
        tabs.launch(
            debug=True
        )
    
