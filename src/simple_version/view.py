import os

from pathlib import Path
import gradio as gr
from controller import MainController


class MainInterface:
    def __init__(
        self, 
        controller:MainController
    ) -> None:
        self.controller = controller


    def render(self):
        with gr.Blocks(title="Chat with Docs!") as self.interface:
            gr.HTML("<h1 style='text-align:center; font-size:48px'><b>Chat with Docs!</b></h1>")
            info = gr.HTML("""<div style='font-size:16px; margin-left:2.5%; margin-right:2.5%'>
<p>This application utilizes Gradio as its GUI, employing GPT-3.5-turbo and GPT-4 as the primary Large Language Models (LLMs). This app implements the Retrieval Augmented Generation (RAG) application using Langchain.</p>
<p>This version is simplified, and all uploaded information (including OpenAI API key, uploaded files, prompts, detailed settings, conversation logs, etc.) will be removed entirely when the webpage is closed or refreshed.</p>
<p></p>
<p>For detailed usage instructions, please refer to <a href='https://github.com/eric050828/Retrieval-augmented-generation--Chat-With-Docs/tree/master/src/simple_version/README.md'>README.md.</a></p>
                        </div>""")
            file_title = gr.Markdown("# File: *Upload a file first...*", )
            with gr.Row():
                with gr.Column(scale=1):
                    # Set API key
                    with gr.Group():
                        api_key = gr.Textbox(label="Open AI API key", info="Never let others know your key. Don't worry, I won't know it :)", type="password", placeholder="input your api key here...", autofocus=True)
                        
                        setup_btn = gr.Button("Setup", variant="stop",)
                        setup_btn.click(
                            fn=self.set_api_key, 
                            inputs=api_key,
                            show_progress="full")
                    
                    # Upload File
                    with gr.Accordion("Upload File"):
                        file = gr.File(label="Upload a PDF file", file_count="single", file_types=[".pdf"])
                        can_extract_images = gr.Checkbox(label="PDF extract images", info="If your PDF contains images, please open it before uploading.")
                        inputs = [file, can_extract_images]
                        outputs = [file_title]
                        upload_btn = gr.Button("Upload", variant="primary",)
                        upload_btn.click(fn=self.upload_file, 
                                         inputs=inputs, 
                                         outputs=outputs, 
                                         show_progress="full")
                    
                    # Advanced Options
                    with gr.Accordion("Advanced Options"):
                        prompt = gr.Textbox(label="Prompt", placeholder="You are a professional...", info="Not necessary, but it would be better if written.", lines=7, max_lines=7)
                        
                    # Settings
                    with gr.Accordion("Settings"):

                        model_options = [
                                ("gpt-3.5-turbo","gpt-3.5-turbo"), 
                                ("gpt-4","gpt-4"), 
                            ]
                        models = gr.Radio(label="Choose a AI model", info="Choosing GPT-4 is better, but it's also more expensive.", choices=model_options, value="gpt-3.5-turbo")
                        models.change(fn=self.change_model, inputs=models, outputs=None)
                        chunk_size = gr.Slider(label="Chunk size", info="LLM parameter settings, please avoid changing it if you don't understand how it work.\n(default:4000)", value=4000, minimum=500, maximum=4000, step=100)
                        temperature = gr.Slider(label="Temperature", info="LLM parameter settings, please avoid changing it if you don't understand how it work.\n(default:0)", value=0, minimum=0, maximum=2, step=0.1)
                        inputs = [chunk_size, temperature]
                        setting_btn = gr.Button("Save settings", variant="stop")
                        setting_btn.click(fn=self.set_model_configs, 
                                          inputs = inputs, 
                                          show_progress="full")
                        
                # Chat Interface
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=800, bubble_full_width=False, show_copy_button=True)
                    chatbot.unrender()
                    textbox = gr.TextArea(scale=7, placeholder="Type a message...", show_label=False, lines=1, max_lines=6)
                    textbox.unrender()
                    gr.ChatInterface(self.get_response, chatbot=chatbot, textbox=textbox, submit_btn=None, additional_inputs=[prompt])
            
            footer = gr.HTML("""<div style='text-align:center'><p>Author: NTUST MIS 李丞穎</p><a href='https://github.com/eric050828/Retrieval-augmented-generation--Chat-With-Docs'>Retrieval-augmented-generation--Chat-With-Docs</a></div>""")
        
        # Launch
        self.interface.queue()
        self.interface.launch(show_api=False)


    def upload_file(self, file, can_extract_images):
        file_path = file.name
        
        if not file_path:
            gr.Error("You haven't upload any files.")
            return "# File: *Upload a file first...*"
        
        file_name = Path(file_path).name
        file_type = Path(file_path).suffix
        # 取得檔案的父目錄名稱
        self.collection_name = os.path.basename(os.path.dirname(file_path))
        if file_type != ".pdf":
            gr.Info(f"Only support PDF file.\nUnsupported file type: '{file_type}'")
            return "# File: *Upload a file first...*"
        
        self.controller.upload_file(self.collection_name, file_path, can_extract_images)
        gr.Info(f"File:'{file_name}' upload is complete.")
        return f"# File: *{file_name}*"


    def get_response(self, query, history, prompt):
        response = self.controller.get_response(self.collection_name, query, prompt)
        return response
    

    def set_api_key(self, api_key):
        self.controller.set_api_key(api_key)
        gr.Info("API key setup completed.")


    def change_model(self, model_name):
        self.controller.change_model(model_name)
        gr.Info(f"The model is changed to: {model_name}")


    def set_model_configs(self, chunk_size, temperature):
        configs = {
            "chunk_size": chunk_size,
            "temperature": temperature
        }
        self.controller.set_model_configs(configs)
        gr.Info("The changes are saved.")