# Chat with Docs

This repo contains a web app made with Gradio. it integrates an LLM from OpenAI using Langchain for exploring an external document, this approach is called Retrieval augmented generation

## Installtion

```shell
$ cd Retrieval-augmented-generation--Chat-With-Docs
$ mkdir config

$ python -m venv .venv

$ source .venv/bin/activate
(.venv)$ pip install -r ./requirements.txt
```

## Add .env file

Edit config/.env

```
OPENAI_API_KEY=my-open-ai-api-key
```

## Run

```shell
(.venv)$ python src/main.py
```

## How to Use

1. Upload the document to be retrieved (_.pdf, _.txt).
2. Ask your assistant questions.

## Future Improvements

Here are some features and improvements that I plan to implement in future versions:

### Features

- [x] Feature 1: Adding the "Settings" tab.
- [ ] Feature 2: Adding the "Create a new chat" function.
- [ ] Feature 3: Adding the capability to load audio files.
- [ ] Feature 4: Adding the function to export chat history.

### Enhancements

- [ ] Improve UI/UX: Enhancing the chatbot interface.

### Modification

- [x] File Upload: The method for file uploads will change from selection to manual uploading.

### Refactoring

- [ ] Codebase Cleanup: Perform general code cleanup and remove redundant or unused code sections.
- [ ] Modularize Components: Break down large components into smaller, reusable modules for better organization.
