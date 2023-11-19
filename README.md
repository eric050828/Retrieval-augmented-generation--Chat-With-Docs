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
