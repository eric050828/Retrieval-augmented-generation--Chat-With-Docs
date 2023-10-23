# My ChatPDF

## Installtion


```shell
$ cd RAG-sample
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