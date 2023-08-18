from fastapi import FastAPI
import json
import gradio as gr
import requests

app = FastAPI()

api_url = "http://10.2.16.175:8080/completion"
headers = {
    "Content-Type": "application/json",
}

system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, proper and polite answers to the human's questions.\n### Human: Hello, Assistant.\n### Assistant: Hello. How may I help you today?\n### Human: "

title = "LLAMA Chatbot"
description = """
"""

css = """.toast-wrap { display: none !important } """
examples = [
    "Hello there!",
    "What is 54*28?",
    "Write me a poem.",
]


def predict(message, chatbot):
    input_prompt = system_message

    for interaction in chatbot:
        input_prompt = (
            input_prompt
            + str(interaction[0])
            + "\n### Assistant: "
            + str(interaction[1])
            + "\n### Human: "
        )

    input_prompt = input_prompt + str(message) + "\n### Assistant: "
    input_prompt = input_prompt.strip()
    input_prompt = input_prompt.rstrip()

    data = {
        "prompt": input_prompt,
        "n_predict": 256,
        "n_keep": -1,
        "stop": ["\n### Human:", "Human:"],
        "temperature": 0.8,
        "stream": True,
    }
    print(data)

    response = requests.post(
        api_url, headers=headers, data=json.dumps(data), stream=True
    )

    partial_message = ""
    for line in response.iter_lines():
        if line:  # filter out keep-alive new lines
            # Decode from bytes to string
            decoded_line = line.decode("utf-8")

            # Remove 'data:' prefix
            if decoded_line.startswith("data:"):
                json_line = decoded_line[5:]  # Exclude the first 5 characters ('data:')
            else:
                gr.Warning(f"This line does not start with 'data:': {decoded_line}")
                continue

            # Load as JSON
            try:
                json_obj = json.loads(json_line)
                if "content" in json_obj:
                    partial_message = partial_message + json_obj["content"]
                    yield partial_message
                elif "error" in json_obj:
                    yield json_obj[
                        "error"
                    ] + ". Please refresh and try again with an appropriate smaller input prompt."
                else:
                    gr.Warning(f"API error")

            except json.JSONDecodeError:
                gr.Warning(f"This line is not valid JSON: {json_line}")
                continue
            except KeyError as e:
                gr.Warning(f"KeyError: {e} occurred for JSON object: {json_obj}")
                continue


io = gr.ChatInterface(
    predict,
    title=title,
    description=description,
    css=css,
    examples=examples,
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
).queue(concurrency_count=75)
app = gr.mount_gradio_app(app, io, path="/")
