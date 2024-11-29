from transformers import pipeline
import gradio as gr

model =pipeline("summaziration")

def predict(promt):
    summary = model(promt)[0]["summary_text"]
    return summary

with gr.Block as demo:
    textbox = gr.Textbox(placeholder="Enter text block", lines = 4)
    gr.Interface(fn = predict, inputs = textbox, outputs = "text")

demo.launch()