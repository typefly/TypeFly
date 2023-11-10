import gradio as gr

import random

def random_response(message, history):
    history.append(("Bot", None))
    return random.choice(["Yes", "No"])

demo = gr.ChatInterface(random_response)
demo.queue().launch()

# if __name__ == "__main__":
#     demo.launch()


# with gr.Blocks() as demo:
#     gr.Chatbot([
#         ("Show me an image and an audio file", "Here is an image"), 
#         (None, ("./images/kitchen.webp",)), 
#         (None, ("./images/kitchen.webp",)), 
#         (None, "And here is an audio file:"),
#         (None, "sdfdsf:"),
#     ])

# demo.launch()