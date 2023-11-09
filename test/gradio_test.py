import time
import gradio as gr

global stop
stop = False
def slow_echo(message, history):
    global stop
    if message == "exit":
        # demo.close()
        stop = True
    for i in range(len(message)):
        time.sleep(0.05)
        yield "You typed: " + message[: i+1]

demo = gr.ChatInterface(slow_echo).queue()

if __name__ == "__main__":
    demo.launch(prevent_thread_lock=True)
    while True:
        time.sleep(1)
        print(f"Running {stop}")
        if stop:
            break
    print("Done")
