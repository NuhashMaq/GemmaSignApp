import gradio as gr
import cv2
import numpy as np

def process_frame(frame):
    # frame is a numpy array image from webcam
    # For demo, just convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(source="webcam", type="numpy"),
    outputs=gr.Image(type="numpy")
)

demo.launch()
