import gradio as gr
from functionality import process_image

def display_results(results):
    original_images = [result[0] for result in results]
    processed_images = [result[1] for result in results]
    recognized_texts = [result[2] for result in results]
    return original_images, processed_images, recognized_texts

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                process_button = gr.Button('Xử lý')
                input_image = gr.Image(label='Tải hình ảnh')
            with gr.Column():
                output_image = gr.Image()
                output_text = gr.Textbox(label="Chữ dự đoán")
            with gr.Column():
                pass
        process_button.click(process_image,inputs=input_image,outputs=[output_image,output_text])


demo.launch(debug=True)
