import base64
import io

import gradio as gr
import cv2
from PIL import Image

from prompt2anything import removeobj_image, repaceobj_image, detect_image, seg_image


async def process_image_pil(input_image, det_prompt,text_prompt,task_type):
    # 将 PIL 对象转换为字节流
    with io.BytesIO() as buffer:
        input_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

    # 将字节流编码为 base64 字符串
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    if(task_type=="remove"):
    # 调用另一个函数并传入 base64 字符串和 prompt 文字
        processed_base64_str = await removeobj_image(base64_str, text_prompt)
    elif(task_type=="replace"):
        processed_base64_str = await repaceobj_image(base64_str,det_prompt, text_prompt)
    elif (task_type == "detect"):
        processed_base64_str = await detect_image(base64_str, det_prompt)
    elif(task_type=="seg"):
        processed_base64_str = await seg_image(base64_str)
    else:
        return
    # 将返回的 base64 字符串解码为字节流
    processed_bytes = base64.b64decode(processed_base64_str)

    # 将字节流转换为 PIL 对象
    processed_image = Image.open(io.BytesIO(processed_bytes))

    # 返回 PIL 对象
    return processed_image


if __name__ == "__main__":
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil")
                text_prompt = gr.Textbox(label="text_prompt")
                det_prompt = gr.Textbox(label="det_prompt")
                task_type = gr.Textbox(label="task type: replace/remove/detect/seg")
                run_button = gr.Button(label="Run")

            with gr.Column():
                gallery = gr.outputs.Image(
                    type="pil",
                ).style(full_width=True, full_height=True)

        run_button.click(fn=process_image_pil, inputs=[
            input_image, det_prompt,text_prompt,task_type], outputs=[gallery])

    block.launch(server_name='0.0.0.0', server_port=8001)