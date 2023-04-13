import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from app.le_util.draw import show_mask, show_box, show_anns, base64_to_cv2, GPUINFO
from demo.inference_on_a_image import get_grounding_output
from lama_inpaint import inpaint_img_with_lama
from sam_segment import predict_masks_with_sam
from segment_anything import SamPredictor, build_sam, sam_model_registry, SamAutomaticMaskGenerator

import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image
from fastapi import FastAPI
from matplotlib import pyplot as plt


from GroundingDINO.groundingdino.util.inference import load_model
from stable_diffusion_inpaint import inpaint_img_with_sd
from utils import save_array_to_img, load_img_to_array, dilate_mask

root = "/home/aimall/lujiale/Inpaint-Anything/"
config_file = root + 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
device = 'cuda'
grounded_checkpoint = root + 'model/groundingdino_swint_ogc.pth'
sam_checkpoint = root + 'model/sam_vit_h_4b8939.pth'
box_threshold = 0.3
text_threshold = 0.3
app = FastAPI()
#load model
model = load_model(config_file, grounded_checkpoint, device=device)
predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
print("load model finish")
@app.post("/seg_image")
async def seg_image(input_image: str = ""):
    if (len(input_image) == 0):
        from app.test_image import my_image
        input_image = my_image.image
        # 解码 base64 编码的图像字符串
    image=base64_to_cv2(input_image)
    image = cv2.resize(image, (640, 480))
    height, width = image.shape[:2]

    # 打印图像的宽度和高度
    print(f"Image width: {width}")
    print(f"Image height: {height}")
    GPUINFO()
    start_time = time.time()
    masks = mask_generator.generate(image)
    end_time = time.time()
    print(f"Time elapsed: {(end_time - start_time) * 1000:.3f} milliseconds")
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    # 将Matplotlib的图像对象转换为Pillow的图像对象
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buffer.seek(0)
    image = Image.open(buffer)

    # 将Pillow的图像对象编码为Base64格式
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded_string = base64.b64encode(buffer.getvalue())

    return encoded_string
@app.post("/detect_image")
async def detect_image(input_image: str = "",
                         det_prompt: str = ""):
    if (len(input_image) == 0):
        from app.test_image import my_image
        input_image = my_image.image

    # 解码 base64 编码的图像字符串
    decoded_image = base64.b64decode(input_image)
    # 将解码后的字节流转换为 PIL.Image 对象
    image_pil = Image.open(io.BytesIO(decoded_image)).convert("RGB")

    # image_pil=image_pil.resize((512, 512))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold
    )

    image = np.array(image_pil)

    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
    if min(transformed_boxes.shape) == 0:
        print('此处张量为空')
        return base64.b64encode(decoded_image).decode('utf-8')
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    maskslist = masks.squeeze(1).cpu().numpy()
    dilate_kernel_size = 15
    if dilate_kernel_size is not None:
        maskslist = [dilate_mask(mask, dilate_kernel_size) for mask in maskslist]
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in maskslist:
        show_mask(mask, plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    # 将Matplotlib的图像对象转换为Pillow的图像对象
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buffer.seek(0)
    image = Image.open(buffer)

    # 将Pillow的图像对象编码为Base64格式
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded_string = base64.b64encode(buffer.getvalue())


    return encoded_string
@app.post("/repaceobj_image")
async def repaceobj_image(input_image: str = "",det_prompt:str="",
                         text_prompt: str = ""):
    if (len(input_image) == 0):
        from app.test_image import my_image
        input_image = my_image.image

    # 解码 base64 编码的图像字符串
    decoded_image = base64.b64decode(input_image)
    # 将解码后的字节流转换为 PIL.Image 对象
    image_pil = Image.open(io.BytesIO(decoded_image)).convert("RGB")

    # image_pil=image_pil.resize((512, 512))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold
    )

    image = np.array(image_pil)

    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
    if min(transformed_boxes.shape) == 0:
        print('此处张量为空')
        return base64.b64encode(decoded_image).decode('utf-8')
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    maskslist = masks.squeeze(1).cpu().numpy()
    dilate_kernel_size = 15
    if dilate_kernel_size is not None:
        maskslist = [dilate_mask(mask, dilate_kernel_size) for mask in maskslist]
    merged_mask = np.zeros_like(maskslist[0])
    for mask in maskslist:
        merged_mask = np.logical_or(merged_mask, mask)
    merged_mask = merged_mask * 255
    merged_mask = merged_mask.astype(np.uint8)
    img_inpainted_p = f"/home/aimall/lujiale/Inpaint-Anything/app/static/myrepaceinpainted.png"
    img_inpainted = inpaint_img_with_sd(
        image, merged_mask, text_prompt)
    img_inpaintedshow = Image.fromarray(img_inpainted)

    with io.BytesIO() as buffer:
        img_inpaintedshow.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

    # 将字节流编码为 base64 字符串
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    save_array_to_img(img_inpainted, img_inpainted_p)
    return base64_str
@app.post("/removeobj_image")
async def removeobj_image(input_image: str = "",
                         text_prompt: str = ""):
    if (len(input_image) == 0):
        from app.test_image import my_image
        input_image = my_image.image

    # 解码 base64 编码的图像字符串
    decoded_image = base64.b64decode(input_image)
    # 将解码后的字节流转换为 PIL.Image 对象
    image_pil = Image.open(io.BytesIO(decoded_image)).convert("RGB")

    # image_pil=image_pil.resize((512, 512))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold
    )
    image = np.array(image_pil)

    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
    if min(transformed_boxes.shape) == 0:
        print('此处张量为空')
        return  base64.b64encode(decoded_image).decode('utf-8')
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    maskslist = masks.squeeze(1).cpu().numpy()
    dilate_kernel_size=15
    if dilate_kernel_size is not None:
        maskslist = [dilate_mask(mask, dilate_kernel_size) for mask in maskslist]
    merged_mask = np.zeros_like(maskslist[0])
    for mask in maskslist:
        merged_mask = np.logical_or(merged_mask, mask)
    merged_mask = merged_mask * 255
    merged_mask = merged_mask.astype(np.uint8)
    img_inpainted_p = f"/home/aimall/lujiale/Inpaint-Anything/app/static/myinpainted.png"
    img_inpainted = inpaint_img_with_lama(
        image, merged_mask, "/home/aimall/lujiale/Inpaint-Anything/lama/configs/prediction/default.yaml", "/home/aimall/lujiale/Inpaint-Anything/big-lama")
    img_inpaintedshow = Image.fromarray(img_inpainted)

    with io.BytesIO() as buffer:
        img_inpaintedshow.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

    # 将字节流编码为 base64 字符串
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    save_array_to_img(img_inpainted, img_inpainted_p)
    return base64_str

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)