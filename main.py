import argparse
import glob
import os

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer

from text_image_merge import ImageManager
from models.gpt_neox import GPTNeoXForCausalLM

assert torch.cuda.is_available(), "CUDA is not available"

parser = argparse.ArgumentParser()

parser.add_argument("--text-model", type=str, default="EleutherAI/gpt-neox-1.3B")
parser.add_argument("--image-model", type=str, default="Bingsu/my-korean-stable-diffusion-v1-5")
parser.add_argument("--lora-path", type=str, default="lora.pth")
parser.add_argument("--fonts-path", type=str, default="./font/naver_handwriting")

args = parser.parse_args()

device = torch.device("cuda")

# load text model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.text_model)
model = GPTNeoXForCausalLM.from_pretrained(args.text_model)
model.load_state_dict(torch.load(args.lora_path), strict=False)
model.to(device)
model.eval()

# load image model
diffusion = StableDiffusionPipeline.from_pretrained("Bingsu/my-korean-stable-diffusion-v1-5", torch_dtype=torch.float32)
diffusion.to(device)

prompt = """키워드를 포함한 짧은 시를 생성하여라
###
키워드: 창문, 소통
시: 창문의 틈새로 들어오는
바람과 함께 
너의 목소리가 들려온다.
###
키워드: 향기, 계절
시: 봄의 향기는 너무 짙어
여름까지 지속된다.
###
키워드: 창문, 파랑
시: 창문의 색과 하늘의 색이 닮아있다.
그래서 우리는 서로에게 스며들 수 밖에 없다.
###
키워드: """

fonts = {os.path.abspath(font_path): "None" for font_path in glob.glob(f"{args.fonts_path}/*.ttf")}


def generate_text(text):
    text = prompt + text + "\n시:"
    with torch.no_grad():
        tokens = tokenizer.encode(text, return_tensors='pt').to(device='cuda')
        gen_tokens = model.generate(
            tokens,
            do_sample=True,
            max_new_tokens=48,
        )
        generateds = tokenizer.batch_decode(gen_tokens)

    return generateds[0][len(text):].split("###")[0]


def generate(text):
    text = generate_text(text)
    image = diffusion(f"감성적인, 풍경, 자연, 사실적인, 8K, {text}").image[0]
    ImageMan = ImageManager.ImageManager(
        img=image,
        fonts=fonts,
        text=text,
    )
    ImageMan.resize_image()
    ImageMan.add_text_in_middle(auto_font_color=True)
    return ImageMan.img


demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(placeholder="가을, 코스모스"),
    outputs="image",
)

demo.launch(share=True)
