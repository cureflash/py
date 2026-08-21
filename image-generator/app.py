import os
import random

import gradio as gr
import torch
from diffusers import QwenImagePipeline

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image")
LOW_VRAM = os.getenv("LOW_VRAM", "1") == "1"


def load_pipeline() -> QwenImagePipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応のNVIDIA GPUが必要です。")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = QwenImagePipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

    if LOW_VRAM:
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
    else:
        pipe.to("cuda")

    return pipe


pipe = load_pipeline()


def generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
):
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("プロンプトを入力してください。")

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or " ",
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        true_cfg_scale=float(cfg_scale),
        generator=generator,
    ).images[0]

    return image, seed


with gr.Blocks(title="Local Image Generator") as demo:
    gr.Markdown("# Local Image Generator\nQwen-Imageを使ったローカル画像生成UI")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                lines=6,
                placeholder="例: a small robot reading a book in a quiet library, anime illustration",
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                lines=3,
                placeholder="blurry, low quality, distorted",
            )

            with gr.Row():
                width = gr.Dropdown([512, 768, 1024], value=1024, label="Width")
                height = gr.Dropdown([512, 768, 1024], value=1024, label="Height")

            steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
            cfg_scale = gr.Slider(1.0, 6.0, value=4.0, step=0.1, label="CFG scale")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Image(label="Generated image", type="pil")
            used_seed = gr.Number(label="Used seed", precision=0)

    generate_button.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, width, height, steps, cfg_scale, seed],
        outputs=[output, used_seed],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
