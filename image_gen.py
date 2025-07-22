from diffusers import StableDiffusionPipeline
import torch
from transformers.utils import move_cache
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Move cache (run once, handle manually if needed)
move_cache()

pipe = None

def load_model():
    global pipe
    if pipe is None:
        with init_empty_weights():
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        pipe = load_checkpoint_and_dispatch(
            pipe,
            device_map="cpu",
            offload_folder="offload",
            offload_state_dict=True
        )
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing()
    return pipe

def generate_image(prompt: str):
    if pipe is None:
        load_model()
    image = pipe(
        prompt,
        num_inference_steps=5,  # Further reduced
        height=32,             # Minimal size
        width=32,
        guidance_scale=7.5
    ).images[0]
    return image

if __name__ == "__main__":
    image = generate_image("A futuristic city")
    image.save("futuristic_city.png")