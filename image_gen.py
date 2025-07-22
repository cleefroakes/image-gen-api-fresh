from diffusers import StableDiffusionPipeline
import torch
from transformers.utils import move_cache
import importlib.util

pipe = None

def load_model():
    global pipe
    if pipe is None:
        move_cache()  # Moved to delay cache migration
        try:
            if importlib.util.find_spec("accelerate") is None:
                print("Warning: accelerate not found, using default loading")
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    use_safetensors=True
                ).to("cpu")
            else:
                from accelerate import init_empty_weights, load_checkpoint_and_dispatch
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
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return pipe

def generate_image(prompt: str):
    if pipe is None:
        load_model()
    image = pipe(
        prompt,
        num_inference_steps=5,
        height=32,
        width=32,
        guidance_scale=7.5
    ).images[0]
    return image

if __name__ == "__main__":
    image = generate_image("A futuristic city")
    image.save("futuristic_city.png")