from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
from fastapi import FastAPI, Request
import uvicorn
import asyncio
import base64
from io import BytesIO
import time
import argparse
import os  

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

bf16 = True
processor = None
model = None
lock = asyncio.Lock()

def _first_device():
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        devs = [d for d in model.hf_device_map.values() if isinstance(d, str) and d.startswith("cuda")]
        if devs:
            return torch.device(sorted(set(devs))[0])
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def _max_memory_map():
    mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            gib = max(int(total / (1024**3)) - 1, 1)  
            mem[f"cuda:{i}"] = f"{gib}GiB"
    return mem or {"cpu": "99GiB"}

def load_model():
    global processor, model, bf16
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        cache_dir='./cache',
        low_cpu_mem_usage=True
    )

    torch_dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",         
        cache_dir='./cache',
        low_cpu_mem_usage=True
    )
    print(f"Loaded with device map: {getattr(model, 'hf_device_map', None)}")

def decode_base64_to_pil_image(b64):
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

@torch.no_grad()
def generate_batch(image_base64_list, text_list):
    global processor, model, bf16
    

    images = [decode_base64_to_pil_image(image_base64) for image_base64 in image_base64_list]
    inputs = processor.process(
        images=images,
        text=text_list,
    )

    d0 = _first_device()
    inputs = {k: (v.to(d0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if bf16 and "images" in inputs and isinstance(inputs["images"], torch.Tensor):
        inputs["images"] = inputs["images"].to(torch.bfloat16)

    use_cuda = d0.type == "cuda"
    with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                        enabled=use_cuda,
                        dtype=torch.bfloat16 if use_cuda else torch.float32):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
    
    generated_text_list = []
    for i in range(len(image_base64_list)):
        generated_tokens = output[i, inputs['input_ids'].size(1):]
        generated_text_list.append(processor.tokenizer.decode(generated_tokens, skip_special_tokens=True))

    print(generated_text)

    return generated_text_list

@torch.no_grad()
def generate(image_base64, text, new_model_name=None):
    global processor, model, bf16
    d0 = _first_device()

    if image_base64 is not None:
        images = [decode_base64_to_pil_image(image) for image in image_base64]
        inputs = processor.process(
            images=images ,
            text=text,
        )
    else:
        inputs = processor.process(
            text=text,
        )

    inputs = {k: (v.to(d0).unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if bf16 and "images" in inputs and isinstance(inputs["images"], torch.Tensor):
        inputs["images"] = inputs["images"].to(torch.bfloat16)

    use_cuda = d0.type == "cuda"
    with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                        enabled=use_cuda,
                        dtype=torch.bfloat16 if use_cuda else torch.float32):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
    
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(generated_text)

    return generated_text



app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    load_model()

@app.post("/generate-batch")
async def handle_generate_batch(request: Request):
    data = await request.json()
    async with lock:
        before = time.time()
        res = generate_batch(data['image_base64'], data['text'])
        print(f"Time taken: {time.time() - before}")
    print(res)
    
    return {"response": res}

@app.post("/generate")
async def handle_generate(request: Request):
    data = await request.json()
    print(f'Processing request: {data}')
    async with lock:
        before = time.time()
        try:
            res = generate(data.get('image_base64', None), data['text'], data.get("model_name", None))
        except Exception as e:
            res = f"Molmo error: {e}"
        print(f"Time taken: {time.time() - before}")
    print(res)
    
    return {"response": res}


def test_model():
    prompt = "Here is some json. {\"car\": \"blue\", \"bike\": \"red\"} What color is the car?"
    res = generate(None,prompt)
    print(res)

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--host", type=str, default="0.0.0.0")
    arg.add_argument("--port", type=int, default=8989)
    args = arg.parse_args()
    host = args.host
    port = args.port
    uvicorn.run(app, host=host, port=port, reload=True, debug=True, log_config="log.ini")



