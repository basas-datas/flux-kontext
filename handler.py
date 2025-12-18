import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import io
import time

import numpy as np
from PIL import Image, ImageOps

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# IMAGE HELPERS (ИЗ ВТОРОГО ПРОЕКТА)
# ------------------------------------------------------------------

MAX_SIDE = 1568

def base64_to_pil(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    raw = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    img.load()
    return img

def remove_alpha_force_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("P", "LA"):
        img = img.convert("RGBA")

    if img.mode == "RGBA":
        arr = np.array(img, dtype=np.uint8)
        rgb = arr[..., :3].astype(np.float32)
        a = arr[..., 3].astype(np.float32) / 255.0

        h, w = a.shape
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True

        mask = border & (a > 0.1)
        if mask.any():
            bg = np.median(rgb[mask], axis=0)
        else:
            bg = np.array([255, 255, 255], dtype=np.float32)

        comp = rgb * a[..., None] + bg * (1.0 - a[..., None])
        out = np.clip(comp, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    return img.convert("RGB")

def resize_max_side(img: Image.Image, max_side=MAX_SIDE) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        return img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS
        )
    return img

def save_pil_for_comfy(img: Image.Image, task_id: str) -> str:
    os.makedirs(task_id, exist_ok=True)
    path = os.path.join(task_id, "input.png")
    img.save(path, "PNG")
    return os.path.abspath(path)

# ------------------------------------------------------------------
# COMFYUI HELPERS
# ------------------------------------------------------------------

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    req = urllib.request.Request(
        url,
        data=json.dumps(
            {"prompt": prompt, "client_id": client_id}
        ).encode("utf-8")
    )
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    return urllib.request.urlopen(f"{url}?{params}").read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    return json.loads(urllib.request.urlopen(url).read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]

    while True:
        msg = ws.recv()
        if isinstance(msg, str):
            msg = json.loads(msg)
            if msg["type"] == "executing":
                if msg["data"]["node"] is None:
                    break

    history = get_history(prompt_id)[prompt_id]
    images = {}

    for node_id, output in history["outputs"].items():
        if "images" in output:
            images[node_id] = []
            for img in output["images"]:
                data = get_image(
                    img["filename"],
                    img["subfolder"],
                    img["type"]
                )
                images[node_id].append(
                    base64.b64encode(data).decode("utf-8")
                )
    return images

def load_workflow(path):
    with open(path) as f:
        return json.load(f)

# ------------------------------------------------------------------
# HANDLER
# ------------------------------------------------------------------

def handler(job):
    job_input = job.get("input", {})
    task_id = f"task_{uuid.uuid4()}"

    if "image" not in job_input:
        return {"error": "image (base64) required"}

    # ---- IMAGE PIPELINE (КАК ВО ВТОРОМ) ----
    img = base64_to_pil(job_input["image"])
    img = remove_alpha_force_rgb(img)
    img = resize_max_side(img)

    image_path = save_pil_for_comfy(img, task_id)

    # ---- LOAD WORKFLOW ----
    prompt = load_workflow("/flux_kontext_example.json")

    # LoadImage
    prompt["41"]["inputs"]["image"] = image_path

    # Text / seed / guidance
    prompt["6"]["inputs"]["text"] = job_input.get("prompt", "")
    prompt["25"]["inputs"]["noise_seed"] = job_input.get("seed", 42)
    prompt["26"]["inputs"]["guidance"] = job_input.get("guidance", 1.2)

    # ❌ НИГДЕ НЕ СТАВИМ WIDTH / HEIGHT

    # ---- CONNECT WS ----
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}:8188/ws?clientId={client_id}")

    images = get_images(ws, prompt)
    ws.close()

    for node_id in images:
        if images[node_id]:
            return {
                "image": images[node_id][0],
                "width": img.width,
                "height": img.height,
            }

    return {"error": "no image generated"}

# ------------------------------------------------------------------

runpod.serverless.start({"handler": handler})
