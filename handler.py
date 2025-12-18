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
import time
import random
import io

from PIL import Image, ImageOps
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# CUDA 검사 및 설정
# -----------------------------
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# CUDA 검사 실행
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)

# -----------------------------
# ComfyUI API helpers
# -----------------------------
server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
        else:
            # binary previews etc
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        images_output = []
        if "images" in node_output:
            for image in node_output["images"]:
                image_data = get_image(image["filename"], image["subfolder"], image["type"])
                if isinstance(image_data, bytes):
                    image_data = base64.b64encode(image_data).decode("utf-8")
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def load_workflow(workflow_path):
    with open(workflow_path, "r") as file:
        return json.load(file)

# -----------------------------
# Image helpers (как во 2-м)
# -----------------------------
MAX_SEED = 2**31 - 1

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
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        mask = border & (a > 0.1)
        if mask.any():
            bg = np.median(rgb[mask], axis=0)
        else:
            bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        comp = rgb * a[..., None] + bg[None, None, :] * (1.0 - a[..., None])
        out = np.clip(comp, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    return img.convert("RGB")

def resize_max_side(img: Image.Image, max_side: int = 1568) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)  # no upscale
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img

def load_any_image(image_input: str) -> Image.Image:
    # path?
    if isinstance(image_input, str) and os.path.exists(image_input):
        img = Image.open(image_input)
        img = ImageOps.exif_transpose(img)
        img.load()
        return img
    # base64
    return base64_to_pil(image_input)

def save_pil_to_file(img: Image.Image, temp_dir: str, filename: str = "input.png") -> str:
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(temp_dir, filename))
    img.save(path, format="PNG", optimize=True)
    return path

# -----------------------------
# Handler
# -----------------------------
def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input keys: {list(job_input.keys())}")

    # --- image from client (как во втором) ---
    image_input = job_input.get("image") or job_input.get("image_path")
    if not image_input:
        return {"error": "image (base64) is required"}

    prompt_text = job_input.get(
        "prompt",
        "[photo content], remove overlays and reconstruct background naturally"
    )
    guidance_scale = float(job_input.get("guidance_scale", job_input.get("guidance", 2.5)))
    steps = int(job_input.get("steps", 28))

    seed = int(job_input.get("seed", 42))
    if job_input.get("randomize_seed"):
        seed = random.randint(0, MAX_SEED)

    max_side = int(job_input.get("max_side", 1568))

    task_id = f"task_{uuid.uuid4()}"
    try:
        # special example passthrough
        if image_input == "/example_image.png":
            img = Image.open("/example_image.png")
            img = ImageOps.exif_transpose(img)
            img.load()
        else:
            img = load_any_image(image_input)

        img = remove_alpha_force_rgb(img)
        img = resize_max_side(img, max_side=max_side)

        image_path = save_pil_to_file(img, temp_dir=task_id, filename="input.png")
        width, height = img.size

    except (binascii.Error, ValueError) as e:
        return {"error": f"Invalid image/base64: {e}"}
    except Exception as e:
        return {"error": f"Failed to load/process image: {e}"}

    # --- Load and patch workflow ---
    prompt = load_workflow("/flux_kontext_example.json")

    # LoadImage
    prompt["41"]["inputs"]["image"] = image_path
    # Prompt
    prompt["6"]["inputs"]["text"] = prompt_text
    # Seed
    prompt["25"]["inputs"]["noise_seed"] = seed
    # Guidance
    prompt["26"]["inputs"]["guidance"] = guidance_scale
    # Steps
    prompt["17"]["inputs"]["steps"] = steps

    # IMPORTANT: no fixed output from client — use processed image size
    prompt["27"]["inputs"]["width"] = int(width)
    prompt["27"]["inputs"]["height"] = int(height)
    prompt["30"]["inputs"]["width"] = int(width)
    prompt["30"]["inputs"]["height"] = int(height)

    # --- Connect ---
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    # check HTTP
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")

    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt + 1})")
            break
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt + 1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                return {"error": "ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."}
            time.sleep(1)

    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt + 1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt + 1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                return {"error": "웹소켓 연결 시간 초과 (3분)"}
            time.sleep(5)

    images = get_images(ws, prompt)
    ws.close()

    if not images:
        return {"error": "이미지를 생성할 수 없습니다."}

    for node_id in images:
        if images[node_id]:
            return {
                "image": images[node_id][0],
                "seed": seed,
                "width": int(width),
                "height": int(height),
            }

    return {"error": "이미지를 찾을 수 없습니다."}

runpod.serverless.start({"handler": handler})
