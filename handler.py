import base64
import binascii  # Base64 에러 처리를 위해 import
import io
import json
import logging
import os
import uuid

import numpy as np
import websocket
from PIL import Image
import urllib.parse
import urllib.request
import runpod


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA 검사 및 설정
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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



server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
def remove_alpha_force_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("P", "LA"):
        img = img.convert("RGBA")

    if img.mode == "RGBA":
        arr = np.array(img, dtype=np.float32)
        rgb = arr[..., :3]
        a = arr[..., 3] / 255.0

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


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img


def normalize_input_image(image_input, task_id: str, max_size: int) -> tuple[str, tuple[int, int]]:
    """Decode the user-provided image, normalize format, resize, and save to PNG."""

    os.makedirs(task_id, exist_ok=True)
    image_bytes = None

    if isinstance(image_input, str):
        cleaned_input = image_input.split(",")[-1]
        try:
            image_bytes = base64.b64decode(cleaned_input, validate=True)
        except (binascii.Error, ValueError):
            with open(image_input, "rb") as f:
                image_bytes = f.read()
    else:
        raise ValueError("Image input must be a base64 string or a valid file path.")

    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb_image = remove_alpha_force_rgb(img)
        target_max_side = min(max_size, max(rgb_image.size))
        processed_image = resize_max_side(rgb_image, target_max_side)

    image_path = os.path.abspath(os.path.join(task_id, "input_image.png"))
    processed_image.save(image_path, format="PNG")
    return image_path, processed_image.size


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
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
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                # bytes 객체를 base64로 인코딩하여 JSON 직렬화 가능하게 변환
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    image_input = job_input["image"]
    max_size = int(job_input["max_size"])

    image_path, image_size = normalize_input_image(image_input, task_id, max_size)

    prompt = load_workflow("/flux_kontext_example.json")

    prompt["41"]["inputs"]["image"] = image_path
    prompt["6"]["inputs"]["text"] = job_input["prompt"]
    prompt["25"]["inputs"]["noise_seed"] = job_input["seed"]
    prompt["26"]["inputs"]["guidance"] = job_input["guidance_scale"]
    prompt["17"]["inputs"]["steps"] = job_input["steps"]

    width, height = image_size
    prompt["27"]["inputs"]["width"] = width
    prompt["27"]["inputs"]["height"] = height
    prompt["30"]["inputs"]["width"] = width
    prompt["30"]["inputs"]["height"] = height

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    # 먼저 HTTP 연결이 가능한지 확인
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    
    # HTTP 연결 확인 (최대 1분)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    # 웹소켓 연결 시도 (최대 3분)
    max_attempts = int(180/5)  # 3분 (1초에 한 번씩 시도)
    for attempt in range(max_attempts):
        import time
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    images = get_images(ws, prompt)
    ws.close()

    # 이미지가 없는 경우 처리
    if not images:
        return {"error": "이미지를 생성할 수 없습니다."}
    
    # 첫 번째 이미지 반환
    for node_id in images:
        if images[node_id]:
            return {"image": images[node_id][0]}
    
    return {"error": "이미지를 찾을 수 없습니다."}

runpod.serverless.start({"handler": handler})