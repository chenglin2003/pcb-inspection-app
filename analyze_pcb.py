import os
import json
import urllib.parse
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
from openai import OpenAI
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv

load_dotenv()

# --- Clients ---
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_image(image_path, target_size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.contain(img, target_size)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    upper_left = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
    new_img.paste(img, upper_left)
    new_img.save(image_path)
    return image_path

# --- Google Drive Logic ---
_drive_instance = None
def upload_to_gdrive(file_path: str) -> str:
    global _drive_instance
    if _drive_instance is None:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth() 
        _drive_instance = GoogleDrive(gauth)
    
    file = _drive_instance.CreateFile({'title': os.path.basename(file_path)})
    file.SetContentFile(file_path)
    file.Upload()
    file.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
    return f"https://drive.google.com/uc?export=download&id={file['id']}"

# --- Roboflow URL Inference (Updated with Parameters) ---
def run_roboflow_inference_url(image_url: str, confidence: int, overlap: int):
    api_key = os.getenv("ROBOFLOW_API_KEY")
    model_id = os.getenv("ROBOFLOW_MODEL_ID")
    
    encoded_url = urllib.parse.quote_plus(image_url)
    # Added confidence and overlap parameters to the query string
    endpoint = (
        f"https://detect.roboflow.com/{model_id}?"
        f"api_key={api_key}&image={encoded_url}&"
        f"confidence={confidence}&overlap={overlap}"
    )
    
    response = requests.post(endpoint)
    if response.status_code != 200:
        raise Exception(f"Roboflow API Error: {response.text}")
    return response.json()

# --- Vision Pro 4 (Semantic Analysis) ---
def get_vision_pro_explanation(image_url, detections):
    client = get_openai_client()
    prompt = (
        "You are Vision Pro 4, an expert PCB quality inspector. "
        "Analyze this PCB image based on these detected candidates:\n"
        f"{json.dumps(detections)}\n\n"
        "Identify if they are actual defects (bridging, missing components, etc.) and explain the risk."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]}]
    )
    return response.choices[0].message.content


def compare_images_and_draw_differences(
    test_image_path: str,
    golden_image_path: str,
    output_path: str = "vision_compare_result.png",
    diff_threshold: int = 35,
    min_area: int = 120,
):
    test_img = cv2.imread(test_image_path)
    golden_img = cv2.imread(golden_image_path)
    if test_img is None or golden_img is None:
        raise ValueError("Could not load one or both images for comparison.")

    if test_img.shape[:2] != golden_img.shape[:2]:
        golden_img = cv2.resize(golden_img, (test_img.shape[1], test_img.shape[0]))

    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray_golden = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_test, gray_golden)
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxed = test_img.copy()
    diff_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 3)
        diff_boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "area": float(area)})

    cv2.imwrite(output_path, boxed)
    return output_path, diff_boxes


def get_vision_pro_comparison_explanation(
    test_image_url: str,
    golden_image_url: str,
    diff_boxes,
    detections=None,
):
    client = get_openai_client()
    detections = detections or []
    prompt = (
        "You are Vision Pro 4, an expert PCB quality inspector.\n"
        "Compare Image A (test PCB) against Image B (golden reference PCB).\n"
        "Focus on manufacturing defects and meaningful differences only.\n"
        "Ignore tiny alignment, lighting, or compression changes.\n\n"
        f"Precomputed red-box candidate differences (x,y,width,height,area): {json.dumps(diff_boxes)}\n"
        f"Roboflow detections on test image: {json.dumps(detections)}\n\n"
        "Return:\n"
        "1) Overall result (PASS/FAIL)\n"
        "2) Top defect differences with short risk explanation\n"
        "3) Suggested operator action"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Image A: Test PCB"},
                    {"type": "image_url", "image_url": {"url": test_image_url}},
                    {"type": "text", "text": "Image B: Golden Reference PCB"},
                    {"type": "image_url", "image_url": {"url": golden_image_url}},
                ],
            }
        ],
    )
    return response.choices[0].message.content

# --- Drawing Utility (Updated with labels) ---
def draw_annotations(image_path, predictions):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 18)
    except: font = ImageFont.load_default()

    for pred in predictions:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
        
        # Label with Confidence
        label = f"{pred['class']} ({pred['confidence']:.2f})"
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 20), label, fill="red", font=font)
    
    save_path = "annotated_result.png"
    img.save(save_path)
    return save_path
