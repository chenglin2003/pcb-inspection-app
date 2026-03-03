import os
import json
import urllib.parse
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from ai_client import AIClient

load_dotenv()

# --- Clients ---
def get_ai_client():
    return AIClient()

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

def _init_gdrive():
    service_account_json = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
    gauth = GoogleAuth()

    # Cloud-friendly mode: use service account JSON provided through env var.
    if service_account_json:
        credentials_dict = json.loads(service_account_json)
        gauth.credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        return GoogleDrive(gauth)

    # Local fallback mode: interactive OAuth flow.
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def upload_to_gdrive(file_path: str) -> str:
    global _drive_instance
    if _drive_instance is None:
        _drive_instance = _init_gdrive()
    
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
    client = get_ai_client()
    prompt = (
    "You are Vision Pro 4, an expert PCB quality inspector specializing in component-level defect analysis.\n\n"
    "You will analyze the provided PCB image against a list of automatically flagged candidates. "
    "Your role is the final expert verification layer before disposition.\n\n"
    "## Detected Candidates\n"
    f"{json.dumps(detections, indent=2)}\n\n"
    "## Defect Classification Guide\n"
    "Classify each candidate into ONLY one of these 3 defect types, or reject as false positive:\n\n"
    "| Defect Type        | Description                                              | Key Visual Cue                          |\n"
    "|--------------------|----------------------------------------------------------|-----------------------------------------|\n"
    "| Component Removal  | Component missing from its footprint entirely            | Empty pads, solder residue, bare PCB    |\n"
    "| Positional Swap    | Component placed at wrong location (swapped with another)| Marking mismatch for that footprint     |\n"
    "| Class Substitution | Wrong component type/value placed at correct location    | Incorrect part marking, color, package  |\n\n"
    "## For Each Candidate, Provide:\n"
    "1. **Defect Type** — from the 3 types above, or 'False Positive'\n"
    "2. **Confidence** — High / Medium / Low (flag if image quality limits assessment)\n"
    "3. **Severity** — Critical / Major / Minor\n"
    "4. **Visual Evidence** — specific observation from the image supporting your call\n"
    "5. **Risk** — functional consequence if unaddressed\n"
    "6. **Action** — Pass / Rework / Scrap / Manual Review\n\n"
    "## Board Summary\n"
    "After all candidates:\n"
    "- Overall verdict: PASS / FAIL / MANUAL REVIEW REQUIRED\n"
    "- Most critical defect found (if any)\n"
    "- Suspected root cause pattern (e.g., pick-and-place error, BOM mismatch, handling damage)\n"
)
    return client.analyze_images(prompt=prompt, image_urls=[image_url])


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

    # Draw difference boxes on the golden reference image for clearer operator review.
    boxed = golden_img.copy()
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
    client = get_ai_client()
    detections = detections or []
    prompt = (
        "You are Vision Pro 4, an expert PCB quality inspector.\n"
        "Compare Image A (test PCB) against Image B (golden reference PCB).\n"
        "Focus on manufacturing defects and meaningful differences only.\n"
        "Ignore tiny alignment, lighting, or compression changes.\n\n"
        "Response behavior rules:\n"
        "- Do not include capability disclaimers or refusal lines.\n"
        "- Do not say you are unable to compare images.\n"
        "- If visual certainty is limited, still return the best defect analysis directly from the provided candidates and images.\n"
        "- Start immediately with the requested output sections.\n\n"
        "Coordinate and boxing rules:\n"
        "- Always include defect coordinates in pixels for each detected defect as x,y.\n"
        "- Use the precomputed red-box candidate differences as the primary coordinate source.\n"
        "- Treat those coordinates as red bounding boxes on the golden reference image (Image B).\n"
        "- If no defect is found, explicitly return: Defect coordinates: none.\n\n"
        f"Precomputed red-box candidate differences (x,y,width,height,area): {json.dumps(diff_boxes)}\n"
        f"Roboflow detections on test image: {json.dumps(detections)}\n\n"
        "Return:\n"
        "1) Overall result (PASS/FAIL)\n"
        "2) Top defect differences with short risk explanation and explicit coordinates in the form (x: <int>, y: <int>)\n"
        "3) Suggested operator action"
    )
    image_prompt = f"{prompt}\n\nImage A: Test PCB\nImage B: Golden Reference PCB"
    response_text = client.analyze_images(
        prompt=image_prompt,
        image_urls=[test_image_url, golden_image_url],
    )
    # Remove common capability-disclaimer boilerplate if a model still emits it.
    filtered_lines = []
    for line in response_text.splitlines():
        normalized = line.strip().lower()
        if (
            "unable to directly compare the images" in normalized
            or ("unable to compare" in normalized and "image" in normalized)
            or ("can analyze the given data" in normalized)
        ):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()

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
