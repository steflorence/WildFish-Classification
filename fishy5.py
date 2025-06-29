import os
import json
import csv
import uuid
import torch
import torch.nn as nn
from datetime import datetime
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms
from PIL import Image
import gradio as gr

# === Paths ===
model_path = "convnext_fish_classifier.pth"
class_map_path = "class_names.json"
save_dir = "Saved_HighConfidence"
history_csv = "prediction_history.csv"

# === Check for dataset/class mapping ===
if not os.path.exists(class_map_path):
    raise FileNotFoundError("❌ Dataset not found. Make sure you've trained the model and class_names.json exists.")

# === Load class names ===
with open(class_map_path, "r") as f:
    class_names = json.load(f)

# === Load model ===
weights = ConvNeXt_Tiny_Weights.DEFAULT
transform = weights.transforms()

model = convnext_tiny(weights=weights)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))

state_dict = torch.load(model_path, map_location="cpu")
try:
    model.load_state_dict(state_dict)
except RuntimeError:
    print("⚠️ State dict mismatch. Retrying with adjusted classifier layer...")
    for key in list(state_dict.keys()):
        if "classifier.2" in key:
            del state_dict[key]
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
    model.load_state_dict(state_dict, strict=False)

model.eval()
os.makedirs(save_dir, exist_ok=True)

# === Predict + Save logic ===
def classify_fish(image):
    if image is None:
        return "❌ No image uploaded.", None, None

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_idxs = torch.topk(probs, 3)
    top_pred_idx = top_idxs[0].item()
    top_pred_name = class_names[top_pred_idx]
    top_conf = top_probs[0].item() * 100

    # Reject low-confidence
    if top_conf < 60:
        return "❌ Sorry, this doesn’t look like a fish I recognize. Try another image.", image, top_conf

    # Save high-confidence image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    img_filename = f"{top_pred_name}_{timestamp}_{unique_id}.jpg"
    img_path = os.path.join(save_dir, img_filename)

    if top_conf >= 90:
        image.save(img_path)
        print(f"✅ Saved high-confidence image: {img_path}")

    # Save to history
    with open(history_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, top_pred_name, f"{top_conf:.2f}", img_filename if top_conf >= 90 else ""])

    # Format top 3 predictions
    result = "🎯 Top Predictions:\n"
    for i in range(3):
        name = class_names[top_idxs[i].item()]
        prob = top_probs[i].item() * 100
        result += f"{name}: {prob:.2f}%\n"

    return result.strip(), image, top_conf

# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft(), css="""
body {
    background: url('https://i.imgur.com/avBvS43.png');
    background-size: cover;
}
.gr-button {
    background-color: #3b82f6;
    color: white;
    border-radius: 12px;
}
h1 {
    font-size: 3em;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 2px 2px #3b82f6;
    text-align: center;
}
""") as demo:

    gr.Markdown("""<h1>🐟 What The Fish?!</h1>
<p style='color:white; font-size:1.2em; text-align:center;'>Upload a fish image and we'll predict its species. High-confidence results get saved!</p>""")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload a Fish Image")
        result_output = gr.Textbox(label="Top Predictions")

    with gr.Row():
        classify_btn = gr.Button("Classify Fish")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        image_display = gr.Image(visible=False, label="Preview")
        confidence_display = gr.Number(visible=False, label="Confidence (%)")

    classify_btn.click(fn=classify_fish, inputs=image_input,
                       outputs=[result_output, image_display, confidence_display])
    
    clear_btn.click(lambda: None, None, image_input)

demo.launch(share=True)
