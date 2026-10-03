import gradio as gr
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import io

# Load model + class names
model = tf.keras.models.load_model("final_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

CATEGORY_EMOJI = {
    "plastic": "🥤", "glass": "🍶", "paper": "📄",
    "cardboard": "📦", "metal": "🥫", "aluminum": "🥫",
    "steel": "🥫", "aerosol": "🧴", "food": "🍎",
    "coffee": "☕", "tea": "🍵", "clothing": "👕",
    "shoes": "👟", "styrofoam": "🥡",
}

def get_emoji(class_name):
    for key, emoji in CATEGORY_EMOJI.items():
        if key in class_name.lower():
            return emoji
    return "♻️"

def predict(image):
    if image is None:
        return {"Upload an image": 1.0}
    img = image.resize((224, 224))
    arr = np.expand_dims(img_to_array(img), axis=0).astype(np.float32)
    preds = model.predict(arr, verbose=0)[0]
    top5 = np.argsort(preds)[-5:][::-1]
    return {
        get_emoji(class_names[i]) + " " + class_names[i].replace("_", " "): float(preds[i])
        for i in top5
    }

with gr.Blocks(title="Waste Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ♻️ Waste Classification CNN\n### EfficientNetB0 · 87.2% accuracy · 30 classes")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Waste Image")
            submit_btn = gr.Button("Classify", variant="primary", size="lg")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Top-5 Predictions")
    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)
    image_input.change(fn=predict, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()
