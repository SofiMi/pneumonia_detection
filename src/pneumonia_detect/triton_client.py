import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from transformers import ViTImageProcessor
import logging

def test_triton():
    MODEL_NAME = "vit-pneumonia"
    URL = "localhost:8000"

    try:
        image_path = "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
        image = Image.open(image_path).convert("RGB")
    except:
        print("Картинка не найдена, создаем случайную...")
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)

    try:
        client = httpclient.InferenceServerClient(url=URL)
        if not client.is_server_live():
            print("Сервер Triton не отвечает")
            return
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return

    inputs = httpclient.InferInput("input", pixel_values.shape, "FP32")
    inputs.set_data_from_numpy(pixel_values)

    outputs = httpclient.InferRequestedOutput("output")

    print(f"🚀 Отправка запроса на {URL}...")
    results = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])

    logits = results.as_numpy("output")[0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred_idx = np.argmax(probs)

    print("---------------- TRITON RESULT ----------------")
    print(f"Logits: {logits}")
    print(f"Probabilities: {probs}")
    print(f"Class ID: {pred_idx}")
    print("-----------------------------------------------")

if __name__ == "__main__":
    test_triton()
