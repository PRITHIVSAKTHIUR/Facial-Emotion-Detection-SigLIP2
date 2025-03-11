![fsedfs.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/IDBZcJQvQ2UvmczGMYS-W.png)

# **Facial-Emotion-Detection-SigLIP2**

> **Facial-Emotion-Detection-SigLIP2** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify different facial emotions using the **SiglipForImageClassification** architecture.  


```py
Classification Report:
              precision    recall  f1-score   support

      Ahegao     0.9916    0.9801    0.9858      1205
       Angry     0.8633    0.7502    0.8028      1313
       Happy     0.9494    0.9684    0.9588      3740
     Neutral     0.7635    0.8781    0.8168      4027
         Sad     0.8595    0.7794    0.8175      3934
    Surprise     0.9025    0.8104    0.8540      1234

    accuracy                         0.8665     15453
   macro avg     0.8883    0.8611    0.8726     15453
weighted avg     0.8703    0.8665    0.8663     15453
```
![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/j29921aYUCg9a5ZqXQ8P2.png)

    The model categorizes images into 6 facial emotion classes:
    
        Class 0: "Ahegao"
        Class 1: "Angry"
        Class 2: "Happy"
        Class 3: "Neutral"
        Class 4: "Sad"
        Class 5: "Surprise"

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def emotion_classification(image):
    """Predicts facial emotion classification for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Ahegao", "1": "Angry", "2": "Happy", "3": "Neutral",
        "4": "Sad", "5": "Surprise"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=emotion_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Facial Emotion Detection",
    description="Upload an image to classify the facial emotion."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**  

The **Facial-Emotion-Detection-SigLIP2** model is designed to classify different facial emotions based on images. Potential use cases include:  

- **Mental Health Monitoring:** Detecting emotional states for well-being analysis.
- **Human-Computer Interaction:** Enhancing user experience by recognizing emotions.
- **Security & Surveillance:** Identifying suspicious or aggressive behaviors.
- **AI-Powered Assistants:** Supporting AI-based emotion recognition for various applications.
