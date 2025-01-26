import torch
from model.mttx_cnn import MTTX_CNN
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the trained model
model = MTTX_CNN()
model.load_state_dict(torch.load('data/models/mttx_cnn.pth'))
model.eval()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def predict_expression(face_image):
    # Transform the image to match model input
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    face_image = Image.fromarray(face_image)
    face_tensor = transform(face_image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(face_tensor)

    _, predicted_class = torch.max(output, 1)
    predicted_emotion = emotion_labels[predicted_class.item()]

    return predicted_emotion
