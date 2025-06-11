from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
import requests
from PIL import Image
from io import BytesIO

# Define CNN Models
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1),
            nn.Sigmoid()
        )
    def forward(self, x): 
        return self.net(x)

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )
    def forward(self, x): 
        return self.net(x)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model1 = SimpleCNN().to(device)
model2 = DeeperCNN().to(device)
model1.load_state_dict(torch.load("model1.pth", map_location=device))
model2.load_state_dict(torch.load("model2.pth", map_location=device))
model1.eval()
model2.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Ensemble prediction logic
def ensemble_predict(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output1 = model1(image_tensor)
        output2 = model2(image_tensor)
        avg_output = (output1 + output2) / 2
        return "real" if avg_output.item() >= 0.5 else "fake"

# Flask application setup
app = Flask(__name__)

@app.route('/predict-url', methods=['POST'])
def predict_url():
    data = request.get_json()
    image_url = data.get('imageUrl')

    if not image_url:
        return jsonify({'error': 'Invalid input: missing imageUrl'}), 400

    try:
        # Download and process the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_tensor = transform(image)

        # Make prediction
        prediction = ensemble_predict(image_tensor)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)