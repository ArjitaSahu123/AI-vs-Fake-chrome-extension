# 🧠 AI-vs-Fake Chrome Extension

A powerful AI-powered Chrome extension that detects whether an image is **real or fake** using deep learning models — directly in your browser.

## 🚀 Live Preview
> Upload any image via the extension popup and get an instant prediction (real or fake) based on ensemble deep learning models.

---

## 🔍 Why AI-vs-Fake? (USP Highlights)

Unlike traditional reverse-image search or metadata-checker tools, **AI-vs-Fake** uses **convolutional neural networks (CNNs)** to analyze the **image content itself**, detecting deepfakes or synthetic images — even when metadata is stripped or altered.

### ✅ Unique Features

- 🧠 **Model-Based Detection**: Uses two deep learning models (SimpleCNN & DeeperCNN) trained on real vs. fake images.
- 🔗 **Runs from Your Browser**: Lightweight Chrome extension for direct access without switching tabs.
- 🔌 **Flask API Integration**: Uses a custom Python backend with OpenCV + PyTorch for prediction.
- 🧠 **Ensemble Prediction**: Combines predictions from multiple models for higher accuracy.
- 🔍 **Content-Based**: Goes beyond metadata and reverse search by inspecting visual patterns.
- 🌐 **Customizable & Open Source**: Modify or improve models and frontend easily.

---

## 🛠️ How It Works

1. 🖼️ Upload an image via the extension popup.
2. 🔁 Image is sent to a Flask backend.
3. 🧠 Backend uses PyTorch models to make predictions.
4. 🔙 Results are returned as "real" or "fake" in the popup.

---

## 🏗️ Tech Stack

| Frontend | Backend        | ML Models       |
|----------|----------------|-----------------|
| HTML/CSS/JS (Popup) | Flask + Python | SimpleCNN, DeeperCNN (PyTorch) |

---

### Folder Structure
AI-vs-Fake-chrome-extension/
├── backend/
│   ├── app.py
│   ├── model1.pth
│   ├── model2.pth
│   └── requirements.txt
├── chrome-extension/
│   ├── popup.html
│   ├── manifest.json
│   └── images/
│       └── icon16.png
├── README.md


## 📦 Installation

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py           # Run the Flask server
```

### 2.Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```
### 3.Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Flask server:

```bash
cd backend
python app.py
```

## Server will start at http://127.0.0.1:5000/predict

### 🌐 Chrome Extension
- Go to chrome://extensions/ in Chrome
- Enable Developer Mode (top right toggle)
- Click "Load Unpacked"
- Select the chrome-extension/ folder

- The extension icon will appear in your Chrome toolbar


### ⚙️ Do Users Need Developer Tools?
- 🛠️ No! End users don’t need to use Developer Tools.
- Developer Mode is only required during development or testing.
- Once the extension is published on the Chrome Web Store, users can install it directly.

### 📦 Requirements
- Python 3.8+
- pip
- Flask
- PyTorch
- OpenCV
- TorchVision
  
### Screenshot
![image](https://github.com/user-attachments/assets/155337c9-d57e-4793-b43d-7dc1d828c349)



### Google colab notebook for ml model training
- https://drive.google.com/file/d/1M7qlWe-L-AGGHDiS8MdcygQsCBhU6Sd5/view?usp=sharing
  
### 🤝 Contributing
- Fork the repository
- Create a new branch (git checkout -b feature-branch)
- Commit your changes (git commit -m 'Add feature')
- Push to the branch (git push origin feature-branch)
- Open a Pull Request 🎉

### 👩‍💻 Author
- Arjita Sahu
- 📧 Contact: [arjitasahu.2020@gmail.com]
- 🔗 GitHub: ArjitaSahu123
  
