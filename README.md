# 🪷 MudraVision – Bharatanatyam Mudra Recognition

## About the Project

MudraVision is a Computer Vision-based application that recognizes Bharatanatyam hand gestures (mudras) in real time using a webcam. It identifies the mudra and provides its meaning through automatically generated MP3 audio.

## Technologies Used

- Python
- TensorFlow & Keras
- OpenCV
- NumPy
- Streamlit
- Google Text-to-Speech (gTTS)

## Recognized Mudras

The application identifies four Bharatanatyam mudras:

- Pataka
- Tripataka
- Ardhapataka
- Kartarimukha

## Key Features

- Real-time webcam-based mudra recognition.
- Deep learning-based gesture classification.
- Confidence threshold-based prediction (75%).
- Displays detected mudra and prediction confidence.
- Provides the traditional meaning (Viniyoga) of each mudra.
- Generates and plays MP3 audio explanations.
- Start and Stop controls through Streamlit.

## How It Works

1. Captures hand gestures through a webcam.
2. Processes the captured image using OpenCV.
3. Resizes the image to 224 × 224 pixels.
4. Uses a trained TensorFlow/Keras model to predict the mudra.
5. Checks the prediction confidence.
6. Displays the recognized mudra and its meaning.
7. Generates and plays the corresponding audio explanation.

## Project Files

- `mudravision_app.py` – Main application.
- `mudra_model1.keras` – Trained deep learning model.
- `requirements.txt` – Required Python libraries.

## How to Run

Install the required libraries:

```bash
pip install streamlit opencv-python numpy tensorflow gTTS
```

Run the application:

```bash
streamlit run mudravision_app.py
```

Make sure the trained model `mudra_model1.keras` is available in the project directory and a webcam is connected.

## Author

**Mohammed Safwan K K**

B.Tech – Artificial Intelligence and Data Science

[GitHub](https://github.com/safwankkmohammed) | [LinkedIn](https://linkedin.com/in/mohammed-safwan-k-k)
