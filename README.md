# 🤟 Gemma Sign Language Interpreter

A real-time, offline **Sign Language to Text & Speech** translator using **MediaPipe**, **OpenCV**, and **Google's Gemma 3n** model.  
Built for the **Gemma 3n Impact Challenge**, this app empowers communication for the hearing and speech impaired with privacy-first, on-device AI.

---

## 📌 Features

- 🖐️ Real-time hand gesture detection with **MediaPipe**
- 📸 Webcam integration for live sign tracking
- 🧠 Trained gesture classification model for ASL signs
- 🗣️ Converts gestures into **text** and **speech (via pyttsx3)**
- 🔒 Works **offline** — no internet required
- 🧩 Integrates **Gemma 3n** via HuggingFace (local model)
- 🪄 Lightweight, fast, and built with accessibility in mind

---

## 🧠 Motivation

Sign language users often face communication barriers in verbal environments.  
This project bridges that gap by converting hand gestures to voice/text **instantly** and **privately**, using on-device AI.

---

## 🛠️ Technologies Used

| Component             | Technology Used                        |
|----------------------|----------------------------------------|
| Hand Tracking         | MediaPipe Hands + OpenCV               |
| Gesture Classification| Custom-trained model (NumPy, Sklearn) |
| AI NLP                | Gemma 3n (HuggingFace integration)     |
| Voice Output          | pyttsx3 (offline TTS)                  |
| GUI                   | `tkinter` / CLI                        |
| Packaging             | PyInstaller (`.exe`)                   |

---

## ⚙️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/NuhashMaq/GemmaSignApp.git
   cd GemmaSignApp
