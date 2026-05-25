# AI Face Recognition System using OpenCV & Dlib

> Real-time Face Recognition System implemented with Modern C++, OpenCV, and Dlib
> 멀티스레드 기반 실시간 얼굴 인식 시스템 프로젝트

---

# 📌 Project Overview

본 프로젝트는 **Modern C++**, **OpenCV**, 그리고 **Dlib Deep Learning Model**을 활용하여
실시간 영상 또는 이미지 속 인물을 탐지하고 식별하는 AI 얼굴 인식 시스템입니다.

단순 얼굴 검출(Face Detection)을 넘어:

* 얼굴 정렬(Face Alignment)
* 128차원 Face Embedding 추출
* Euclidean Distance 기반 인물 분류
* 멀티스레드 기반 실시간 처리
* FPS 최적화

까지 구현한 프로젝트입니다.

---

# 🎥 Demo Features

✅ Image Face Recognition
✅ Real-time Video Face Recognition
✅ Multi-threaded Processing
✅ FPS Counter
✅ Face Alignment using Landmark Detection
✅ 128D Face Embedding (ResNet)
✅ Euclidean Distance Face Matching
✅ Average Descriptor Learning
✅ Video UI Overlay System

---

# 🧠 System Architecture

```text
                +------------------+
                |   Video Input    |
                |  OpenCV Capture  |
                +---------+--------+
                          |
                          v
                +------------------+
                |  Face Detection  |
                | Dlib HOG Detector|
                +---------+--------+
                          |
                          v
                +------------------+
                | Face Alignment   |
                | Landmark Model   |
                +---------+--------+
                          |
                          v
                +------------------+
                | 128D Embedding   |
                | ResNet Network   |
                +---------+--------+
                          |
                          v
                +------------------+
                | Face Recognition |
                | Distance Compare |
                +---------+--------+
                          |
                          v
                +------------------+
                | Render & Display |
                +------------------+
```

---

# ⚙️ Development Environment

| Category | Description        |
| -------- | ------------------ |
| Language | C++17              |
| IDE      | Visual Studio 2022 |
| Library  | OpenCV             |
| Library  | Dlib               |
| OS       | Windows 10 / 11    |

---

# 📚 Libraries Used

## OpenCV

* Video Input / Output
* GUI Rendering
* Image Processing
* FPS Display
* UI Overlay

## Dlib

* Face Detection
* Landmark Detection
* Face Alignment
* Deep Metric Learning
* 128D Face Embedding

---

# 🧩 Core Technologies

---

## 1. Face Detection

Dlib의 `frontal_face_detector`를 활용하여 얼굴 영역을 검출합니다.

```cpp
frontal_face_detector detector_;
```

---

## 2. Face Alignment

Landmark 기반 얼굴 정렬을 수행합니다.

```cpp
extract_image_chip(
    cimg,
    get_face_chip_details(shape, 150, 0.25),
    faceChip
);
```

얼굴을 정규화하여 인식 정확도를 향상시켰습니다.

---

## 3. 128D Face Embedding

Dlib의 ResNet 기반 Face Recognition Model을 사용합니다.

```cpp
using anet_type = loss_metric<
```

각 얼굴은 128차원 특징 벡터로 변환됩니다.

---

## 4. Face Recognition

Euclidean Distance 기반으로 등록된 얼굴과 비교합니다.

```cpp
double distance = length(descriptor - known.descriptor);
```

Threshold 기반으로 동일 인물 여부를 판별합니다.

---

## 5. Average Descriptor Learning

동일 인물의 여러 이미지를 평균화하여 인식 안정성을 향상시켰습니다.

```cpp
known.descriptor =
    (known.descriptor * known.sampleCount + desc)
    / (known.sampleCount + 1);
```

---

## 6. Multi-threading Optimization

영상 출력과 얼굴 인식을 분리하여 FPS 저하를 최소화했습니다.

```cpp
std::thread worker(faceWorker, std::ref(faceSystem));
```

---

# 🚀 Execution Flow

```text
Program Start
    ↓
Load Dlib Models
    ↓
Load Known Faces
    ↓
Video / Image Input
    ↓
Face Detection
    ↓
Face Alignment
    ↓
128D Embedding Extraction
    ↓
Distance Comparison
    ↓
Recognition Result
    ↓
Render UI & FPS
```

---

# 📂 Project Structure

```text
📦 FaceRecognitionProject
 ┣ 📂 known_faces
 ┃ ┣ 📜 karina_1.jpg
 ┃ ┣ 📜 winter_1.jpg
 ┃ ┗ 📜 ...
 ┣ 📜 main.cpp
 ┣ 📜 shape_predictor_5_face_landmarks.dat
 ┣ 📜 dlib_face_recognition_resnet_model_v1.dat
 ┗ 📜 README.md
```

---

# 🖥️ Program Modes

## 🖼️ Image Mode

* 폴더 내 이미지 자동 탐색
* 얼굴 인식 결과 출력
* FPS 측정 지원

---

## 🎬 Video Mode

* 실시간 얼굴 인식
* 멀티스레드 기반 처리
* FPS Counter
* Pause / Resume 기능
* UI Overlay 지원

---

# 🎮 Controls

| Key | Function     |
| --- | ------------ |
| P   | Play / Pause |
| ESC | Exit Program |

---

# 📈 Performance Optimization

본 프로젝트는 실시간 처리를 위해 다음과 같은 최적화를 적용했습니다.

* Multi-threading
* Mutex Synchronization
* Atomic Variables
* Frame Cloning Optimization
* Average Descriptor Learning
* Jittering-based Embedding Stabilization

---

# 🧪 Recognition Pipeline

```text
Input Frame
    ↓
Face Detection
    ↓
Landmark Extraction
    ↓
Face Alignment
    ↓
Face Embedding
    ↓
Distance Matching
    ↓
Recognition Result
```

---

# 📸 Sample Result

```text
Karina 0.42
Winter 0.51
Unknown 0.78
```

* 낮은 distance일수록 동일 인물일 확률이 높음
* Threshold 기준으로 Unknown 처리 수행

---

# 📊 Threshold Strategy

| Threshold | Description |
| --------- | ----------- |
| 0.45      | Strict      |
| 0.60      | Recommended |
| 0.70      | Loose       |

현재 프로젝트 권장값:

```cpp
threshold = 0.6
```

---

# 🔥 Key Features

✅ Real-time Face Recognition
✅ Deep Learning Face Embedding
✅ Modern C++ Architecture
✅ OpenCV + Dlib Integration
✅ Multi-threaded Video Processing
✅ FPS Monitoring System
✅ AI-based Distance Matching

---

# 📌 Future Improvements

* GPU CUDA Acceleration
* Face Tracking
* Cosine Similarity Matching
* Dynamic Threshold System
* YOLO / RetinaFace Integration
* Face Database Management
* GUI Application Upgrade

---

# 🎓 Educational Purpose

본 프로젝트는 다음 학습 목표를 기반으로 제작되었습니다.

* Modern C++ Programming
* OpenCV Computer Vision
* Deep Learning Integration
* Real-time Video Processing
* Multi-threading System Design
* AI Face Recognition Pipeline

---

# 👨‍💻 Author

### Face Recognition Project using OpenCV & Dlib

Developed with:

* Modern C++
* OpenCV
* Dlib
* Multi-threading

---

# ⭐ Reference

* Dlib Official Example
* OpenCV Documentation
* dlib-models Repository

---

# 📜 License

This project is created for educational and research purposes.
