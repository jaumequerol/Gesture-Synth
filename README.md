# Gesture-Controlled Synthesizer

A Python-based software synthesizer controlled entirely by hand gestures. It uses a standard webcam to track hand movements, translating finger positions into musical chords.

## Interaction Design

The system currently tracks one hand at a time and maps physical gestures to musical outputs.

### 1. Tonality

* **Right Hand:** Major chords.
* **Left Hand:** Minor chords.

### 2. Chord Selection

The specific chord degree is triggered by the combination of raised fingers:

* **1 to 5 Fingers:** Degrees I through V.
* **Index + Pinky:** Degree VI.
* **Index + Pinky + Thumb:** Degree VII.

## Tech Stack

* **Python 3.x**
* **MediaPipe:** Real-time hand landmark detection.
* **PyGame:** Audio generation and playback.
* **OpenCV:** Webcam video capture and visualisation.
* **NumPy:** Audio waveform generation.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/jqredo/Gesture-Synth.git
   ```

2. Enter the project directory:

   ```bash
   cd Gesture-Synth
   ```

3. Install the required dependencies:

   ```bash
   pip install opencv-python mediapipe numpy pygame
   ```

4. Make sure the MediaPipe model is located at:

   ```text
   models/hand_landmarker.task
   ```

5. Run the program:

   ```bash
   python synth.py
   ```

6. Place your hand in front of the webcam and use your fingers to control the chords.

7. Press **Q** to exit.

## Controls

| Gesture               | Output      |
| --------------------- | ----------- |
| Right hand            | Major chord |
| Left hand             | Minor chord |
| 1 finger              | Degree I    |
| 2 fingers             | Degree II   |
| 3 fingers             | Degree III  |
| 4 fingers             | Degree IV   |
| 5 fingers             | Degree V    |
| Index + Pinky         | Degree VI   |
| Index + Pinky + Thumb | Degree VII  |

## Project Status

The current version supports real-time hand tracking and gesture-controlled chord generation using a webcam.
