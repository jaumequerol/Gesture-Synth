import cv2
import mediapipe as mp
import os
import time
import numpy as np
import pygame

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

timestamp_ms = 0

previous_state = (None, 0)
pygame.mixer.init(44100, -16, 2, 1024)
t = np.linspace(0, 1, 44100, False)
synth_channel = pygame.mixer.Channel(0)
synth_channel.set_volume(0.0)

def create_chord(f1, f2, f3):
    wave = (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + np.sin(2*np.pi*f3*t)) / 3
    audio = (wave * 8000).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

f1, f2, f3 = 262, 330, 392
MAJOR_CHORDS = {
    1: create_chord(f1, f2, f3),
    2: create_chord(f1*(2**(2/12)), f2*(2**(2/12)), f3*(2**(2/12))),
    3: create_chord(f1*(2**(4/12)), f2*(2**(4/12)), f3*(2**(4/12))),
    4: create_chord(f1*(2**(5/12)), f2*(2**(5/12)), f3*(2**(5/12))),
    5: create_chord(f1*(2**(7/12)), f2*(2**(7/12)), f3*(2**(7/12))),
    6: create_chord(f1*(2**(9/12)), f2*(2**(9/12)), f3*(2**(9/12))),
    7: create_chord(f1*(2**(11/12)), f2*(2**(11/12)), f3*(2**(11/12))),
    8: create_chord(f1/2, f2/2, f3/2),
    9: create_chord(f1/2*(2**(2/12)), f2/2*(2**(2/12)), f3/2*(2**(2/12))),
    10: create_chord(f1/2*(2**(4/12)), f2/2*(2**(4/12)), f3/2*(2**(4/12))),
    11: create_chord(f1/2*(2**(5/12)), f2/2*(2**(5/12)), f3/2*(2**(5/12))),
    12: create_chord(f1/2*(2**(7/12)), f2/2*(2**(7/12)), f3/2*(2**(7/12))),
    13: create_chord(f1/2*(2**(9/12)), f2/2*(2**(9/12)), f3/2*(2**(9/12))),
    14: create_chord(f1/2*(2**(11/12)), f2/2*(2**(11/12)), f3/2*(2**(11/12))),
    15: create_chord(f1*2, f2*2, f3*2),
    16: create_chord(f1*2*(2**(2/12)), f2*2*(2**(2/12)), f3*2*(2**(2/12))),
    17: create_chord(f1*2*(2**(4/12)), f2*2*(2**(4/12)), f3*2*(2**(4/12))),
    18: create_chord(f1*2*(2**(5/12)), f2*2*(2**(5/12)), f3*2*(2**(5/12))),
    19: create_chord(f1*2*(2**(7/12)), f2*2*(2**(7/12)), f3*2*(2**(7/12))),
    20: create_chord(f1*2*(2**(9/12)), f2*2*(2**(9/12)), f3*2*(2**(9/12))),
    21: create_chord(f1*2*(2**(11/12)), f2*2*(2**(11/12)), f3*2*(2**(11/12)))
}

f1, f2, f3 = 262, 312, 392
MINOR_CHORDS = {
    1: create_chord(f1, f2, f3),
    2: create_chord(f1*(2**(2/12)), f2*(2**(2/12)), f3*(2**(2/12))),
    3: create_chord(f1*(2**(4/12)), f2*(2**(4/12)), f3*(2**(4/12))),
    4: create_chord(f1*(2**(5/12)), f2*(2**(5/12)), f3*(2**(5/12))),
    5: create_chord(f1*(2**(7/12)), f2*(2**(7/12)), f3*(2**(7/12))),
    6: create_chord(f1*(2**(9/12)), f2*(2**(9/12)), f3*(2**(9/12))),
    7: create_chord(f1*(2**(11/12)), f2*(2**(11/12)), f3*(2**(11/12))),
    8: create_chord(f1/2, f2/2, f3/2),
    9: create_chord(f1/2*(2**(2/12)), f2/2*(2**(2/12)), f3/2*(2**(2/12))),
    10: create_chord(f1/2*(2**(4/12)), f2/2*(2**(4/12)), f3/2*(2**(4/12))),
    11: create_chord(f1/2*(2**(5/12)), f2/2*(2**(5/12)), f3/2*(2**(5/12))),
    12: create_chord(f1/2*(2**(7/12)), f2/2*(2**(7/12)), f3/2*(2**(7/12))),
    13: create_chord(f1/2*(2**(9/12)), f2/2*(2**(9/12)), f3/2*(2**(9/12))),
    14: create_chord(f1/2*(2**(11/12)), f2/2*(2**(11/12)), f3/2*(2**(11/12))),
    15: create_chord(f1*2, f2*2, f3*2),
    16: create_chord(f1*2*(2**(2/12)), f2*2*(2**(2/12)), f3*2*(2**(2/12))),
    17: create_chord(f1*2*(2**(4/12)), f2*2*(2**(4/12)), f3*2*(2**(4/12))),
    18: create_chord(f1*2*(2**(5/12)), f2*2*(2**(5/12)), f3*2*(2**(5/12))),
    19: create_chord(f1*2*(2**(7/12)), f2*2*(2**(7/12)), f3*2*(2**(7/12))),
    20: create_chord(f1*2*(2**(9/12)), f2*2*(2**(9/12)), f3*2*(2**(9/12))),
    21: create_chord(f1*2*(2**(11/12)), f2*2*(2**(11/12)), f3*2*(2**(11/12)))
}

CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17)
]

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int(time.time() * 1000)

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            previous_degree = -1
            for start, end in CONNECTIONS:
                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)

                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

            degree = 0

            thumb = True
            if ((hand[4].x - hand[2].x) * (hand[2].x - hand[1].x)) < 0:
                thumb = False
            if ((hand[4].x - hand[2].y) * (hand[2].x - hand[1].y)) < 0:
                thumb = False
            if thumb:
                degree += 1
                cv2.putText(
                    frame,
                    "Thumb",
                    (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

            index = True
            for i in range(5, 9):
                landmark = hand[i]
                if i == 5:
                    previous_y = int(landmark.y * height)
                else:
                    if previous_y < int(landmark.y * height):
                        index = False
                previous_y = int(landmark.y * height)
            if index:
                degree += 1
                cv2.putText(
                    frame,
                    "Index",
                    (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
            
            middle = True
            for i in range(9, 13):
                landmark = hand[i]
                if i == 9:
                    previous_y = int(landmark.y * height)
                else:
                    if previous_y < int(landmark.y * height):
                        middle = False
                previous_y = int(landmark.y * height)
            if middle:
                degree += 1
                cv2.putText(
                    frame,
                    "Middle",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0 ,0),
                    2
                )
            
            ring = True
            for i in range(13, 17):
                landmark = hand[i]
                if i == 13:
                    previous_y = int(landmark.y * height)
                else:
                    if previous_y < int(landmark.y * height):
                        ring = False
                previous_y = int(landmark.y * height)
            if ring:
                degree += 1
                cv2.putText(
                    frame,
                    "Ring",
                    (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
            
            pinky = True
            for i in range(17, 21):
                landmark = hand[i]
                if i == 17:
                    previous_y = int(landmark.y * height)
                else:
                    if previous_y < int(landmark.y * height):
                        pinky = False
                previous_y = int(landmark.y * height)
            if pinky:
                degree += 1
                cv2.putText(
                    frame,
                    "Pinky",
                    (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

            if (thumb and index) and (thumb and pinky) and (not ring) and (not middle):
                degree = 7
            elif pinky and index and (not ring) and (not middle) and not thumb:
                degree = 6

            landmark = hand[20]
            if landmark.x < 0.5:
                is_major = True
                cv2.putText(
                    frame,
                    "Left",
                    (210, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )
                if degree == 1:
                    roman = "I"
                elif degree == 2:
                    roman = "II"
                elif degree == 3:
                    roman = "III"
                elif degree == 4:
                    roman = "IV"
                elif degree == 5:
                    roman = "V"
                elif degree == 6:
                    roman = "VI"
                elif degree == 7:
                    roman = "VII"
            else:
                is_major = False
                cv2.putText(
                    frame,
                    "Right",
                    (210, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )
                if degree == 1:
                    roman = "i"
                elif degree == 2:
                    roman = "ii"
                elif degree == 3:
                    roman = "iii"
                elif degree == 4:
                    roman = "iv"
                elif degree == 5:
                    roman = "v"
                elif degree == 6:
                    roman = "vi"
                elif degree == 7:
                    roman = "vii"
                
            cv2.putText(
                frame,
                "Degree: " + roman,
                (210, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )

            current_state = (is_major, degree)

            if current_state != previous_state:
                pygame.mixer.stop()
                if is_major and degree in MAJOR_CHORDS:
                    synth_channel.play(MAJOR_CHORDS[degree], loops=-1)
                elif not is_major and degree in MINOR_CHORDS:
                    synth_channel.play(MINOR_CHORDS[degree], loops=-1)
                previous_state = current_state

            z_sum = 0
            for i, landmark in enumerate(hand):
                z_sum += landmark.z
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0, 0, 0),
                    -1
                )
            
                previous_y = y

            volume = (-10 * z_sum - 3) / 17
            if volume >= 1:
                volume = 1
            elif volume <= 0:
                volume = 0

            cv2.putText(
                frame,
                "Volume: " + str(int(volume * 100)) + "%",
                (360, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            synth_channel.set_volume(volume)

        cv2.putText(
            frame,
            "HAND DETECTED",
            (360, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        if previous_state != (None, 0):
            pygame.mixer.stop()
            previous_state = (None, 0)
        
        cv2.putText(
            frame,
            "Searching for hand...",
            (320, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Gesture Synth - Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

pygame.quit()
camera.release()
detector.close()
cv2.destroyAllWindows()

