import cv2
import mediapipe as mp
import os
import time
import numpy as np
import pygame




# ============================================================
# CONFIGURACIÓ
# ============================================================

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


# Crear el detector
detector = HandLandmarker.create_from_options(options)


# ============================================================
# CONNEXIÓ AMB LA CÀMERA
# ============================================================

camara = cv2.VideoCapture(0)

print("----------------------------------------")
print("GESTURE SYNTH")
print("----------------------------------------")
print("Càmera iniciada correctament.")
print("Posa la mà davant de la càmera.")
print("Prem Q per sortir.")
print("----------------------------------------")


# Timestamp inicial
timestamp_ms = 0


# ============================================================
# CONNEXIONS DE LA MÀ
# ============================================================
estat_anterior = (None, 0)
pygame.mixer.init(44100, -16, 2, 1024)
t = np.linspace(0, 1, 44100, False)

def crear_acord(f1, f2, f3):
    wave = (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + np.sin(2*np.pi*f3*t)) / 3
    audio = (wave * 8000).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

# Guardem els acords generats en un diccionari
f1, f2, f3 = 262, 330, 392
MAJORS = {
    1: crear_acord(f1, f2, f3),
    2: crear_acord(f1*(2**(2/12)), f2*(2**(2/12)), f3*(2**(2/12))),
    3: crear_acord(f1*(2**(4/12)), f2*(2**(4/12)), f3*(2**(4/12))),
    4: crear_acord(f1*(2**(5/12)), f2*(2**(5/12)), f3*(2**(5/12))),
    5: crear_acord(f1*(2**(7/12)), f2*(2**(7/12)), f3*(2**(7/12))),
    6: crear_acord(f1*(2**(9/12)), f2*(2**(9/12)), f3*(2**(9/12))),
    7: crear_acord(f1*(2**(11/12)), f2*(2**(11/12)), f3*(2**(11/12)))
}
f1, f2, f3 = 262, 312, 392
MENORS = {
    1: crear_acord(f1, f2, f3),
    2: crear_acord(f1*(2**(2/12)), f2*(2**(2/12)), f3*(2**(2/12))),
    3: crear_acord(f1*(2**(4/12)), f2*(2**(4/12)), f3*(2**(4/12))),
    4: crear_acord(f1*(2**(5/12)), f2*(2**(5/12)), f3*(2**(5/12))),
    5: crear_acord(f1*(2**(7/12)), f2*(2**(7/12)), f3*(2**(7/12))),
    6: crear_acord(f1*(2**(9/12)), f2*(2**(9/12)), f3*(2**(9/12))),
    7: crear_acord(f1*(2**(11/12)), f2*(2**(11/12)), f3*(2**(11/12)))
}

# Els 21 punts de la mà de MediaPipe
CONNECTIONS = [
    # Polze
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Índex
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Mig
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Anular
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Menovell
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palma
    (0, 17)
]


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:

    # Llegir imatge de la càmera
    correcte, frame = camara.read()

    if not correcte:
        print("ERROR: No es pot llegir la càmera.")
        break


    # Efecte mirall
    frame = cv2.flip(frame, 1)


    # Dimensions de la imatge
    height, width, _ = frame.shape


    # Convertir BGR -> RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Crear objecte MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )


    # Timestamp en mil·lisegons
    timestamp_ms = int(time.time() * 1000)


    # Detectar la mà
    resultat = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # DIBUIXAR LA MÀ
    # ========================================================

    if resultat.hand_landmarks:

        # Agafem la primera mà detectada
       
        for hand in resultat.hand_landmarks:



        # ----------------------------------------------------
        # DIBUIXAR LES LÍNIES
        # ----------------------------------------------------
            grau_anterior = -1
            for inici, final in CONNECTIONS:

                x1 = int(hand[inici].x * width)
                y1 = int(hand[inici].y * height)

                x2 = int(hand[final].x * width)
                y2 = int(hand[final].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )


            # ----------------------------------------------------
            # DIBUIXAR ELS 21 PUNTS
            # ----------------------------------------------------
            grau = 0

            polze = True
            if ((hand[4].x-hand[2].x)*(hand[2].x-hand[1].x)) < 0:
                polze = False
            if ((hand[4].x-hand[2].y)*(hand[2].x-hand[1].y)) < 0:
                polze = False
            if polze:
                grau += 1
                cv2.putText(
                    frame,
                    "Polze",
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
                    y_anterior = int(landmark.y * height)
                else:
                    if y_anterior < int(landmark.y * height):
                        index = False
                y_anterior = int(landmark.y * height)
            if index:
                grau += 1
                cv2.putText(
                    frame,
                    "Index",
                    (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
            
            mig = True
            for i in range(9, 13):
                landmark = hand[i]
                if i == 9:
                    y_anterior = int(landmark.y * height)
                else:
                    if y_anterior < int(landmark.y * height):
                        mig = False
                y_anterior = int(landmark.y * height)
            if mig:
                grau += 1
                cv2.putText(
                    frame,
                    "Mig",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0 ,0),
                    2
                )
            
            anular = True
            for i in range(13,17):
                landmark = hand[i]
                if i == 13:
                    y_anterior = int(landmark.y * height)
                else:
                    if y_anterior < int(landmark.y * height):
                        anular = False
                y_anterior = int(landmark.y * height)
            if anular:
                grau += 1
                cv2.putText(
                    frame,
                    "Anular",
                    (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
            
            petit = True
            for i in range(17, 21):
                landmark = hand[i]
                if i == 17:
                    y_anterior = int(landmark.y * height)
                else:
                    if y_anterior < int(landmark.y * height):
                        petit = False
                y_anterior = int(landmark.y * height)
            if petit:
                grau += 1
                cv2.putText(
                    frame,
                    "Petit",
                    (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
            if (polze and index) and (polze and petit) and (not(anular)) and (not(mig)):
                grau = 7
            elif petit and index and (not(anular)) and (not(mig)) and not(polze):
                grau = 6
            cv2.putText(
                frame,
                str(grau),
                (150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )
            
            
            """alternativa si no es giren les mans
            landmark1 = hand[20]
            landmark2 = hand[4]

            if landmark1.x-landmark2.x < 0:
            """

            landmark = hand[20]
            if landmark.x < 0.5:
                es_major = True
                cv2.putText(
                    frame,
                    str("Esquerra"),
                    (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
            else:
                es_major = False
                cv2.putText(
                    frame,
                    "Dreta",
                    (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
            estat_actual = (es_major, grau)

            if estat_actual != estat_anterior:
                pygame.mixer.stop()
                if es_major and grau in MAJORS:
                    MAJORS[grau].play(-1)
                elif not es_major and grau in MENORS:
                    MENORS[grau].play(-1)
                estat_anterior = estat_actual


            for i, landmark in enumerate(hand):

                x = int(landmark.x * width)
                y = int(landmark.y * height)



                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0, 255, 0),
                    -1
                )
            
                y_anterior = y
           
                


        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

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
        if estat_anterior != (None, 0):
            pygame.mixer.stop()
            estat_anterior = (None, 0)
        
        cv2.putText(
            frame,
            "Searching for a hand...",
            (320, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # MOSTRAR LA CÀMERA
    # ========================================================

    cv2.imshow(
        "Gesture Synth - Prova",
        frame
    )


    # ========================================================
    # SORTIR
    # ========================================================

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break


# ============================================================
# TANCAR TOT
# ============================================================

print("Tancant programa...")
pygame.quit()
camara.release()
detector.close()
cv2.destroyAllWindows()

print("Programa tancat correctament.")

