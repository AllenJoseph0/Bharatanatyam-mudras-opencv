from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import cvzone

app = Flask(__name__)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mp.solutions.drawing_utils

# Mudra Descriptions
mudra_descriptions = {
    "Pataka": "Flag; used for clouds, wind, denial, etc.",
    "Tripataka": "Like Pataka but with the ring finger bent; represents lightning, crown, or a tree.",
    "Shikaram": "Thumb extended upward, all other fingers folded in; symbolizes a bell or bow.",
    "Ardhapataka": "Half-flag; used for riverbanks, leaves, or a knife.",
    "Kartharimukha": "Scissors face; represents lightning, separation, or opposition.",
    "Mayura": "Peacock; used to depict a peacock's beak, tilak, or sprinkling water.",
    "Ardhachandra": "Half-moon; represents the moon, a spear, or an offering.",
    "Arala": "Bent index finger; denotes drinking nectar, wind, or courage.",
    "Katakamukha": "Bracelet; used for holding objects like flowers or a garland.",
    "Simhamukha": "Lion face; symbolizes a lion, ferocity, or a fire flame.",
    "Sukatunda": "Parrot's beak; represents a shooting arrow or a spear.",
    "Mushti": "Fist; represents power, holding objects, or fighting.",
    "Soochi": "Needle; indicates pointing, showing direction, or a single object.",
    "Chandrakala": "Crescent moon; represents the moon, crown, or marking.",
    "Mrigashirsha": "Deer head; represents deer, a woman's face, or beauty.",
    "Alapadmakam": "Fully bloomed lotus; used to denote fruits, flowers, or offerings.",
    "Hamsasya": "The Swan’s Beak (Elegance, Teaching, Offering)",
    "Trisula": "The Trident (Power, Energy, Divine Force)"
}


def detect_mudra(fingers, distances,landmarks):
    if fingers == [0, 1, 1, 1, 1] and distances['thumb_index'] < 150:
        return "Pataka"
    
    if fingers == [1, 1, 1, 0, 1] and distances['ring_thumb'] > 40:
        return "Tripataka"
    
    if fingers == [1, 0, 0, 0, 0]:
        return "Shikaram"
    
    if fingers == [1, 1, 1, 0, 0]:
        return "Ardhapataka"
    
    if fingers == [0, 1, 1, 0, 0] and 19 <= distances['index_middle'] <= 94:
        return "Kartharimukha"
    
    if fingers == [0, 1, 1, 0, 1] and 12 <= distances['ring_thumb'] <= 40:
        return "Mayura"
    
    if fingers == [1, 1, 1, 1, 1]:
        return "Ardhachandra"
    
    if fingers == [1, 0, 1, 1, 1]:
        return "Arala"
    
    if fingers == [0, 0, 0, 1, 1] and 7 <= distances['middle_thumb'] <= 40 and 5 <= distances['thumb_index'] <= 30 and 10 <= distances['index_middle'] <= 33:
        return "Katakamukha"
    
    if fingers == [0, 1, 0, 0, 1] and 3 <= distances['ring_thumb'] <= 30 and 1 <= distances['middle_thumb'] <= 15 and 1 <= distances['middle_ring'] <= 25:
        return "Simhamukha"
    
    if fingers == [1, 0, 1, 0, 1]:
        return "Sukatunda"
    
    if fingers == [0, 0, 0, 0, 0] and 3 <= distances['thumb_index'] <= 15:
        return "Mushti"
    
    if fingers == [0, 1, 0, 0, 0]:
        return "Soochi"
    
    if fingers == [1, 1, 0, 0, 0]:
        return "Chandrakala"
    
    if fingers == [1, 0, 0, 0, 1]:
        return "Mrigashirsha"
    
    if fingers == [1, 1, 1, 1, 0] and 30 <= distances['thumb_index'] <= 155:
        return "Alapadmakam"
    
    if fingers == [0, 0, 1, 1, 1]:
        return "Hamsasya"
    
    if fingers == [0, 1, 1, 1, 0]:
        return "Trisula"
    
    return "Unknown Mudra"

def process_frame(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    detected_mudra = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in hand_landmarks.landmark]
            fingers = [int(lm[tip][1] < lm[tip - 2][1]) for tip in [8, 12, 16, 20]]
            fingers.insert(0, int(lm[4][0] > lm[3][0]))
            distances = {key: hypot(lm[i][0] - lm[j][0], lm[i][1] - lm[j][1]) for key, (i, j) in {
                'thumb_index': (4, 8), 'index_middle': (8, 12), 'middle_ring': (12, 16),
                'ring_pinky': (16, 20), 'ring_thumb': (4, 16), 'middle_thumb': (4, 12)
            }.items()}
            detected_mudra = detect_mudra(fingers, distances,lm)
            if detected_mudra:
                cvzone.putTextRect(img, detected_mudra, (50, 150), scale=3, thickness=3, colorR=(0, 200, 0))
    return img, detected_mudra

def generate_frames():
    global detected_mudra
    while True:
        success, img = cap.read()
        if not success:
            break
        img, detected_mudra = process_frame(img)
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mudra_info')
def mudra_info():
    return jsonify({
        "mudra": detected_mudra or "None",
        "description": mudra_descriptions.get(detected_mudra, "Waiting for detection...")
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
