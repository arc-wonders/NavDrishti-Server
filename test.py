import cv2
import requests
import numpy as np
import time
from ultralytics import YOLO


# =====================
# CONFIG
# =====================

PI_URL = "http://raspberrypi.local:8000"

MODEL_PATH = "yolov8n-face.pt"

IMG_SIZE = 256
CONFIDENCE = 0.45

# smooth control params
Kp = 0.05           # responsiveness (0.03 - 0.08)
SMOOTH = 0.8        # smoothing factor (0.7 - 0.9)
DEADZONE = 20       # ignore tiny movements

SHOW_FPS = True


# =====================
# LOAD MODEL
# =====================

model = YOLO(MODEL_PATH)

angle = 90.0
last_sent_angle = 90.0

stream_url = f"{PI_URL}/video"

print("Connecting to stream:", stream_url)

stream = requests.get(stream_url, stream=True)

bytes_data = b""

last_time = time.time()


# =====================
# MAIN LOOP
# =====================

while True:

    bytes_data += stream.raw.read(2048)

    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')

    if a != -1 and b != -1:

        jpg = bytes_data[a:b+2]

        bytes_data = bytes_data[b+2:]

        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue


        # =====================
        # YOLO INFERENCE
        # =====================

        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )


        frame_center = frame.shape[1] // 2


        for r in results:

            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()

            if len(boxes) == 0:
                continue


            # choose biggest face
            largest = max(
                boxes,
                key=lambda b:(b[2]-b[0])*(b[3]-b[1])
            )


            x1,y1,x2,y2 = largest.astype(int)

            face_center = (x1+x2)//2

            error = face_center - frame_center


            # =====================
            # SMOOTH PAN CONTROL
            # =====================

            if abs(error) > DEADZONE:

                # proportional movement
                delta = Kp * error

                target_angle = angle + delta

                target_angle = max(0, min(180, target_angle))

                # smoothing filter
                angle = SMOOTH * angle + (1-SMOOTH) * target_angle


                # send only if change noticeable
                if abs(angle - last_sent_angle) > 0.7:

                    try:

                        requests.post(
                            f"{PI_URL}/set_angle",
                            json={"angle": int(angle)},
                            timeout=0.1
                        )

                        last_sent_angle = angle

                    except:
                        print("connection lost")


            # draw face box
            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )


        # =====================
        # UI
        # =====================

        # center reference line
        cv2.line(
            frame,
            (frame_center,0),
            (frame_center,frame.shape[0]),
            (255,0,0),
            2
        )


        if SHOW_FPS:

            fps = 1/(time.time()-last_time)

            last_time = time.time()

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )


        cv2.imshow(
            "YOLO Smooth Face Tracking",
            frame
        )


        if cv2.waitKey(1) == 27:
            break


cv2.destroyAllWindows()