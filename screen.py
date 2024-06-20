import sys
import os
import cv2
import dlib
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
import mss
import pyautogui

# Initialize dlib face detector and predictor
if getattr(sys, 'frozen', False):
    LOL_FILE = os.path.join(sys._MEIPASS, 'shape_predictor_68_face_landmarks.dat')
else:
    LOL_FILE = 'shape_predictor_68_face_landmarks.dat'
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LOL_FILE)

def overlay_face(background_frame, face_frame):
    gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
    
    faces_background = detector(gray_background)
    faces_face = detector(gray_face)
    
    if len(faces_background) == 0 or len(faces_face) == 0:
        return background_frame

    shape_background = predictor(gray_background, faces_background[0])
    shape_face = predictor(gray_face, faces_face[0])

    points_background = np.array([[p.x, p.y] for p in shape_background.parts()], np.int32)
    points_face = np.array([[p.x, p.y] for p in shape_face.parts()], np.int32)

    hull_background = cv2.convexHull(points_background)
    hull_face = cv2.convexHull(points_face)

    mask_background = np.zeros_like(background_frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask_background, hull_background, (255, 255, 255))

    rect_background = cv2.boundingRect(hull_background)
    rect_face = cv2.boundingRect(hull_face)

    rect_face = (
        max(0, rect_face[0]),
        max(0, rect_face[1]),
        min(rect_face[2], face_frame.shape[1] - rect_face[0]),
        min(rect_face[3], face_frame.shape[0] - rect_face[1])
    )
    rect_background = (
        max(0, rect_background[0]),
        max(0, rect_background[1]),
        min(rect_background[2], background_frame.shape[1] - rect_background[0]),
        min(rect_background[3], background_frame.shape[0] - rect_background[1])
    )

    center = (rect_background[0] + rect_background[2] // 2, rect_background[1] + rect_background[3] // 2)
    x1, y1, w1, h1 = rect_background
    x2, y2, w2, h2 = rect_face

    if x2 < 0 or y2 < 0 or x2 + w2 > face_frame.shape[1] or y2 + h2 > face_frame.shape[0]:
        return background_frame

    face_region = face_frame[y2:y2+h2, x2:x2+w2]
    face_region_resized = cv2.resize(face_region, (w1, h1))

    if center[0] < 0 or center[1] < 0 or center[0] >= background_frame.shape[1] or center[1] >= background_frame.shape[0]:
        return background_frame

    mask_face = np.zeros((h1, w1, 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask_face, hull_face - np.array([x2, y2]), (255, 255, 255))

    seamless_clone = cv2.seamlessClone(
        face_region_resized, background_frame, mask_face[:, :, 0], center, cv2.MIXED_CLONE
    )

    return seamless_clone

def main():
    sct = mss.mss()

    monitor = sct.monitors[1]  # Change the monitor index if necessary

    cap2 = cv2.VideoCapture(0)

    if not cap2.isOpened():
        raise RuntimeError('Could not open video sources')

    pref_width = 1280
    pref_height = 720
    pref_fps_in = 60
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
    cap2.set(cv2.CAP_PROP_FPS, pref_fps_in)

    width = pref_width
    height = pref_height
    fps = 60

    with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR, device="Unity Video Capture") as cam:
        while True:
            sct_img = sct.grab(monitor)
            screen_frame = np.array(sct_img)[:, :, :3]
            screen_frame = cv2.resize(screen_frame, (width, height))

            ret2, background_frame = cap2.read()
            if not ret2:
                break

            frame = overlay_face(background_frame, screen_frame)

            cam.send(frame)
            cam.sleep_until_next_frame()

if __name__ == "__main__":
    main()
