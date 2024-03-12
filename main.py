import numpy as np
import mediapipe as mp
import math
import cv2
import osascript
import time
import tkinter as tk

def detectLandmarks(frameIn, handsLandmarker: mp.solutions.hands.Hands()) -> tuple[bool, list, list]:

    # Get hand landmark prediction
    result = handsLandmarker.process(frameIn)

    # post process the result
    if result.multi_hand_landmarks:

        normalizedLandmarksOut = []
        for handslms in result.multi_hand_landmarks:
             normalizedLandmarksOut.append(handslms)

        worldLandmarksOut = []
        for handslms in result.multi_hand_world_landmarks:
            for lm in handslms.landmark:
                worldLandmarksOut.append([lm.x, lm.y])

        return True, normalizedLandmarksOut, worldLandmarksOut
    
    return False, [], []

def calculatePercentage(worldLandmarksIn: list) -> int:

    # Store landmarks
    thumbX = worldLandmarksIn[4][0]
    thumbY = worldLandmarksIn[4][1]
    pointerX = worldLandmarksIn[8][0]
    pointerY = worldLandmarksIn[8][1]

    # Calculate distance and normalize
    dist = math.dist([thumbX, thumbY], [pointerX, pointerY])
    scaledDist = int((dist-0.005)/(0.095-0.005)*100)

    if scaledDist > 100: scaledDist = 100
    elif scaledDist < 0: scaledDist = 0

    return scaledDist

def collectFrame(capIn: cv2.VideoCapture()) -> tuple[np.ndarray, np.ndarray]:

    # Read in frame
    _, frameTemp = capIn.read()

    # Flip the frame vertically
    frameColored = cv2.flip(frameTemp, 1)
    frameOut = cv2.cvtColor(frameColored, cv2.COLOR_BGR2RGB)

    return frameOut, frameColored

def drawLandmarks(drawUtils: mp.solutions.drawing_utils, frameIn: np.ndarray, handLMS: list):
    drawUtils.draw_landmarks(frameIn, handLMS, [(4,8)])

def outputFrame (frameIn: np.ndarray):
    cv2.imshow("Output", frameIn)

def setVolume(percentage: int):
    command = "set volume output volume " + str(percentage)
    osascript.osascript(command)

def buildGUI():
    window = tk.Tk()
    

def main():

    frameCounter = 0

    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    percentage = -1

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        frameDetect, frameOutput = collectFrame(capIn= cap)

        handsDetected, normalizedLandmarks, worldLandmarks = detectLandmarks(frameIn= frameDetect, handsLandmarker= hands)

        if handsDetected:

            drawLandmarks(mpDraw, frameOutput, normalizedLandmarks[0])

            if frameCounter >= 15:
                percentage = calculatePercentage(worldLandmarksIn= worldLandmarks)
                setVolume(percentage)
                frameCounter = 0

            else: frameCounter += 1
        
        # Show the final output
        outputFrame(frameIn=frameOutput)

        if cv2.waitKey(1) == ord('q'):
            break
    
    # release the webcam and destroy all active windows
    cap.release()

main()