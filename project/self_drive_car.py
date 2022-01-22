import cv2
import os
import numpy as np
import time
import serial

stop_sign = cv2.CascadeClassifier("CascadeClassifier/stop_sign_classifier_2.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

folderPath = "photo"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
# Define state value
state = {"state": 1}

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# define motor pins
IN1 = 4  #GPIO Pin GPIO 4 /7
IN2 = 17 #GPIO Pin GPIO 17/ 11
IN3 = 27 #GPIO Pin GPIO 27/ 13
IN4 = 22 #GPIO Pin GPIO 22/ 15
En1 = 23 #GPIO Pin GPIO 23/ 16
En2 = 24 #GPIO Pin GPIO 24/ 18

# Set up output
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(En1, GPIO.OUT)
GPIO.setup(En2, GPIO.OUT)

# Set up pwn
R_PWM = GPIO.PWM(En1, 100)
L_PWM = GPIO.PWM(En2, 100)
R_PWM.start(0)
L_PWM.start(0)

def move_forward(): # robot move forward
    GPIO.output(IN1, 1)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 1)
    GPIO.output(IN4, 0)
    R_PWM.ChangeDutyCycle(30)
    L_PWM.ChangeDutyCycle(30)
    print('move forward')

def move_back():    # robot move back
    #time.sleep(10)
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 1)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 1)
    R_PWM.ChangeDutyCycle(30)
    L_PWM.ChangeDutyCycle(25)
    print('move back')

def move_right():  # robot move right
    GPIO.output(IN1, 1)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 0)
    R_PWM.ChangeDutyCycle(20)
    L_PWM.ChangeDutyCycle(20)
    print('turn right')

def move_left():   # robot move left
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 1)
    GPIO.output(IN4, 0)
    R_PWM.ChangeDutyCycle(20)
    L_PWM.ChangeDutyCycle(20)
    print('turn left')

def stop(): # robot stop
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 0)
    R_PWM.ChangeDutyCycle(0)
    L_PWM.ChangeDutyCycle(0)
    print('Stop')

def stopSign():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_sign_detect = stop_sign.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in stop_sign_detect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'STOP', (x + 6, y - 6), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        state.update({"state": 0})
        img[0:100, 0:100] = overlayList[3]
        stop() #calling stop function
        print("Stop Sign Detect")

def trafficSign():
    if state == {'state': 1}:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # color range
        lower_red = np.array([0, 150, 150])
        upper_red = np.array([10, 255, 255])

        lower_green = np.array([40, 100, 100])
        upper_green = np.array([90, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        size = img.shape
        # print(size)

        # hough circle detect
        r_circles = cv2.HoughCircles(mask_red, cv2.HOUGH_GRADIENT, 1, 80,
                                     param1=50, param2=10, minRadius=0, maxRadius=30)

        g_circles = cv2.HoughCircles(mask_green, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=0, maxRadius=30)

        # traffic light detect
        r = 5
        bound = 4.0 / 10
        if r_circles is not None:
            r_circles = np.uint16(np.around(r_circles))

            for i in r_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += mask_red[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(mask_red, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(img, 'RED', (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    state.update({"state": 0})
                    img[0:100, 0:100] = overlayList[3]
                    stop() #calling stop function
                    print("Red light detect")

        if g_circles is not None:
            g_circles = np.uint16(np.around(g_circles))

            for i in g_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += mask_green[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(mask_green, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(img, 'Green', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    #print("Green light detect")


def laneDetect():
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    contours, hierarchy = cv2.findContours(mask_white, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0 and state == {'state': 1}:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # print("CX:"+str(cx)+"CY:"+str(cy))
            if cx >= 360:
                print("Turn Right")
                img[0:100, 0:100] = overlayList[2]
                move_right() #calling turn right function

            if cx < 360 and cx > 240:
                print("On Track")
                img[0:100, 0:100] = overlayList[0]
                move_forward() #calling move forward function

            if cx <= 240:
                print("Turn Left")
                img[0:100, 0:100] = overlayList[1]
                move_left() #calling turn left function

            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.drawContours(img, c, -1, (0, 255, 0), 1)
        cv2.imshow("Black_Mask", mask_white)


if __name__ == '__main__':
    cam = cv2.VideoCapture(1)

    #######################
    wCam = 480  # 480
    hCam = 360  # 360
    #######################
    cam.set(3, wCam)
    cam.set(4, hCam)

    cTime = 0
    pTime = 0

    while True:
        success, img = cam.read()

        #### Add Function ####
        stopSign()
        trafficSign()
        laneDetect()
        #####################

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {str(int(fps))}', (120, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        state.update({"state": 1})
        cv2.imshow("OUTPUT", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ArduinoSerial.write(b'S')
            break
