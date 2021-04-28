# Imports
import time
from tkinter import *
import cv2
from PIL import Image, ImageTk
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import pose_module as pm
import numpy as np


class App:
    def __init__(self, video_source=0):
        self.player = QMediaPlayer()  # Initialize player for click sound
        self.app_name = "AI Trainer"
        self.window = Tk()
        self.window.title(self.app_name)
        self.window.resizable(0, 0)
        self.window.wm_iconbitmap("gym.ico")
        self.window["bg"] = "#FBC088"
        self.video_source = video_source
        self.photo = None

        self.vid = MyVideoCapture(self.video_source)
        self.label = Label(self.window, text=self.app_name, font=20, bg="gray", fg="white").pack(side=TOP, fill=BOTH)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(self.window, width=1280, height=720)
        self.canvas.pack()

        # Button which takes a snapshot
        self.btn_snapshot = Button(self.window, text="Save", font="lucida 15 bold", width=30, bg="#8FB48E",
                                   activebackground="#564BF7",
                                   command=self.snapshot)
        self.btn_snapshot.pack(anchor=CENTER, expand=True)
        self.update()
        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        check, frame = self.vid.getFrame()
        if check:
            image = "IMG-" + time.strftime("%H-%M-%S-%d-%m") + ".jpg"
            cv2.imwrite(image, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Show the message on window that image was saved
            # msg = Label(self.window, text="Image Saved: " + image, font="lucida 10 bold", bg="#EBC16A", fg="#36454F")
            # msg.place(x=710, y=900)
            # Sound
            file = QUrl("click.wav")
            content = QMediaContent(file)
            self.player.setMedia(content)
            self.player.play()

    def update(self):
        # Get a frame from the video source
        isTrue, frame = self.vid.getFrame()
        if isTrue:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        self.window.after(15, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open Camera")

        # Get video source width and height
        # self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.detector = pm.poseDetector()
        self.count = 0
        self.dir = 0

    def getFrame(self):
        if self.vid.isOpened():
            # 1. Import image
            success, img = self.vid.read()
            img = cv2.resize(img, (1280, 720))

            img = self.detector.findPose(img, False)
            lmList = self.detector.findPosition(img, False)
            # print(lmList)
            if len(lmList) != 0:
                # Right Arm
                angle = self.detector.findAngle(img, 12, 14, 16)
                # # Left Arm
                # angle = detector.findAngle(img, 11, 13, 15,False)
                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (650, 100))
                # print(angle, per)

                # Check for the dumbbell curls
                color = (255, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if self.dir == 0:
                        self.count += 0.5
                        self.dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if self.dir == 1:
                        self.count += 0.5
                        self.dir = 0
                print(self.count)

                # Draw Bar
                cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (1050, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                            color, 4)

                # Draw Curl Count
                cv2.rectangle(img, (0, 450), (350, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(self.count)), (30, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                            (255, 0, 0), 25)

            if success:
                # If isTrue is true then current frame converted to RGB
                return success, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                return success, None

    def __del__(self):
        self.vid.release()


if __name__ == "__main__":
    App(0)
