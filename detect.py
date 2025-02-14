#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import argparse
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class GenderAgeDetector:
    def __init__(self, window):
        self.window = window
        self.window.title("Gender and Age Detection")
        self.window.geometry("800x600")

        # Initialize models with corrected paths
        self.faceProto = "opencv_face_detector.pbtxt"
        self.faceModel = "opencv_face_detector_uint8.pb"
        self.ageProto = "age_deploy.prototxt"
        self.ageModel = "age_net.caffemodel"
        self.genderProto = "gender_deploy.prototxt"
        self.genderModel = "gender_net.caffemodel"

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']
        
        # Load networks
        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)

        self.video = None
        self.padding = 20

        # Create UI elements
        self.create_ui()

    def create_ui(self):
        # Create buttons frame
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)

        # Create buttons
        ttk.Button(button_frame, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_video).pack(side=tk.LEFT, padx=5)

        # Create video frame
        self.video_label = ttk.Label(self.window)
        self.video_label.pack(pady=10)

        # Create status frame
        self.status_label = ttk.Label(self.window, text="Status: Ready")
        self.status_label.pack(pady=10)

    def highlightFace(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn,faceBoxes

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Video files", "*.mp4 *.avi *.mov"),
            ("All files", "*.*")
        ])
        if file_path:
            self.stop_video()
            self.video = cv2.VideoCapture(file_path)
            self.process_video()

    def start_webcam(self):
        self.stop_video()
        self.video = cv2.VideoCapture(0)
        self.process_video()

    def stop_video(self):
        if self.video is not None:
            self.video.release()
            self.video = None
            self.video_label.configure(image='')
            self.status_label.configure(text="Status: Stopped")

    def process_video(self):
        if self.video is None:
            return

        hasFrame, frame = self.video.read()
        if not hasFrame:
            self.stop_video()
            return

        resultImg, faceBoxes = self.highlightFace(self.faceNet, frame)
        
        if not faceBoxes:
            self.status_label.configure(text="Status: No face detected")
        else:
            for faceBox in faceBoxes:
                face = frame[max(0,faceBox[1]-self.padding):
                           min(faceBox[3]+self.padding,frame.shape[0]-1),max(0,faceBox[0]-self.padding)
                           :min(faceBox[2]+self.padding, frame.shape[1]-1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), self.MODEL_MEAN_VALUES, swapRB=False)
                
                self.genderNet.setInput(blob)
                genderPreds = self.genderNet.forward()
                gender = self.genderList[genderPreds[0].argmax()]

                self.ageNet.setInput(blob)
                agePreds = self.ageNet.forward()
                age = self.ageList[agePreds[0].argmax()]

                cv2.putText(resultImg, f'{gender}, {age}', 
                          (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, (0,255,255), 2, cv2.LINE_AA)
                
                self.status_label.configure(text=f"Detected: {gender}, Age: {age}")

        # Convert to PIL format and display
        rgb_image = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Resize image to fit window while maintaining aspect ratio
        display_size = (700, 500)
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image=pil_image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

        self.window.after(10, self.process_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeDetector(root)
    root.mainloop()
