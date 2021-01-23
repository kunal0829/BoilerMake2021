from tkinter import *
import tkinter as tk
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import streamlink
import cv2

def cam1():
    streams = streamlink.streams('https://www.earthcam.com/usa/florida/keywest/?cam=irishkevins')
    url = streams['450p'].to_url()

    retry = True
    while retry:
        try:
            cap = cv2.VideoCapture(url)
            retry = False
        except Exception as e:
            print(e)
            retry = True
    fps = 20
    frame_time = int((1.0 / fps) * 1000.0)

    while True:
        try:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (416, 416))
                cv2.imshow('frame', frame)
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break
            else:
                pass
        except Exception as e:
            print(e)
            break

    cv2.destroyAllWindows()
    cap.release()
    
def cam2():
    streams = streamlink.streams('https://www.earthcam.com/usa/florida/keywest/?cam=irishkevins')
    url = streams['450p'].to_url()

    retry = True
    while retry:
        try:
            cap = cv2.VideoCapture(url)
            retry = False
        except Exception as e:
            print(e)
            retry = True
    fps = 20
    frame_time = int((1.0 / fps) * 1000.0)

    while True:
        try:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (416, 416))
                cv2.imshow('frame', frame)
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break
            else:
                pass
        except Exception as e:
            print(e)
            break

    cv2.destroyAllWindows()
    cap.release()
    
def cam3():
    streams = streamlink.streams('https://www.earthcam.com/usa/florida/keywest/?cam=irishkevins')
    url = streams['450p'].to_url()

    retry = True
    while retry:
        try:
            cap = cv2.VideoCapture(url)
            retry = False
        except Exception as e:
            print(e)
            retry = True
    fps = 20
    frame_time = int((1.0 / fps) * 1000.0)

    while True:
        try:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (416, 416))
                cv2.imshow('frame', frame)
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break
            else:
                pass
        except Exception as e:
            print(e)
            break

    cv2.destroyAllWindows()
    cap.release()

root = tk.Tk()
root.title('Purdue')
root.geometry("1097x802")
#picture = PhotoImage(file="Purdue.png")
img = Image.open("Purdue.png")
img = img.resize((1097,802), Image.ANTIALIAS)
picture =  ImageTk.PhotoImage(img)


label0 = Label(root, image=picture, borderwidth=0, highlightthickness=10)
label0.place(x=0, y=0)

#star = tk.PhotoImage(file="star.png")
img = Image.open("circle.png")
img = img.resize((25,25), Image.ANTIALIAS)
star =  ImageTk.PhotoImage(img)



camera1 = Button(root, image = star, command = cam1)
camera1.place(x = 675,y = 600)

camera2 = Button(root, image = star, command = cam2)
camera2.place(x = 450,y = 425)

camera3 = Button(root, image = star, command = cam3)
camera3.place(x = 690,y = 75)

root.mainloop()
