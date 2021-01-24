from tkinter import *
import tkinter as tk
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import streamlink
import cv2
from yolov5.detect import detect
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import plot_one_box
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    print("Matrix = ", M)
    # or use this function
    # M = cv2.findHomography(pts, dst)[0]

    print("angle of rotation: {}".format(np.arctan2(-M[1, 0], M[0, 0]) * 180 / np.pi))

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def cam1():
    streams = streamlink.streams('https://www.earthcam.com/cams/newyork/timessquare/?cam=tsrobo1')
    source = streams['720p'].to_url()
    device = select_device('')
    face_model = attempt_load('maskdetect.pt', map_location = device)
    human_model = attempt_load('yolov5l.pt', map_location = device)
    cap = cv2.VideoCapture(source)
    imgsz = 480
    iou_thresh = 0.35
    conf_thresh = 0.25

    nframes = 0
    while True:
        ret, frame = cap.read()
        if ret:
            nframes += 1
            faces = detect(face_model, device, frame, imgsz, iou_thresh+.5, conf_thresh+.5)
            objects = detect(human_model, device, frame, imgsz, iou_thresh, conf_thresh)
            humans = [xyxy for xyxy, label in objects if label == 'person']

            if humans:
                for xyxy, label in faces:
                    if label == 'good':
                        plot_one_box(xyxy, frame, label='', color=[0, 255, 0], line_thickness=2)
                    else:
                        plot_one_box(xyxy, frame, label='', color=[0, 0, 255], line_thickness=2)

                avg_pixel_height = sum([abs(xyxy[1] - xyxy[3]) for xyxy in humans]) / len(humans)
                avg_human_height = 68

                total_dist = 0
                counter = 0

                num_safe = 0

                for curr in range(len(humans)):
                    curr_human = humans[curr]
                    curr_center_x, curr_center_y = (curr_human[0] + curr_human[2]) / 2, (curr_human[1] + curr_human[3]) / 2

                    safe = True
                    min_dist = 1000000
                    for other in range(len(humans)):
                        if other != curr:
                            other_human = humans[other]
                            other_center_x, other_center_y = (other_human[0] + other_human[2]) / 2, (other_human[1] + other_human[3]) / 2

                            pix_dist = distance(curr_center_x, curr_center_y, other_center_x, other_center_y)
                            actual_dist = pix_dist / avg_pixel_height * 68

                            if actual_dist < min_dist:
                                min_dist = actual_dist

                            if actual_dist < 72: #closer tahn 6 ft
                                safe = False

                    total_dist += min_dist
                    counter += 1

                    if safe:
                        num_safe += 1
                        plot_one_box(curr_human, frame, label='', color=[0, 255, 0], line_thickness=2)
                    else:
                        plot_one_box(curr_human, frame, label='', color=[0, 0, 255], line_thickness=2)

                average_dist = int(total_dist / counter)
                avg_ft = average_dist // 12
                avg_in = average_dist % 12

                frame = cv2.putText(frame, "Average distance between people is {} feet and {} inches".format(avg_ft, avg_in), (200, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2, cv2.LINE_AA)
                frame = cv2.putText(frame, "{}% of people are following proper social distancing guidelines".format(round(num_safe / len(humans) * 100), 1), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('img', frame)
            cv2.waitKey(1)

            if nframes >= 150:
                break


    cap.release()
    cv2.destroyAllWindows()

def cam2():
    streams = streamlink.streams('https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=catsmeow2')
    source = streams['720p'].to_url()
    device = select_device('')
    face_model = attempt_load('maskdetect.pt', map_location = device)
    human_model = attempt_load('yolov5l.pt', map_location = device)
    cap = cv2.VideoCapture(source)
    imgsz = 720
    iou_thresh = 0.35
    conf_thresh = 0.45

    nframes = 0

    while True:
        ret, frame = cap.read()
        if ret:
            nframes += 1
            faces = detect(face_model, device, frame, imgsz, iou_thresh+.5, conf_thresh+.5)
            objects = detect(human_model, device, frame, imgsz, iou_thresh, conf_thresh)
            humans = [xyxy for xyxy, label in objects if label == 'person']

            if humans:
                avg_pixel_height = sum([abs(xyxy[1] - xyxy[3]) for xyxy in humans]) / len(humans)
                avg_human_height = 68

                good_mask = 0
                for xyxy, label in faces:
                    if label == 'good':
                        good_mask += 1
                        plot_one_box(xyxy, frame, label='', color=[0, 255, 0], line_thickness=2)
                    else:
                        plot_one_box(xyxy, frame, label='', color=[0, 0, 255], line_thickness=2)

                for curr in range(len(humans)):
                    curr_human = humans[curr]
                    curr_center_x, curr_center_y = (curr_human[0] + curr_human[2]) / 2, (curr_human[1] + curr_human[3]) / 2

                    safe = True

                    for other in range(len(humans)):
                        if other != curr:
                            other_human = humans[other]
                            other_center_x, other_center_y = (other_human[0] + other_human[2]) / 2, (other_human[1] + other_human[3]) / 2

                            pix_dist = distance(curr_center_x, curr_center_y, other_center_x, other_center_y)
                            actual_dist = pix_dist / avg_pixel_height * 68

                            if actual_dist < 72: #closer tahn 6 ft
                                safe = False
                                break

                    if safe:
                        plot_one_box(curr_human, frame, label='', color=[0, 255, 0], line_thickness=2)
                    else:
                        plot_one_box(curr_human, frame, label='', color=[0, 0, 255], line_thickness=2)

                if faces:
                    frame = cv2.putText(frame, "{}% of people are wearing a mask".format(round(good_mask / len(faces) * 100, 1)), (200, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('img', frame)
            cv2.waitKey(1)

            if nframes >= 150:
                break

    cap.release()
    cv2.destroyAllWindows()

def cam3():
    streams = streamlink.streams('https://www.earthcam.com/usa/florida/keywest/?cam=irishkevins')
    source = streams['450p'].to_url()

    device = select_device('')
    face_model = attempt_load('maskdetect.pt', map_location = device)
    human_model = attempt_load('yolov5l.pt', map_location = device)
    cap = cv2.VideoCapture(source)
    imgsz = 480
    iou_thresh = 0.35
    conf_thresh = 0.25

    nframes = 0
    while True:
        ret, frame = cap.read()
        if ret:
            nframes += 1
            faces = detect(face_model, device, frame, imgsz, iou_thresh+.5, conf_thresh+.5)
            objects = detect(human_model, device, frame, imgsz, iou_thresh, conf_thresh)
            humans = [xyxy for xyxy, label in objects if label == 'person']

            if humans:
                avg_pixel_height = sum([abs(xyxy[1] - xyxy[3]) for xyxy in humans]) / len(humans)
                avg_human_height = 68

                for xyxy, label in faces:
                    if label == 'good':
                        plot_one_box(xyxy, frame, label='', color=[0, 255, 0], line_thickness=2)
                    else:
                        plot_one_box(xyxy, frame, label='', color=[0, 0, 255], line_thickness=2)

                total_dist = 0
                counter = 0
                num_safe = 0

                for curr in range(len(humans)):
                    curr_human = humans[curr]
                    curr_center_x, curr_center_y = (curr_human[0] + curr_human[2]) / 2, (curr_human[1] + curr_human[3]) / 2

                    safe = True

                    min_dist = 100000
                    for other in range(len(humans)):
                        if other != curr:
                            other_human = humans[other]
                            other_center_x, other_center_y = (other_human[0] + other_human[2]) / 2, (other_human[1] + other_human[3]) / 2

                            pix_dist = distance(curr_center_x, curr_center_y, other_center_x, other_center_y)
                            actual_dist = pix_dist / avg_pixel_height * 68

                            if actual_dist < min_dist:
                                min_dist = actual_dist

                            if actual_dist < 72: #closer tahn 6 ft
                                safe = False
                                break

                    total_dist += min_dist
                    counter += 1

                    if safe:
                        num_safe += 1
                        plot_one_box(curr_human, frame, label='', color=[0, 255, 0], line_thickness=2)
                    else:
                        plot_one_box(curr_human, frame, label='', color=[0, 0, 255], line_thickness=2)

                average_dist = int(total_dist / counter)
                avg_ft = average_dist // 12
                avg_in = average_dist % 12

                frame = cv2.putText(frame, "Average distance between people is {} feet and {} inches".format(avg_ft, avg_in), (200, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1, cv2.LINE_AA)
                frame = cv2.putText(frame, "{}% of people are following proper social distancing guidelines".format(round(num_safe / len(humans) * 100), 1), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('img', frame)
            cv2.waitKey(1)

            if nframes >= 100:
                break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title('Purdue')
root.geometry("1097x802")
#picture = PhotoImage(file="Purdue.png")
img = Image.open("Images/PurdueWithRegions.png")
img = img.resize((1097,802), Image.ANTIALIAS)
picture =  ImageTk.PhotoImage(img)


label0 = Label(root, image=picture, borderwidth=0, highlightthickness=10)
label0.place(x=0, y=0)

#star = tk.PhotoImage(file="star.png")
cir = Image.open("Images/blue.png")
cir = cir.resize((25,25), Image.ANTIALIAS)
circle =  ImageTk.PhotoImage(cir)



camera1 = Button(root, image = circle, command = cam1)
camera1.place(x = 675,y = 600)



camera2 = Button(root, image = circle, command = cam2)
camera2.place(x = 450,y = 425)

camera3 = Button(root, image = circle, command = cam3)
camera3.place(x = 690,y = 75)



root.mainloop()
