import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
import pickle

# Load color classification model and label encoder
color_model = load_model("Car_Color_Detection.h5")
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load YOLO model
yolo = YOLO('yolov8n.pt')

# GUI setup
top = tk.Tk()
top.geometry('900x700')
top.title('Car Color Detector')
top.configure(background='#CDCDCD')

label_info = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
sign_image = Label(top)

def Detect(file_path):
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo(image_rgb)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = yolo.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_name == "car":
            pad = 10
            h, w, _ = image.shape
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            car_crop = image[y1p:y2p, x1p:x2p]
            try:
                car_rgb = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
                car_resized = cv2.resize(car_rgb, (128, 128)) / 255.0
                car_input = np.expand_dims(car_resized, axis=0)

                pred = color_model.predict(car_input, verbose=0)
                pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]

                print(f"Predicted color: {pred_label}")
                color = (0, 0, 255) if pred_label.lower() == 'blue' else (255, 0, 0)

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{pred_label} Car", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print("Color prediction failed:", e)

        elif class_name == "person":
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_display = Image.fromarray(image_display)
    image_display.thumbnail((600, 400))
    imgtk = ImageTk.PhotoImage(image_display)

    sign_image.configure(image=imgtk)
    sign_image.image = imgtk
    label_info.configure(text="Detection Complete. See Colors & People Highlighted.")

def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Colors", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.8, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label_info.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print("Error:", e)

# GUI layout
upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=20)

sign_image.pack(side='bottom', expand=True)
label_info.pack(side="bottom", expand=True)

heading = Label(top, text="Car Color Detection + Person Identifier", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
