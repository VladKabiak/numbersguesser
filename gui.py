import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

model = load_model('mnist_model.keras')

root = Tk()
root.title('Digit Recognizer')

cv = Canvas(root, width=300, height=300, bg='black')
cv.grid(row=0, column=0, pady=2, sticky=W)

img = Image.new("RGB", (300, 300), (0, 0, 0))
draw = ImageDraw.Draw(img)


def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_rectangle(x1, y1, x2, y2, fill='white', outline='white', width=5)
    draw.rectangle([x1, y1, x2, y2], fill='white', width=5)
    cv.update()


def clear_canvas():
    cv.delete("all")
    draw.rectangle([0, 0, 300, 300], fill='black')


def predict_digit():
    img_gray = img.convert('L')
    img_resized = img_gray.resize((28, 28))
    img_array = np.array(img_resized)
    img_array = img_array.reshape(1, 28, 28) / 255.0

    probabilities = model.predict(img_array)
    prediction = np.argmax(probabilities)
    label.config(text=f'Predicted Digit: {prediction}')


cv.bind('<B1-Motion>', paint)

btn_predict = Button(root, text='Predict', command=predict_digit)
btn_predict.grid(row=1, column=0, pady=2)

btn_clear = Button(root, text='Clear', command=clear_canvas)
btn_clear.grid(row=2, column=0, pady=2)

label = Label(root, text='', font=('Helvetica', 18))
label.grid(row=3, column=0, pady=2)

root.mainloop()
