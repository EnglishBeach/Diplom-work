## Imports
print("Importing...")
import re
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
import easyocr

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider

########################################################
##########               INPUTS               ##########
########################################################

VIDEO_PATH = "Video_process/Videos/Exp0/Exp0_1.avi"
# VIDEO_PATH = None
rules = dict(re_rule=r'-?\d{1,3}\.\d', )
RECOGNIZABLE_VARIABLES = [
    dict(name='Viscosity', rules=rules),
    dict(name='Temperature', rules=rules),
]

## Global settings
if VIDEO_PATH is None:
    input_path = ''
    while input_path == '':
        input_path = input(f"Input video path: ")
    VIDEO_PATH = input_path

CAP = cv2.VideoCapture(VIDEO_PATH)
FPS = int(CAP.get(cv2.CAP_PROP_FPS))
LENTH = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT) / FPS)
CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, START_FRAME = CAP.read()

def foo():
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(left=0.25, right=1, bottom=0.25, top=1, hspace=0, wspace=0)

    image_processor = lambda x:x
    PLOT = ax.imshow(image_processor(START_FRAME), cmap='binary')

    ax_time_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    TIME_slider = Slider(
        ax=ax_time_slider,
        label='Time',
        valmin=0,
        valmax=LENTH,
        valinit=0,
        valstep=1,
    )

    ax_blur_slider = fig.add_axes([0.1, 0.25, 0.03, 0.6])
    BLUR_slider = Slider(
        ax=ax_blur_slider,
        orientation='vertical',
        label='Blur',
        valmin=1,
        valmax=50,
        valinit=1,
        valstep=1,
    )


    def update(val):
        time = TIME_slider.val
        global image_processor
        print(BLUR_slider.val)

        CAP.set(cv2.CAP_PROP_POS_FRAMES, int(FPS * time))
        _, frame = CAP.read()
        frame = image_processor(frame)

        PLOT.set_data(frame)
        PLOT.autoscale()

        fig.canvas.draw_idle()


    TIME_slider.on_changed(update)
    BLUR_slider.on_changed(update)
    print('Configurate image processing')
    plt.show()

foo()
plt.plot([1,23,4,56,4,5,6,77,5])
plt.show()