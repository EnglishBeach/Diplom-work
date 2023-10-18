import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract as tes
from matplotlib.widgets import  Slider

cap = cv2.VideoCapture("Video_process/Videos/Full_font1.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()



# The parametrized function to be plotted
def apply_thresh(thresh):
    frame_2color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh, frame_bw = cv2.threshold(
        frame_2color,
        thresh,
        255,
        cv2.THRESH_BINARY,
    )
    return frame_bw


# Define initial parameters
thresh = 0

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
frame = apply_thresh(thresh)
image_plot = ax.imshow(frame)
# image_plot.set_label(f'Recognize: {tes.image_to_string(frame)}')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(top=1, hspace=0, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.2, 0.1, 0.6, 0.03])
thresh_slider = Slider(
    ax=axfreq,
    label='Thresh',
    valmin=0,
    valmax=256,
    valinit=thresh,
)


def update(val):
    thresh= thresh_slider.val
    frame = apply_thresh(thresh)

    image_plot.autoscale()
    image_plot.set_data(frame)
    # image_plot.set_label(f'Recognize: {tes.image_to_string(frame)}')

    fig.canvas.draw_idle()


thresh_slider.on_changed(update)

plt.show()