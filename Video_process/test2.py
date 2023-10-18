import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pytesseract as tes
from matplotlib.widgets import Button, Slider

cap = cv2.VideoCapture("Video_process/Videos/Full_font1.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()

# The parametrized function to be plotted
def apply_thresh(thresh):
    frame_2color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh, frame_bw = cv2.threshold(frame_2color, thresh, 255, cv2.THRESH_BINARY)
    return frame_bw

# Define initial parameters
thresh = 5

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
image_plot = ax.imshow(apply_thresh(thresh))

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
thresh_slider = Slider(
    ax=axfreq,
    label='Thresh',
    valmin=0,
    valmax=500,
    valinit=thresh,
)

# Make a vertically oriented slider to control the amplitude
# axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
# amp_slider = Slider(
#     ax=axamp,
#     label="Amplitude",
#     valmin=0,
#     valmax=10,
#     valinit=init_amplitude,
#     orientation="vertical"
# )


# The function to be called anytime a slider's value changes
def update(val):
    thresh=thresh_slider.val
    image_plot.autoscale()
    frame = apply_thresh(thresh)
    image_plot.set_data()
    fig.canvas.draw_idle()


# register the update function with each slider
thresh_slider.on_changed(update)
# amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
# resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    thresh_slider.reset()
# button.on_clicked(reset)

plt.show()