import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image using OpenCV
img_path = r"C:/Documents_LOCAL/EPFL/MA2/Aerial_robotics/micro-502/controllers/main/image_analysis/gate_features_multiple_contours1.png"
img_bgr = cv2.imread(img_path) #openCV loads images in BGR format
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots()
im_display = ax.imshow(img_rgb)
ax.set_title("Scroll to Zoom | Drag to Pan")

def format_coord(x, y):
    if 0 <= int(x) < img_rgb.shape[1] and 0 <= int(y) < img_rgb.shape[0]:
        r, g, b = img_rgb[int(y), int(x)]
        return f"x={int(x)}, y={int(y)}, R={r}, G={g}, B={b}"
    else:
        return f"x={int(x)}, y={int(y)}"

ax.format_coord = format_coord

# ---------- Zooming ----------
def on_scroll(event):
    base_scale = 1.5
    xdata = event.xdata
    ydata = event.ydata

    if xdata is None or ydata is None:
        return

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    if event.button == 'up':
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        scale_factor = base_scale
    else:
        scale_factor = 1

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
    rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
    ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1 - rely)])
    ax.figure.canvas.draw()

# ---------- Panning with Drag ----------
class PanHandler:
    def __init__(self):
        self.press = None
        self.dragging = False

    def on_press(self, event):
        if event.inaxes and event.button == 1:
            self.press = event.x, event.y, ax.get_xlim(), ax.get_ylim()

    def on_release(self, event):
        self.press = None
        self.dragging = False
        fig.canvas.draw()

    def on_motion(self, event):
        if self.press is None or event.x is None or event.y is None:
            return
        self.dragging = True
        xpress, ypress, xlim, ylim = self.press
        dx = event.x - xpress
        dy = event.y - ypress
        scale_x = (xlim[1] - xlim[0]) / ax.bbox.width
        scale_y = (ylim[1] - ylim[0]) / ax.bbox.height

        ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
        ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
        fig.canvas.draw()

pan_handler = PanHandler()

fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('button_press_event', pan_handler.on_press)
fig.canvas.mpl_connect('button_release_event', pan_handler.on_release)
fig.canvas.mpl_connect('motion_notify_event', pan_handler.on_motion)

plt.show()
