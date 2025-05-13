from dslr_gphoto import *

cam = camera_init()

path, img = capture_image(cam)
print(path)
print(img.__len__())
import matplotlib.pyplot as plt
plt.imshow(img)
