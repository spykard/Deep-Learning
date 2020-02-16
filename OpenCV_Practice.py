import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an color image in grayscale
img = cv2.imread('cat.2.jpg',1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# or for a specific key
#while(1):
#    cv2.imshow('image',img)
#    if cv2.waitKey(20) & 0xFF == 27:
#        break
#cv2.destroyAllWindows()

# Import image
# OpenCV uses BGR, Matplotlib uses RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()