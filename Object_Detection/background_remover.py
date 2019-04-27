import cv2
import numpy as np
from matplotlib import pyplot as plt

# Parameters are set before hand.
BLUR = 21
CANNY_THRESH_1 = 5  #10
CANNY_THRESH_2 = 150 #200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 12 #10
MASK_COLOR = (0.0,0.0,1.0) # this is in BGR format


# PROCESSING STARTS HERE
# Read the image at PATH:/....
img = cv2.imread('Source Images/AdidasUB2.jpg')
# img = cv2.imread('Source Images/NikeAJ1Retro.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Here edge detection starts with first applying a gray colour to the edges that are detected
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)

# Using erode and dilate helps use sharpen the edges that are detected
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

# Find contours in edges, and sort by area. Stored in an array
contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
# sort the areas found in contours.
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

# Create empty mask, draw filled polygon on it corresponding to largest contour
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

# Smooth the mask and then blur it out
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

# Blend masked img into MASK_COLOR background

#  Use float matrices,
mask_stack = mask_stack.astype('float32') / 255.0

#  for easy blending
img = img.astype('float32') / 255.0

# Blend
masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)

# Convert back to 8-bit
masked = (masked * 255).astype('uint8')

# split image into channels
c_red, c_green, c_blue = cv2.split(img)

# merge with mask got on one of a previous steps
img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

# show on screen (optional in jupiter)
#matplotlib inline
plt.imshow(img_a)
plt.show()

# save to disk
cv2.imwrite('Background Removed Images/BGR_IMG1.png', masked)
