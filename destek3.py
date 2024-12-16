import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("stop2.jpg")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


min_red1 = np.array([0, 70, 50])   
max_red1 = np.array([10, 255, 255])
min_red2 = np.array([170, 70, 50]) 
max_red2 = np.array([180, 255, 255])
min_red3 = np.array([140, 70, 40]) 
max_red3 = np.array([170, 200, 160])

mask1 = cv2.inRange(hsv_image, min_red1, max_red1)
mask2 = cv2.inRange(hsv_image, min_red2, max_red2)
mask3 = cv2.inRange(hsv_image, min_red3, max_red3)
mask = mask1 + mask2 + mask3


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 7000:  
        x, y, z, h = cv2.boundingRect(contour) 
        cv2.rectangle(image, (x, y), (x + z, y + h), (0,255,0), 5) 


center_x = x + z // 2
center_y = y + h // 2
print(f"Karenin merkezi: ({center_x}, {center_y})")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.imshow(image_rgb)
plt.title("Stop Tabelası")
plt.axis("off")
plt.show()
