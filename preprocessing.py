from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


pil_image1 = cv2.imread("sample stabilization test projections/fiducials/automatic acquisition/projections_Fibres_plasticfiller _16-02_15-13-32/Fibres_plasticfiller _Cu_L2160_B2_E100.tiff",
                       cv2.IMREAD_ANYDEPTH)
pil_image2 = cv2.imread("sample stabilization test projections/fiducials/automatic acquisition/projections_Fibres_plasticfiller _16-02_15-13-32/Fibres_plasticfiller _Cu_L2160_B2_E100_01.tiff",
                       cv2.IMREAD_ANYDEPTH)
image1_array = np.array(pil_image1)
image2_array = np.array(pil_image2)
flow = cv2.calcOpticalFlowFarneback(pil_image1, pil_image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

hsv = np.zeros((image1_array.shape[0], image1_array.shape[1], 3), dtype=np.uint8)
hsv[..., 1] = 255

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
value = ang * 180 / np.pi / 2
hsv[..., 0] = value
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print(bgr)
print(bgr.shape)

cv2.imshow("colored flow", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(image_array.shape)
#print(image_array)

#plt.imshow(pil_image, cmap='gray')
#plt.show()
