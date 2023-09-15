import cv2
import numpy as np
from imread_from_url import imread_from_url

from hitnet import HitNet, ModelType, CameraConfig
import time
# Select model type
# model_type = ModelType.middlebury
# model_type = ModelType.flyingthings
model_type = ModelType.middlebury

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400/saved_model_720x1280/model_float32.onnx"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d/saved_model_720x1280/model_float32.onnx"

# Initialize model
depth_estimator = HitNet(model_path, model_type)

start_time = time.time()
left_img = cv2.imread("im2.png")
right_img = cv2.imread('im6.png')
disparity_map = depth_estimator(left_img, right_img)

color_disparity = depth_estimator.draw_disparity()
print (time.time() - start_time)
combined_image = np.hstack((left_img, color_disparity))
gray = cv2.cvtColor(color_disparity, cv2.COLOR_BGR2GRAY)

cv2.imwrite("out.jpg", combined_image)

cv2.imshow("Estimated disparity", combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
