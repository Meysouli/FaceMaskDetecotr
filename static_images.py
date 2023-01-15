# Created by Oumaima Souli.

import numpy as np
from keras.models import load_model
import keras.utils as image

# Load trained model
model = load_model('model.h5')

print("================ Testing provided images ================")

# Test the 1st individual image
without_mask_image = image.load_img(r'1.jpeg',
                            target_size=(150, 150, 3))
without_mask_image
without_mask_image = image.img_to_array(without_mask_image)
without_mask_image = np.expand_dims(without_mask_image, axis=0)

without_mask_prediction = model.predict(without_mask_image)[0][0]
if without_mask_prediction == 1:
    print("the person in the provided image 1.jpeg isn't wearing a mask")
else:
    print("the person in the provided image 1.jpeg is wearing a mask")

# Test the 2nd individual image
with_mask = image.load_img(r'2.jpeg',
                            target_size=(150, 150, 3))
with_mask
with_mask = image.img_to_array(with_mask)
with_mask = np.expand_dims(with_mask, axis=0)
with_mask_prediction = model.predict(with_mask)[0][0]
if with_mask_prediction == 1:
    print("the person in the provided image 2.jpeg isn't wearing a mask")
else:
    print("the person in the provided image 2.jpeg is wearing a mask")
