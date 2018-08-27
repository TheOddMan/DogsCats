import os
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

path = "test"
model = load_model('first_try.h5')
classes=['cat','dog']                                                                       #更改處

files = os.listdir(path)
accuracy = 0
totalcount = 0
for f in files :
    totalcount += 1

    img = image.load_img(path +"\\"+ f, target_size=(150, 150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    result = model.predict(images)
    ind = np.argmax(result, 1)

    filebasename = f.split(".")[0]
    if (filebasename == classes[ind[0]]):
        accuracy += 1
    print('this ' + f + ' is a ', classes[ind[0]])

print("Test accuracy : ",round((accuracy/totalcount),2))

