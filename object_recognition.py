from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np
from os import listdir


def preprocess(img_dir, img_file):
    img = np.array(Image.open(img_dir + '/' + img_file))
    img = img.astype(np.float32) - np.mean(img)
    img /= np.std(img)
    img = np.expand_dims(img, axis=0)
    return img

ip = tflite.Interpreter(model_path='object_recognition.tflite')
ip.allocate_tensors()


input_index = ip.get_input_details()[0]['index']
output_index = ip.get_output_details()[0]['index']

img_dir = 'test_imgs'
num_correct = 0
for img_file in listdir(img_dir):
    

    img = preprocess(img_dir, img_file)
    ip.set_tensor(input_index, img)


    ip.invoke()
    preds = ip.get_tensor(output_index)

    label = int(img_file.split('_')[0])
    if np.argmax(preds) == label:
        num_correct += 1


num_imgs = len(listdir(img_dir))
print('{} correct out of {}'.format(num_correct, num_imgs))
