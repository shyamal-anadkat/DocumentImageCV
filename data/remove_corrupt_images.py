import os
from os import listdir
from PIL import Image


def remove_corrupt_images(dir_path):
    for filename in listdir(dir_path):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(dir_path + '/' + filename)  # open the image file
                img.verify()  # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print(e)
                print('Deleting file.')
                os.remove(dir_path + '/' + filename)


for data in ['telugu', 'english', 'russian']:
    remove_corrupt_images(dir_path='raw/data_to_transform/' + data)

print('All done...')
