import os
from os import listdir
from PIL import Image


def remove_corrupt_and_nonjpg_images(dir_path):
    """
     Removes corrupt & non jpg images from the specified dir path for raw data
    :param dir_path: path to directory with raw images
    """
    cntr = 0
    for filename in listdir(dir_path):
        cntr += 1
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            try:
                img = Image.open(dir_path + "/" + filename)  # open the image file
                img.verify()  # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print(e)
                print("Deleting file.")
                os.remove(dir_path + "/" + filename)
        else:
            print("Deleting non .jpg file")
            os.remove(dir_path + "/" + filename)
    print(f"--> Scanned {cntr} files within {dir_path}")


if __name__ == "__main__":
    SUPPORTED_LANGUAGES = ["telugu", "english", "russian"]
    for data in SUPPORTED_LANGUAGES:
        remove_corrupt_and_nonjpg_images(dir_path="raw/data_to_transform/" + data)
    print(":: All done!")
