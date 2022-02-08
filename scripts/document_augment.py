import os
from pathlib import Path

import cv2


def augment_document_image(img_dir, output_dir_name):
    """
    Augments/sharpens document image
    Source: https://dontrepeatyourself.org/post/learn-opencv-by-building-a-document-scanner/
    :param img_dir:
    :param output_dir_name:
    """
    valid_formats = [".jpg", ".jpeg"]
    get_text = lambda f: os.path.splitext(f)[1].lower()

    img_files = [
        img_dir + f for f in os.listdir(img_dir) if get_text(f) in valid_formats
    ]
    # create a new folder that will contain our images
    Path("../data/raw/data_to_transform/" + output_dir_name).mkdir(exist_ok=True)

    errors = []
    # go through each image file
    for img_file in img_files:
        # read, resize, and make a copy of the image
        try:
            img = cv2.imread(img_file)

            # img = cv2.resize(img, (width, height))
            # height, width, channels = img.shape
            # orig_img = img.copy()

            # preprocess the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # sharpen image
            sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
            sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

            # apply adaptive threshold to get black and white effect
            final_img = cv2.adaptiveThreshold(
                sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
            )

            # to be done: find and sort the contours
            # To debug and show orig vs scanned
            # cv2.imshow("Scanned", final_img)
            # cv2.imshow("Original", orig_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # write the image in the ouput directory
            cv2.imwrite(
                "../data/raw/data_to_transform/"
                + output_dir_name
                + "/"
                + os.path.basename(img_file),
                final_img,
            )
        except Exception:  # pylint: disable=broad-except
            errors.append(img_file)
            # print('Skipping due to exception:', e)
    print("Following files were skipped due to errors:", errors)


if __name__ == "__main__":
    for lang in ["telugu", "russian", "english"]:
        augment_document_image(
            f"../data/raw/data_to_transform/{lang}/", lang + "_augmented"
        )
