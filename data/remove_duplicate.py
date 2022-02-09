from difPy import dif


def remove_duplicate_images(dir_path):
    """
    Removes duplicate and low resolution raw data images within the specified directory
    similarity: "high" = search for exact duplicates, very sensitive to details
    :param dir_path: python remove_duplicate.py
    """
    print(f"--> Deleting duplicate images within {dir_path}")
    dif(dir_path, delete=True, similarity="high")


if __name__ == "__main__":
    SUPPORTED_LANGUAGES = ["telugu", "english", "russian"]
    for data in SUPPORTED_LANGUAGES:
        remove_duplicate_images(dir_path="raw/data_to_transform/" + data)
    print(":: All done!")
