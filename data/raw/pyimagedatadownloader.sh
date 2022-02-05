#!/bin/bash

# google image scraper utility tool for AIPI540 cv project
# Usage: https://google-images-download.readthedocs.io/en/latest/arguments.html

# clone google-images-download path & install
git clone https://github.com/Joeclinton1/google-images-download.git
cd google-images-download && sudo python setup.py install
cd google_images_download

# google_images_download.py with "-k" query or "-u" option (see the arguments docs)
# k denotes the keywords/key phrases you want to search for
# u allows search by image when you have URL from the Google Images page (recommended option)
python3 google_images_download.py -u "{related images url}" -l 100 -f "jpg" -ct "black-and-white" -s ">640*480" -i "out"




