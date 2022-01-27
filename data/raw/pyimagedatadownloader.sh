#!/bin/bash

# google image scraper utility tool for aipi540 cv project
# Usage: https://google-images-download.readthedocs.io/en/latest/arguments.html

git clone https://github.com/Joeclinton1/google-images-download.git
cd google-images-download && sudo python setup.py install

#pip install -r requirements.txt
#python setup.py install
cd google_images_download
python3 google_images_download.py -k "russian paragraphs" -l 100 -f "jpg" -la "Russian" -ct "black-and-white" -s ">640*480" -i "out"




