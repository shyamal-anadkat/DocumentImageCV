[![build](https://github.com/shyamal-anadkat/AIPI540_ComputerVision/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/shyamal-anadkat/AIPI540_ComputerVision/actions/workflows/main.yml)

AIPI 540 Computer Vision
========================

Christopher Oblack, Shyamal Anadkat, Yudong Liu

**Problem definition:** Detect the language from a text based image/picture of a dense document. 

Note that OCR is a separate problem and we envision this  model to be a “preprocessing” step for language detection 
from an image. The idea is to build a classification CV model that would output the confidence levels of the detected
language(s). For example, I can click a picture of a piece of text on my iPhone, and the model would output if it’s in Gujarati or Telugu. 


---------------


Note: if you use Colab Pro for your project, you can include a **run.ipynb** notebook file which accesses your data, imports modules and runs your project.  Be sure not to accidentally include any secrets in this file.

```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile                <- setup and run project from command line
├── setup.py                <- script to set up project (get data, build features, train model)
├── app.py                  <- app to run project / user interface
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── make_dataset.py     <- script to transfomr data
    ├── remove_corrupt.py   <- script to remove corrupt images from raw
    ├── remove_duplicate.py <- script to remove duplicate/similar & low res. raw images
    ├── model.py            <- script to train model and predict
├── models                  <- directory for trained models
├── data                    <- directory for project data
    ├── raw                 <- directory for raw & transformed data or script to download
├── notebooks               <- directory to store any exploration notebooks used
├── .gitignore              <- git ignore file
```

Run app: ` streamlit run app.py
`

```
100 epochs | Feb 5 
Training complete in 29m 2s
Best val Acc: 0.810651
Test set accuracy is 0.811
For class english, recall is 0.8484848484848485
For class russian, recall is 0.6727272727272727
For class telugu, recall is 0.9166666666666666
```

![image](https://user-images.githubusercontent.com/12115186/152719849-61043715-60fc-4bbd-bfc6-9db34b5e4b76.png)
