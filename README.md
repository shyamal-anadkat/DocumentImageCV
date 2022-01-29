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
├── Makefile [OPTIONAL]     <- setup and run project from command line
├── run.ipynb [OPTIONAL]    <- run project on Google Colab (only include if using Google Colab for project)
├── setup.py                <- script to set up project (get data, build features, train model)
├── app.py                  <- app to run project / user interface
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── make_dataset.py     <- script to get data [OPTIONAL]
    ├── build_features.py   <- script to run pipeline to generate features [OPTIONAL]
    ├── model.py            <- script to train model and predict [OPTIONAL]
├── models                  <- directory for trained models
├── data                    <- directory for project data
    ├── raw                 <- directory for raw data or script to download
    ├── processed           <- directory to store processed data
    ├── outputs             <- directory to store any output data
├── notebooks               <- directory to store any exploration notebooks used
├── .gitignore              <- git ignore file
```
