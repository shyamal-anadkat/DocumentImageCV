AIPI 540 Computer Vision
========================

Christopher Oblack, Shyamal Anadkat, Yudong Liu

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