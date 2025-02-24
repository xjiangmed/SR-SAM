## Requirement
``pip install -r requirements.txt``

## Data Preparation
[Polyp Segmentation](https://github.com/DengPingFan/PraNet) Download the polyp segmentation dataset from PraNet's repo and reorganize it into four folders, one for each dataset.
[Prostate Segmentation](https://liuquande.github.io/SAML/)

Taking the polyp dataset as an example, the data is organized as follows:
Polyp
    ETIS
        images
            1.png
            ...
        masks
            1.png
            ...
    Kvasir
    CVC-ClinicDB
    CVC-ColonDB
    ETIS.csv
    Kvasir.csv
    CVC-ClinicDB.csv
    CVC-ColonDB.csv

## Polyp Segmentation
bash run_CVC-ClinicDB.sh

## TO DO
The complete train and test code will be provided soon.






