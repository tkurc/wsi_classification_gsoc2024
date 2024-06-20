# [GSoC 2024 Project.](https://github.com/camicroscope/GSOC)

**Overview**
This project aims to leverage pre-trained and foundation AI models for patch-level classification in whole slide tissue images. The approach involves using these models as encoders to train task-specific models that classify image patches. Benefits include reduced training costs, less required training data, and potentially more accurate and robust models. The project will utilize models from Hugging Face and develop software to (1) search for and download pre-trained models, (2) choose a classification network, (3) train this network using the pre-trained model as the encoder, and (4) apply the trained model to image patches.

# 1. Specification of dependencies
```
git clone https://github.com/tkurc/wsi_classification_gsoc2024
cd code # go into code directory

# If you GPU follow this:
conda update conda
conda env create -f environment.yml
conda activate camicro

# Don't have GPU!!! No worries follow this:
conda create -n camicro python=3.9 -y # Create fresh env
conda activate camicro
pip install -r requirements.txt 
```

# 2. Dataset
Training dataset by combining manually annotated patches(strong annotations)from 18 TCGA cancer types(ACC, BRCA,COAD, ESCA, HNSC, KIRC, LIHC, LUAD, MESO, OV, PAAD, PRAD, SARC, SKCM, TGCT, THYM, UCEC, and UVM) and model generated annotations from 4 TCGA cancer types (CESC, LUSC, READ, and STAD). For more details and download the dataset visit [here](https://zenodo.org/records/6604094).

<img src="https://drive.google.com/file/d/1aLL1PWk9LibT_p5ieH9izxFjVwUEKdqo/view?usp=drive_link" width="350" alt="accessibility text">


# 3. Run Demo
```
streamlit run app.py
```
