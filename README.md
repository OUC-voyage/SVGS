# AD-GS
1. Clone this repo: git clone https://github.com/OUC-voyage/AD-GS.git --recursive

2. Install dependencies:

   conda env create --file environment.yml//

   conda activate AD_GS
   
4. Due to the size of the dataset, we host it via an anonymous external link: https://drive.google.com/drive/folders/1Ch1iyiIHldGdVhuYf5JrVIMe7pCTiJpg?usp=drive_link

5. First, create a data/ folder inside the project path by: mkdir data
   The data structure will be organised as follows:
    data/
    ├── dataset_name
    │   ├── scene1/
    │   │   ├── images
    │   │   │   ├── IMG_0.jpg
    │   │   │   ├── IMG_1.jpg
    │   │   │   ├── ...
    │   │   ├── sparse/
    │   │       └──0/
    │   ├── scene2/
    │   │   ├── images
    │   │   │   ├── IMG_0.jpg
    │   │   │   ├── IMG_1.jpg
    │   │   │   ├── ...
    │   │   ├── sparse/
    │   │       └──0/
    ...
6. Run rendering and evaluation
   Run rendering: python render.py -s "AD-GS/data/waymo/$dataset" -m "output/waymo/$dataset" --skip_train
   Evaluate metrics: python metrics.py -m "output/waymo/$dataset"
