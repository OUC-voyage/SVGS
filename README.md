# AD-GS
1. Clone this repo: git clone https://github.com/OUC-voyage/AD-GS.git --recursive

2. Install dependencies:
   conda env create --file environment.yml
   
   conda activate AD_GS
   
4. Due to the size of the dataset, we host it via an anonymous external link: https://drive.google.com/drive/folders/1Ch1iyiIHldGdVhuYf5JrVIMe7pCTiJpg?usp=drive_link

   Due to the large size of the 3D scene PLY files required for inference, we provide the training results via an external anonymous link: https://drive.google.com/drive/folders/1IPYexCjijKZh8EKflaT3ImhiigX3HMCj?usp=drive_link
   
6. First, create a data/ folder inside the project path by: mkdir data
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

    ...
8. Run rendering and evaluation
9. 
   Run rendering: python render.py -s "AD-GS/data/waymo/$dataset" -m "output/waymo/$dataset" --skip_train
   
   Evaluate metrics: python metrics.py -m "output/waymo/$dataset"
