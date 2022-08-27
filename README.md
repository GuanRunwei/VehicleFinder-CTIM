# ***VehicleFinder-CTIM: a text-image cross-modal vehicle retrieval system***


  ***VehicleFinder***
  
  <img src="https://github.com/GuanRunwei/VehicleFinder-CTIM/blob/main/VehicleFinder.png" alt="Entity Types of FindVehicle" align=center/>

  Since the whole system would be used for commercial purposes, we only open-source the core module ***CTIM*** (contrastive text-image module).
  
  NanoDet: [link](https://github.com/RangiLyu/nanodet) |  Dataset-> UA-DETRAC: [link](https://pan.baidu.com/s/1-r355_V14YaMXwEhmteqRA) password: bygu 
  
  BiLSTM-CRF: [link](https://github.com/jidasheng/bi-lstm-crf) |  Dataset-> FindVehicle: [link](https://github.com/GuanRunwei/FindVehicle)
  
____________________________________________________________________________

  ***CTIM***
  
  <img src="https://github.com/GuanRunwei/VehicleFinder-CTIM/blob/main/CTIM.png" width=700 height=870 alt="Entity Types of FindVehicle" align=center/>
  
## Requirements:
> einops==0.4.1]

  gensim==4.1.2

  jieba==0.42.1

  matplotlib==3.5.2

  numpy==1.22.4+mkl

  opencv_python_headless==4.5.5.64

  pandas==1.4.2

  Pillow==9.2.0

  scipy==1.8.1

  thop==0.1.0.post2206102148

  torch==1.9.1+cu111

  torchvision==0.10.1+cu111

  tqdm==4.64.0

## Dataset 
### [multi-label -> vehicle proposal] cross modal matching [Baidu Cloud Disk](https://pan.baidu.com/s/1Z5SItSCk437OsR5JnsoWuw) password: so3m
    
    Forward from [UA-DETRAC-ML](https://github.com/GuanRunwei/UA-DETRAC-ML), please cite it if you use it in your research
    
    {
    @misc{uadetracml,
    title={UA-DETRAC-ML},
    author={Runwei Guan},
    howpublished = {\url{https://github.com/GuanRunwei/UA-DETRAC-ML}},
    year = {2022},
    }
    
    

## Implementation
> pip install requirements.txt
  python train.py

  The code of this project is clear, you could find out and replace the hyperparameters and file paths without any difficulty.
