# ***VehicleFinder-CTIM: a text-image cross-modal vehicle retrieval system***

  Since the whole system would be used for commercial purposes, we only open-source the core module ***CTIM*** (contrastive text-image module).
  
  <img src="https://github.com/GuanRunwei/VehicleFinder-CTIM/blob/main/CTIM.png" width=700 height=870 alt="Entity Types of FindVehicle" align=center/>
  
## Requirements:
> einops==0.4.1
> gensim==4.1.2
> jieba==0.42.1
> matplotlib==3.5.2
> numpy==1.22.4+mkl
> opencv_python_headless==4.5.5.64
> pandas==1.4.2
> Pillow==9.2.0
> scipy==1.8.1
> thop==0.1.0.post2206102148
> torch==1.9.1+cu111
> torchvision==0.10.1+cu111
> tqdm==4.64.0

## Dataset 
### [multi-label -> vehicle proposal] cross modal matching [Baidu Cloud Disk](https://pan.baidu.com/s/1Z5SItSCk437OsR5JnsoWuw)
    
    Forward from [UA-DETRAC-ML](https://github.com/GuanRunwei/UA-DETRAC-ML)
    
    password: so3m

## Implementation
> python train.py
