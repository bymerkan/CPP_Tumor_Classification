# CPP_Tumor_Classification
Transfer Learning for Image Classification with PyTorch C++ API 

This repo contains training files for PyTorch models with custom datasets.

* Please see [<code>src/main.cpp</code>](https://github.com/bymerkan/CPP_Tumor_Classification/blob/0c89687f172741c28aec82572a8409dd5b29db0b/src/main.cpp) file. 
* In order to change options for training, use [<code>include/customDataset.hpp</code>](https://github.com/bymerkan/CPP_Tumor_Classification/blob/0c89687f172741c28aec82572a8409dd5b29db0b/include/customDataset.hpp).
* Dataset should be placed in <code>./data</code> folder.
* Class name-label pairs should be changed. [(<code>src/customDataset.cpp:100</code>)](https://github.com/bymerkan/CPP_Tumor_Classification/blob/45a1fcd5640fc05f547f406adc84b58c9a5893c1/src/customDataset.cpp#L100)
* LibTorch path must be specified. (<code>cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..</code>)
* Run <code>make</code>.
