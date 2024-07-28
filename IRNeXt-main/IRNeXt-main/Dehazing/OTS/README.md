## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
pip install -r requirements.txt
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~

### Train
~~~
cd IRNeXT-main/Dehazing/OTS
python test.py --mode train --data_dir your_path/dataset
~~~
#### Testing 
~~~
cd IRNeXT-main/Dehazing/OTS
python test.py --data_dir your_path/dataset --test_model path_to_model
~~~

#### Params

Available arguments for testing and training are available in test.py

For training and testing, your directory structure should look like this
Images should be named numeracally only, e.g. 1.png, 2.png, etc ...

`Your path` <br/>
`├──dataset` <br/>
     `├──train`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──gt`  <br/>
          `└──hazy`   