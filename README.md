# CrysNetwork

# Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [CrysNet Framework](#crysnet-framework)
* [Contributors](#contributors)
* [References](#references)

<a name="Installation"></a>
# Installation

CrysNet can be installed easily through anaconda!

Create a new conda environment named 'crysnet' by command：  
```bash
      conda create -n crysnet python=3.8  
```
Then activate environment crysnet：  
```bash
      conda activate crysnet  
```    
Configure dependencies of crysnet:  
```bash
      conda install tensorflow-gpu==2.6.0  
```
If your conda can not find tensorflow-gpu==2.6.0, you can add a new source：  
```bash
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  
```
Install pymatgen:  
```bash
      conda install --channel conda-forge pymatgen  
```      
Install other dependencies:  
```bash
      pip install atom2vec  
      pip install mendeleev  
      pip install graphviz or conda install graphviz  
      pip install pydot or conda install pydot  
```
Install crysnet:  
```bash
      pip install crysnet  
```
      

<a name="Usage"></a>
# Usage

CrysNet is very easy to use!  
Get test datas from https://github.com/huzongxiang/CrysNetwork/datas/  
There are three json files in datas: dataset_classification.json, dataset_multiclassification.json and dataset_regression.json.  
Download datas and put it in your trainning directory, test.py file should also be put in the directory, then run command:  
```bash
      python test.py  
```


<a name="CrysNet Framework"></a>
# CrysNet Framework
CrysNet 


<a name="Contributors"></a>
# Contributors
Zongxiang Hu
