# CrysNetwork


## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Framework](#crysnet-framework)
* [Contributors](#contributors)
* [References](#references)


<a name="Installation"></a>
## Installation

CrysNet can be installed easily through anaconda!

Create a new conda environment named 'crysnet' by command：  
```bash
      conda create -n crysnet python=3.8  
```
Then activate environment 'crysnet'：  
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
## Usage
### do your test
CrysNet is very easy to use!  
Get test datas from https://github.com/huzongxiang/CrysNetwork/datas/  
There are three json files in datas: dataset_classification.json, dataset_multiclassification.json and dataset_regression.json.  
Download datas and put it in your trainning directory, test.py file should also be put in the directory, then run command:  
```bash
      python test.py  
```
You have finished your testing multi-classification trainning! The trainning results and model weight could be saved in ./results and ./models, respectively.

### trainning script
You can use crysnet by provided trainning scripts in user_easy_trainscript only, but understanding script will help you custom your trainning task!   
#### get datas
Get current work directory of running trainning script, the script will read datas from 'workdir/datas/' , then saves results and models to 'workdir/results/' and 'workdir/models/'
```python
from pathlib import Path
ModulePath = Path(__file__).parent.absolute()
```
#### fed trainning datas
Module Dataset will read data from 'ModulePath/datas/dataset.json', 'task_type' defines regression/classification/multi-classification, 'data_path' gets path of trainning datas.
```python
from crysnet.data import Dataset
dataset = Dataset(task_type='dos_fermi', data_path=ModulePath)
```
#### generator
Module GraphGenerator feds datas into model during trainning. The Module splits datas into train, valid, test sets, and transform structures data into labelled graphs.
BATCH_SIZE is batch size during trainning, DATA_SIZE defines number of datas your used in entire datas, CUTOFF is cutoff of bond distance in crystal.
```python
from crysnet.data.generator import GraphGenerator
BATCH_SIZE = 64
DATA_SIZE = None
CUTOFF = 2.5
Generators = GraphGenerator(dataset, data_size=DATA_SIZE, batch_size=BATCH_SIZE, cutoff=CUTOFF)
train_data = Generators.train_generator
valid_data = Generators.valid_generator
test_data = Generators.test_generator
```
<a name="CrysNet Framework"></a>
## Framework
CrysNet 


<a name="Contributors"></a>
## Contributors
Zongxiang Hu
