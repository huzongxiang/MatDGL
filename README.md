# CrysNetwork


## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Framework](#crysnet-framework)
* [Contributors](#contributors)
* [References](#references)


<a name="Installation"></a>
## Installation

CrysNet can be installed easily through anaconda! As follows:

+ Create a new conda environment named "crysnet" by command, then activate environment "crysnet":    
```bash
      conda create -n crysnet python=3.8  
      conda activate crysnet 
```
 
+ Configure dependencies of crysnet:
```bash
      conda install tensorflow-gpu==2.6.0  # for CPU conda install tensorflow==2.6.0
```

*If your conda can't find tensorflow-gpu==2.6.0, you can add a new source, e.g.:*
```bash
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  
```

+ Install pymatgen:  
```bash
      conda install --channel conda-forge pymatgen  
```   

+ Install other dependencies:  
```bash
      pip install atom2vec  
      pip install mendeleev  
      conda install graphviz # or pip install graphviz  
      conda install pydot # or pip install pydot  
```

+ Install crysnet:  
```bash
      pip install crysnet  
```
      

<a name="Usage"></a>
## Usage
### Fast testing soon
CrysNet is very easy to use!  
<font color=#00ffff> Just ***three steps*** can finish a fast test using crysnet:</font>  
+ **download test data**  
Get test datas from https://github.com/huzongxiang/CrysNetwork/datas/  
There are three json files in datas: dataset_classification.json, dataset_multiclassification.json and dataset_regression.json.  
+ **prepare workdir**  
Download datas and put it in your trainning work directory, test.py file should also be put in the directory  
+ **run command**
run command:  
```bash
      python test.py  
```
You have finished your testing multi-classification trainning! The trainning results and model weight could be saved in /results and /models, respectively.

### Understanding trainning script
You can use crysnet by provided trainning scripts in user_easy_trainscript only, but understanding script will help you custom your trainning task!   
     
+ **get datas**  
Get current work directory of running trainning script, the script will read datas from 'workdir/datas/' , then saves results and models to 'workdir/results/' and 'workdir/models/'
```python
      from pathlib import Path
      ModulePath = Path(__file__).parent.absolute() # workdir
```

+ **fed trainning datas**  
Module Dataset will read data from 'ModulePath/datas/dataset.json', 'task_type' defines regression/classification/multi-classification, 'data_path' gets path of trainning datas.
```python
      from crysnet.data import Dataset
      dataset = Dataset(task_type='multiclassfication', data_path=ModulePath)
```

+ **generator**  
Module GraphGenerator feds datas into model during trainning. The Module splits datas into train, valid, test sets, and transform structures data into labelled graphs and gets three generators.
BATCH_SIZE is batch size during trainning, DATA_SIZE defines number of datas your used in entire datas, CUTOFF is cutoff of graph edges in crystal.
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

+ **building model**  
Module GNN defines a trainning model. TransformerModel, GraphModel and MpnnModel are different model. TransformerModel is a graph transformer. MpnnModel is a massege passing neural network. GraphModel is a combination of TransformerModel and MpnnModel.
```python
      from crysnet.models import GNN
      from crysnet.models.graphmodel import GraphModel, MpnnModel, TransformerModel 
      gnn = GNN(model=TransformerModel,
            atom_dim=16
            bond_dim=64
            num_atom=118
            state_dim=16
            sp_dim=230
            units=32
            edge_steps=1
            message_steps=1
            transform_steps=1
            num_attention_heads=8
            dense_units=64
            output_dim=64
            readout_units=64
            dropout=0.0
            reg0=0.00
            reg1=0.00
            reg2=0.00
            reg3=0.00
            reg_rec=0.00
            batch_size=BATCH_SIZE
            spherical_harmonics=True
            regression=dataset.regression
            optimizer = 'Adam'
            )
```

+ **trainning**  
Using trainning function of model to train. Common trainning parameters can be defined, workdir is current directory of trainning script, it saves results of model during trainning. If test_data exists, model will predict on test_data.
```python
      gnn.train(train_data, valid_data, test_data, epochs=700, lr=3e-3, warm_up=True, load_weights=False, verbose=1, checkpoints=None, save_weights_only=True, workdir=ModulePath)
```

+ **prediction**  
The simplest method for predicting is using script predict.py in /user_easy_train_scripts.  
Using predict_data funciton to predict.
```python
      gnn.predict_datas(test_data, workdir=ModulePath)
```

+ **preparing your custom datas**  
If you have your structures (and labels), the Dataset receives pymatgen.core.Structure type. So you should transform your POSCAR or cif to pymatgen.core.Structure type.
```python
      import os
      from pymatgen.core.structure import Structure
      structures = []
      for cif in os.listdir(cif_path):
            structures.append(Structure.from_file(cif)) # the same as POSCAR

      # save your structures and labels to dataset
      from crysnet.data import Dataset
      dataset = Dataset(task_type='train', data_path=ModulePath)
      dataset.save_datasets(strurtures, labels)
```

<a name="Crysnet-framework"></a>
## Framework
CrysNet 


<a name="Contributors"></a>
## Contributors
Zongxiang Hu
