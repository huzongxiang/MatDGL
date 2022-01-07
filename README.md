# CrysNet
GrysNet is a neural network package that allows researchers to train custom models for crystal modeling tasks. It aims to accelerate the research and application of material science.  

## Table of Contents

* [Hightlights](#hightlights)
* [Installation](#installation)
* [Usage](#usage)
* [Framework](#crysnet-framework)
* [Contributors](#contributors)
* [References](#references)

<a name="Hightlights"></a>
## Hightlights
+ Easy to installation.
+ Three steps to fast testing.
+ Flexible and adaptive to user's trainning task.

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
      
      #if task is multiclassfication, should define variable multiclassifiction
      multiclassification = Generators.multiclassification  
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
      gnn.predict_datas(test_data, workdir=ModulePath)    # predict on test datas with labels
      y_pred_keras = gnn.predict(datas)                   # predict on new datas without labels
```

+ **preparing your custom datas**  
If you have your structures (and labels), the Dataset receives pymatgen.core.Structure type. So you should transform your POSCAR or cif to pymatgen.core.Structure type.
```python
      import os
      from pymatgen.core.structure import Structure
      structures = []                                      # your structure list
      for cif in os.listdir(cif_path):
            structures.append(Structure.from_file(cif))    # for POSCAR too

      # construct your dataset
      from crysnet.data import Dataset
      dataset = Dataset(task_type='my_classification', data_path=ModulePath)  # task_type could be my_regression, my_classification, my_multiclassification
      dataset.prepare_x(structures)
      dataset.prepare_y(labels)   # if you have labels used to trainning model, labels could be None in prediction on new datas without labels
      
      # alternatively, you can construct dataset as follow
      dataset.structures = structures
      dataset.labels = labels

      # save your structures and labels to dataset in dataset_my*.json
      dataset.save_datasets(strurtures, labels)
      
      # for prediction on new datas without labels, Generators has not attribute multiclassification, should assign definite value
      Generators = GraphGenerator(dataset, data_size=DATA_SIZE, batch_size=BATCH_SIZE, cutoff=CUTOFF)     # dataset.labels is None
      Generators.multiclassification = 5
      multiclassification = Generators.multiclassification  # multiclassification = 5
      
```

+ **custom your model and trainning**  
The Module GNN provides a flexible trainning framework to accept tensorflow.keras.models.Model type customized by user. Yon can custom your model and train the model according to the following example.
```python
      from tensorflow.keras.models import Model
      from tensorflow.keras import layers
      from crysnet.layers import MessagePassing
      from crysnet.layers import PartitionPadding

      def MyModel(
              bond_dim,
              atom_dim=16,
              num_atom=118,
              state_dim=16,
              sp_dim=230,
              units=32,
              message_steps=1,
              readout_units=64,
              batch_size=16,
              regression=False,
              multiclassification=None,
              ):
              atom_features = layers.Input((), dtype="int32", name="atom_features_input")
              atom_features_ = layers.Embedding(num_atom, atom_dim, dtype="float32", name="atom_features")(atom_features)
              bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
              local_env = layers.Input((6), dtype="float32", name="local_env")
              state_attrs = layers.Input((), dtype="int32", name="state_attrs_input")   
              state_attrs_ = layers.Embedding(sp_dim, state_dim, dtype="float32", name="state_attrs")(state_attrs)

              pair_indices = layers.Input((2), dtype="int32", name="pair_indices")

              atom_graph_indices = layers.Input(
                  (), dtype="int32", name="atom_graph_indices"
              )

              bond_graph_indices = layers.Input(
                  (), dtype="int32", name="bond_graph_indices"
              )

              pair_indices_per_graph = layers.Input((2), dtype="int32", name="pair_indices_per_graph")

              x = MessagePassing(message_steps)(
                  [atom_features_, edge_features, state_attrs_, pair_indices,
                   atom_graph_indices, bond_graph_indices]
              )

              x = x[0]

              x = PartitionPadding(batch_size)([x, atom_graph_indices])

              x = layers.BatchNormalization()(x)

              x = layers.GlobalAveragePooling1D()(x)

              x = layers.Dense(readout_units, activation="relu", name='readout0')(x)

              x = layers.Dense(readout_units//2, activation="relu", name='readout1')(x)

              if regression:
                  x = layers.Dense(1, name='final')(x)
              elif multiclassification is not None:
                  x = layers.Dense(multiclassification, activation="softmax", name='final_softmax')(x)
              else:
                  x = layers.Dense(1, activation="sigmoid", name='final')(x)

              model = Model(
                  inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                          bond_graph_indices, pair_indices_per_graph],
                  outputs=[x],
              )
              return model

      from crysnet.models import GNN
      gnn = GNN(model=MyModel,        
              atom_dim=16,
              bond_dim=64,
              num_atom=118,
              state_dim=16,
              sp_dim=230,
              units=32,
              message_steps=1,
              readout_units=64,
              batch_size=16,
              optimizer='Adam',
              regression=False,
              multiclassification=None,)
      gnn.train(train_data, valid_data, test_data, epochs=700, lr=3e-3, warm_up=True, load_weights=False, verbose=1, checkpoints=None, save_weights_only=True, workdir=ModulePath)
```
      You can set edge as your model output.
```python
      from crysnet.layers import EdgeMessagePassing
      def MyModel(
              bond_dim,
              atom_dim=16,
              num_atom=118,
              state_dim=16,
              sp_dim=230,
              units=32,
              message_steps=1,
              readout_units=64,
              batch_size=16,
              regression=False,
              multiclassification=None,
              ):
              atom_features = layers.Input((), dtype="int32", name="atom_features_input")
              atom_features_ = layers.Embedding(num_atom, atom_dim, dtype="float32", name="atom_features")(atom_features)
              bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
              local_env = layers.Input((6), dtype="float32", name="local_env")
              state_attrs = layers.Input((), dtype="int32", name="state_attrs_input")   
              state_attrs_ = layers.Embedding(sp_dim, state_dim, dtype="float32", name="state_attrs")(state_attrs)

              pair_indices = layers.Input((2), dtype="int32", name="pair_indices")

              atom_graph_indices = layers.Input(
                  (), dtype="int32", name="atom_graph_indices"
              )

              bond_graph_indices = layers.Input(
                  (), dtype="int32", name="bond_graph_indices"
              )

              pair_indices_per_graph = layers.Input((2), dtype="int32", name="pair_indices_per_graph")

              x = EdgeMessagePassing(units,
                                        edge_steps,
                                        kernel_regularizer=l2(reg0),
                                        sph=spherical_harmonics
                                        )([bond_features, local_env, pair_indices])

              x = x[1]

              x = PartitionPadding(batch_size)([x, bond_graph_indices])

              x = layers.BatchNormalization()(x)

              x = layers.GlobalAveragePooling1D()(x)

              x = layers.Dense(readout_units, activation="relu", name='readout0')(x)

              x = layers.Dense(readout_units//2, activation="relu", name='readout1')(x)

              if regression:
                  x = layers.Dense(1, name='final')(x)

              model = Model(
                  inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                          bond_graph_indices, pair_indices_per_graph],
                  outputs=[x],
              )
              return model
```

      The Module GNN has some basic parameter necessary to be defined but not necessary to be used：
```python
      class GNN:
          def __init__(self,
              model: Model,
              atom_dim=16,
              bond_dim=32,
              num_atom=118,
              state_dim=16,
              sp_dim=230,
              batch_size=16,
              regression=True,
              optimizer = 'Adam',
              multiclassification=None,
              **kwargs,
              ):
              """
              pass
              """
```


<a name="Crysnet-framework"></a>
## Framework
CrysNet 


<a name="Contributors"></a>
## Contributors
Zongxiang Hu


<a name="References"></a>
## References
