# CMT316 Group Project - NLP Sentimental Analysis

## Preparation
### Packages to Install before running codes:
pandas_profiling  
ipywidgets  
pandas  
sklearn  
numpy  
nltk  
Commands to install them all with pip:
```
pip install pandas_profiling ipywidgets pandas sklearn numpy nltk
```

### Data
Data files should be put in a directory with name "data".  
Final file tree should looks like:  
* |-- data  
  * |-- mapping.txt  
  * |-- test_labels.txt  
  * |-- test_text.txt  
  * |-- train_labels.txt  
  * |-- train_text.txt  
  * |-- val_labels.txt  
  * |-- val_text.txt  
* |-- README.md  
* |-- descriptive_analysis.ipynb  
* |-- train_model.ipynb  
* |-- train_model.py  

## Execution
Since descriptive analysis will generate some graphs and run its code with python directly will not show the graph in the terminal, using jupyter notebook to view and execute the codes is recommended.  
To run/view the descriptive analysis and model training codes with jupyter notebook:
```sh
jupyter notebook
```
or with jupyter lab
```
jupyter lab
```

To run the model training codes with python:
```
python train_model.py
```

Note:  
Run train_model.py will take a very long time, since it trains all 5 models, approximately 45 minutes.  
So comment out some codes if you do not want to wait that long.