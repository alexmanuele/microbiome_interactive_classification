# "Vizomics", a hastily named, interactive, code-free interface for ML on microbial community datasets.
This is an interactice visual analytics tool for analysing 16S amplicon sequence datasets found on [The Knight Lab Microbiome ML Repo](https://knights-lab.github.io/MLRepo/). Currently, the tool includes a subset of classification tasks on the Repo, all of which contain both OTU and taxa representations from both Greengenes and Refseq.<br><br>
The tool allows you to visually select features based on their abundances. After selecting a dataset, adjust sliders for the corresponding feature representation to remove features based on their abundances. In real time, statistics tables and plots will update to reflect the feature distribution of the included features. <br><br>
After selecting features, you can proceed to the Classification page. This page allows you to select from a number of common machine learning models and specify which of their parameters to include in a grid search. The app will automatically perform an exhaustive cross vlaidated search of the parameter space and report the resutls in both table and graph format. This will allow you to identify the best models and best feature representations for each dataset.
<br>
# Installation
In progress. Currently, installation requires Anaconda or miniconda. Check back for pip instructions<br>
For now:<br>

```
git clone https://github.com/alexmanuele/microbiome_interactive_classification.git
```
```
cd microbiome_interactive_classification
```
```
conda env create -f environment.yml && conda activate vizomics
```
```
chmod +x get_data.sh && get_data.sh 
``` 
<br>

# Usage
After installing dependancies and running data script, launch the app with:<br>

```
python app.py
```

Then, open your web browser and navigate to [localhost:8050/](127.0.0.1:8050/)
