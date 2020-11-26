import itertools as it
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import plotly.express as px
import plotly.graph_objects as go

# Returns four dataframes: OTU representation and Taxa represntation form both greengenes and refseq.
def load_dataset(directory):
    dir_path = "datasets/{}/".format(directory)
    label_path = dir_path + 'task.txt'
    otu_gg_path = dir_path + 'otu_gg.txt'
    otu_rf_path = dir_path + 'otu_refseq.txt'
    tax_gg_path = dir_path + 'taxa_gg.txt'
    tax_rf_path = dir_path + 'taxa_refseq.txt'

    labels = pd.read_table(label_path)
    labels.columns = ['#SampleID', 'label']
    # Reformat to row-wise samples. This is my personal preference.
    def reformat_table(df):
        df = df.T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        return df

    #The datasets from Knight et al are sometimes agglomerated from several tasks
    # remove rows and columns with only zeros.
    def drop_zeros(df):
        val_cols = [col for col in df if col not in ['#SampleID', 'label']]
        #drop rows with all zeros. There shouldnt be any but just in case.
        df = df.loc[~(df[val_cols]==0).all(axis=1)]
        #drop columns with all zeros.
        df = df.loc[:, (df!=0).any(axis=0)]
        return df

    otu_gg = reformat_table(pd.read_table(otu_gg_path))
    otu_rf = reformat_table(pd.read_table(otu_rf_path))
    taxa_gg = reformat_table(pd.read_table(tax_gg_path))
    taxa_rf = reformat_table(pd.read_table(tax_rf_path))


    #Join the labels and drop the zero-only rows and columns. return as dict
    data = {'greengenes': {'otu': drop_zeros(labels.join(otu_gg, on="#SampleID")),
                           'taxa': drop_zeros(labels.join(taxa_gg, on='#SampleID'))},
            'refseq': {'otu': drop_zeros(labels.join(otu_rf, on="#SampleID")),
                       'taxa': drop_zeros(labels.join(taxa_rf, on='#SampleID'))}
           }
    return data


#get feature values
def get_feature_frequencies(df):
    value_cols = [col for col in df.columns if col not in ['#SampleID', 'label']]
    feature_sum = df[value_cols].sum(axis=0).values
    return feature_sum

#Expects datasets as formatted from load_dataset.
def dataset_to_X_y(data, encode_labels=False):
    instance_column = '#SampleID'
    y_column = 'label'

    X = data[[col for col in data.columns if col not in [instance_column, y_column]]].values
    y = data[y_column].values
    if encode_labels:
        enc = LabelEncoder()
        enc.fit(y.reshape(-1,1))
        y = enc.transform(y)
    return X,y


# Adapted from http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
# Let's you do sklearn GridSearchCV with multiple different models.
class PipelineHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Missing params for %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=8, verbose=1, scoring=None):
        for key in self.keys:
            print("Gridsearch for %s" % key)

            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,verbose=verbose, scoring=scoring)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d= {'estimator':key,
                'min_score':min(scores),
                'max_score':max(scores),
                'mean_score':np.mean(scores),
                'std_score': np.std(scores),
               }
            return pd.Series({**params, **d})
        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

#convenienc method for doing gridsearch for each feature representation.
def get_all_results(models, params, dataset):
    clf_results = {}
    for feature_rep in it.product(['greengenes', 'refseq'], ['otu', 'taxa']):
        data = dataset[feature_rep[0]][feature_rep[1]]
        X, y = dataset_to_X_y(data)
        gridsearch = PipelineHelper(models, params)
        gridsearch.fit(X,y)
        clf_results["{}_{}".format(*feature_rep)] = gridsearch
    return clf_results

def bar_plot_best(results):
    x = []
    y = []
    hovertext= []
    error_y = []
    colors= ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    for trial in results:
        clf = results[trial]
        df = clf.score_summary()
        best = clf.grid_searches[df.iloc[0]['estimator']].best_estimator_
        description = df.iloc[0].dropna().to_dict()
        desc_str = ""
        for key, value in description.items():
            desc_str += "{} : {}</br>".format(key, value)
        x.append(trial)
        y.append(description['mean_score'])
        hovertext.append(desc_str)
        error_y.append(description['std_score'])
    fig = go.Figure(data=[go.Bar(x=x, y=y, hovertext=hovertext, error_y=dict(array=error_y), marker_color=colors)])
    return fig
