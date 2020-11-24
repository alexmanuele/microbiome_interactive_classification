import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    if encode_labels:
        enc = LabelEncoder()
        enc.fit(data['label'].values.reshape(-1,1))
        data['label'] = enc.transform(data['label'].values)

    instance_column = '#SampleID'
    y_column = 'label'

    X = data[[col for col in data.columns if col not in [instance_column, y_column]]].values
    y = data[y_column].values

    return X,y
