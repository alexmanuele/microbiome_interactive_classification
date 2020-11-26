import itertools as it

import numpy as np
import pandas as pd

import dash
from dash.dependencies import Output, Input, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from components import *
from utils import *

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

#Update here if you add more datasets.
data_paths = ['IBD_Gevers', 'plant_v_animal', 'usa_vs_malawi']
datasets = {path: load_dataset(path) for path in data_paths}

################################################################################
### Page Layouts                                                             ###
################################################################################

### Entry Point. Also Serves as data dump for sharing between apps  ###
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    #Stores for data persistence.
    dcc.Store(id='selected-dataset', storage_type='session'),
    dcc.Store(id='slider-values', storage_type='session'),

    #html.Div(id='selected-dataset', style={'display':'none'}),
    #html.Div(id='slider-values', style={'display':'none'}),
    make_navbar(active=0),
    make_dataset_dropdown(id='summary-dataset-dropdown'),
    dbc.Row(id='data-sliders',children=[
        dbc.Col([
            dcc.Graph(id='feature-boxplot'),
            html.H5('Use sliders to remove features based on their frequencies.'),
            dbc.Label("Greengenes OTU"),
            dcc.RangeSlider(id='gg-otu-slider', allowCross=False, tooltip={'placement': "top"}, disabled=True),
            dbc.Label("Greengenes Taxa"),
            dcc.RangeSlider(id='gg-taxa-slider', allowCross=False,tooltip={'placement': "top"},disabled=True),
            dbc.Label('Refseq OTU'),
            dcc.RangeSlider(id='refseq-otu-slider', allowCross=False, tooltip={'placement': "top"},disabled=True),
            dbc.Label('Refseq Taxa'),
            dcc.RangeSlider(id='refseq-taxa-slider', allowCross=False,tooltip={'placement': "top"},disabled=True),

        ]),
    ]),
    html.Div(id='page-content'),
])

### Data Preprocessing Layout ###
page1_layout = dbc.Container(fluid=True,children=[
    dbc.Row([
        dbc.Col(id='otu-gg-table', width=3,),
        dbc.Col(id='taxa-gg-table', width=3,),
        dbc.Col(id='otu-refseq-table',width=3,),
        dbc.Col(id='taxa-refseq-table',width=3),
    ]),

])
### Machine Learning Layout ###
page2_layout = dbc.Container(fluid=True,children=[
    dbc.Row([
        #Machine Learning Results Area
        dbc.Col(children=[dcc.Graph(id='ml-graph-area')], width=8),
        #Machine Learning Params Area
        dbc.Col(children=[
            dcc.Dropdown(id='feature-selection',
                        options=[
                            {'label': "Chi2 Information Test", 'value':'chi2'},
                            {'label': "Mutual Information (Entropy)", 'value': "mi"}
                        ],
                        multi=False,
                        placeholder="Choose feature selection algorithm or leave blank for none"
            ),
            dbc.Input(
                id='n-features',
                type='number',
                min=1,
                max=100,
                step=1,
                disabled=True,
                placeholder="N features"
            ),
            dcc.Dropdown(id='models-select',
                         options=[
                            {'label': "Random Forest", 'value':'RF'},
                            {'label': "Naive Bayes", 'value': 'NB'},
                            {'label': "LinearSVM", 'value': 'SVM'}
                         ],
                         multi=True,
                         placeholder="Select models",
            ),
            #Ranfom Forest Params
            dbc.Card(children=[
                        dbc.CardHeader('Random Forest Parameters'),
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    [dcc.Dropdown(id='rf-criterion',
                                        options=[
                                            {'label': 'Gini Impurity', 'value':'gini'},
                                            {'label': 'Entropy', 'value': 'entropy'}],
                                        multi=True,
                                        placeholder='Criteria'
                                    )]),
                                dbc.ListGroupItem(
                                    [dcc.Dropdown(id='rf-n-estimators',
                                    options=[
                                        {'label': '10', 'value': 10},
                                        {'label': '50', 'value': 50},
                                        {'label': '100', 'value': 100},
                                        {'label': '500', 'value': 500},
                                    ],
                                    multi=True,
                                    placeholder='N Estimators')]
                                ),
                            ])
                    ],
                    id='rf-params', style={'display':'none'},
                    ),
            #SVC Params
            dbc.Card(children=[
                        dbc.CardHeader('SVC Parameters'),
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    [dcc.Dropdown(id='svc-penalty',
                                        options=[
                                            {'label': 'L1', 'value':'l1'},
                                            {'label': 'L2', 'value': 'l2'}],
                                        multi=True,
                                        placeholder='Criteria'
                                    )]),
                                dbc.ListGroupItem(
                                    [dcc.Dropdown(id='svc-loss',
                                        options=[
                                            {'label': 'Hinge', 'value':'hinge'},
                                            {'label': 'Squared Hinge', 'value': 'squared_hinge'}],
                                        multi=True,
                                        placeholder='Loss'
                                    )]),
                                dbc.ListGroupItem(
                                    [dcc.Dropdown(id='svc-c',
                                    options=[
                                        {'label': '2^-3', 'value': 0.125},
                                        {'label': '2^-2', 'value': 0.25},
                                        {'label': '2^-1', 'value': 0.5},
                                        {'label': '1', 'value': 1},
                                        {'label': '2^2', 'value': 4},
                                        {'label': '2^3', 'value': 8},
                                        {'label': '2^4', 'value': 16},
                                    ],
                                    multi=True,
                                    placeholder='C')]
                                ),
                            ])
                    ],
                    id='svc-params', style={'display':'none'},
                    ),
            #Naive Bayes Params
            dbc.Card(children=[
                        dbc.CardHeader('Naive Bayes Parameters'),
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    [dcc.Dropdown(id='nb-alpha',
                                        options=[
                                            {'label': 'No smoothing', 'value': 0},
                                            {'label': 'Laplacian smoothing', 'value': 1}],
                                        multi=True,
                                        placeholder='Smoothing'
                                    )]),
                                dbc.ListGroupItem(
                                    [dcc.Dropdown(id='nb-prior',
                                        options=[
                                            {'label': 'Fit prior', 'value': 'true'},
                                            {'label': 'Uniform Prior', 'value': 'false'}],
                                        multi=True,
                                        placeholder="Prior",
                                    )]),
                            ])
                    ],
                    id='nb-params', style={'display':'none'},
                    ),
            dbc.Button("Run GridSearch", id='submit-button', color="warning", disabled=True),

        ], width=4),
    ]),
])


###############################################################################
### Data Preprocessing Callbacks                                            ###
###############################################################################

#populate sliders and update plot when selecting a dataset
@app.callback(
    [#Output('feature-boxplot', 'figure'),
    Output('gg-otu-slider', 'min'),
    Output('gg-otu-slider', 'max'),
    Output('gg-otu-slider', 'value'),
    Output('gg-otu-slider', 'disabled'),
    Output('gg-taxa-slider', 'min'),
    Output('gg-taxa-slider', 'max'),
    Output('gg-taxa-slider', 'value'),
    Output('gg-taxa-slider', 'disabled'),
    Output('refseq-otu-slider', 'min'),
    Output('refseq-otu-slider', 'max'),
    Output('refseq-otu-slider', 'value'),
    Output('refseq-otu-slider', 'disabled'),
    Output('refseq-taxa-slider', 'min'),
    Output('refseq-taxa-slider', 'max'),
    Output('refseq-taxa-slider', 'value'),
    Output('refseq-taxa-slider', 'disabled'),],
    [Input('summary-dataset-dropdown', 'value'),]
)
def update_dataset(value):
    if value:
        dataset = datasets[value]
        otu_gg_ff = get_feature_frequencies(dataset['greengenes']['otu'])
        taxa_gg_ff = get_feature_frequencies(dataset['greengenes']['taxa'])
        otu_refseq_ff = get_feature_frequencies(dataset['refseq']['otu'])
        taxa_refseq_ff = get_feature_frequencies(dataset['refseq']['taxa'])


        return (otu_gg_ff.min(), otu_gg_ff.max(), [otu_gg_ff.min(), otu_gg_ff.max(),], False,
                taxa_gg_ff.min(), taxa_gg_ff.max(), [taxa_gg_ff.min(), taxa_gg_ff.max(),], False,
                otu_refseq_ff.min(), otu_refseq_ff.max(), [otu_refseq_ff.min(), otu_refseq_ff.max(),], False,
                taxa_refseq_ff.min(), taxa_refseq_ff.max(), [taxa_refseq_ff.min(), taxa_refseq_ff.max()], False,

                )


    return ( 1,10,[1,10], True,
            1,10,[1,10], True,
            1,10,[1,10], True,
            1,10,[1,10], True,)

#Callbacks for updating box plot and tables
@app.callback(
    [Output('feature-boxplot', 'figure'),
    Output('otu-gg-table', 'children'),
    Output('taxa-gg-table', 'children'),
    Output('otu-refseq-table', 'children'),
    Output('taxa-refseq-table', 'children'),],
    [Input('gg-otu-slider', 'value'),
    Input('gg-taxa-slider','value'),
    Input('refseq-otu-slider', 'value'),
    Input('refseq-taxa-slider', 'value'),
    State('summary-dataset-dropdown', 'value')]
)
def update_plot(gg_otu, gg_taxa, refseq_otu, refseq_taxa, dataname):
    if dataname:
        dataset = datasets[dataname]
        otu_gg_stats, otu_gg_ff = feature_stats(dataset['greengenes']['otu'], gg_otu[0], gg_otu[1])
        taxa_gg_stats, taxa_gg_ff = feature_stats(dataset['greengenes']['taxa'], gg_taxa[0], gg_taxa[1])
        otu_refseq_stats, otu_refseq_ff = feature_stats(dataset['refseq']['otu'], refseq_otu[0], refseq_otu[1])
        taxa_refseq_stats, taxa_refseq_ff = feature_stats(dataset['refseq']['taxa'], refseq_otu[0], refseq_otu[1])

        # get formatted.

        #Plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=otu_gg_ff, name='OTU Greengenes'))
        fig.add_trace(go.Box(y=taxa_gg_ff, name='Taxa Greengenes'))
        fig.add_trace(go.Box(y=otu_refseq_ff, name='OTU Refseq'))
        fig.add_trace(go.Box(y=taxa_refseq_ff, name='Taxa Refseq'))

        #tables
        table_formats = [(otu_gg_stats, '#636EFA'),
                         (taxa_gg_stats, '#EF553B'),
                         (otu_refseq_stats, '#00CC96'),
                         (taxa_refseq_stats, '#AB63FA')]

        tables = []
        for format in table_formats:
            tables.append( dash_table.DataTable(
                data = format[0].to_dict('records'),
                columns = [{'id':c, 'name':c} for c in format[0].columns],
                style_as_list_view=True,
                style_cell={'textAlign':'center'},
                style_header={
                    'backgroundColor': format[1],
                    'fontWeight': 'bold'
                },

            ))

        return fig, tables[0], tables[1], tables[2], tables[3]
    return {}, [], [], [], []



################################################################################
### Machine learning callbacks                                               ###
################################################################################

### Feature selection: Enable N-features when model is chosen ###
@app.callback(
    Output('n-features', 'disabled'),
    [Input('feature-selection', 'value')]
)
def enable_n_features(value):
    if value:
        return False
    return True

### Enable submit button ###
# If feature selection specificed, must have n-features specified.
# Model by itself is fine
@app.callback(
    Output('submit-button', 'disabled'),
    [Input('models-select', 'value'),
    Input('feature-selection', 'value'),
    Input('n-features', 'value')]
)
def enable_submit(model, select, n_features):
    if model:
        if not select:
            return False
        if not n_features:
            return True
        return False
    return True


### Dynamic param selection ###
# Display model params if they are selected.
@app.callback(
    [Output('svc-params', 'style'),
    Output('rf-params', 'style'),
    Output('nb-params', 'style')],
    [Input('models-select', 'value')]
)
def manage_param_fields(value):
    vals = [{'display': 'none'},{'display': 'none'},{'display': 'none'}]
    if value:
        if 'SVM' in value:
            vals[0] = {'display': 'block'}
        if "RF" in value:
            vals[1] = {'display':'block'}
        if 'NB' in value:
            vals[2] = {'display':'block'}
    return vals

### Compute Classification ###
# Compute classification, create the resultant figures, and serialize them.
@app.callback(
    Output('ml-graph-area', 'figure'),
    [Input('submit-button', 'n_clicks'),
    #models
    State('models-select', 'value'),
    #rf params
    State('rf-criterion', 'value'),
    State('rf-n-estimators', 'value'),
    #svm params
    State('svc-penalty', 'value'),
    State('svc-loss', 'value'),
    State('svc-c', 'value'),
    #NB params
    State('nb-alpha', 'value'),
    State('nb-prior', 'value'),
    #Feature selection params
    State('feature-selection', 'value'),
    State('n-features', 'value'),
    #Selected dataset and features
    State('summary-dataset-dropdown', 'value'),
    State('gg-otu-slider', 'value'),
    State('gg-taxa-slider', 'value'),
    State('refseq-otu-slider', 'value'),
    State('refseq-taxa-slider', 'value')]
)
def run_grid_search(click, models, rf_criterion, rf_n_estimators,
    svc_penalty, svc_loss, svc_c,
    nb_alpha, nb_prior,
    feature_selection, n_features,
    dataset_sel, gg_otu_ff, gg_taxa_ff, rf_otu_ff, rf_taxa_ff):
    #Leave graph empty if no submit button.
    if not click:
        return {}
    #cast types for nb_prior, which is currently JS typed.
    if nb_prior:
        nb_prior = [True if e=='true' else False for e in nb_prior]
    #Lookup dict to group inputs to their relevant models.
    input_dict = {'RF': {'criterion': rf_criterion,
                         'n_estimators':rf_n_estimators},
                  'SVM': {'penalty': svc_penalty,
                          'loss': svc_loss,
                          'C': svc_c,},
                  'NB': {'alpha': nb_alpha,
                         'prior': nb_prior,}}


    model_dict = {'RF': RandomForestClassifier,
                  'SVM': LinearSVC,
                  'NB': MultinomialNB}
    sel_models = {model: model_dict[model]() for model in models}
    #The param dict will contain all the default values for the selected models.
    params = {model: {k:[v] for k, v in sel_models[model].get_params().items()} for model in models}
    # If user specified params, replace the defaults
    for key in input_dict.keys():
        if key in params.keys():
            for param, value in input_dict[key].items():
                if value:
                    params[key][param] = value


    #Get the dataset.
    dataset = datasets[dataset_sel]
    # Get results for each feature representation.
    results = get_all_results(sel_models, params, dataset)
    #Send result data.
    bar_plot = bar_plot_best(results)

    #soon: make the params a thing.
    return bar_plot

################################################################################
### Page Navigation callbacks                                                ###
################################################################################
@app.callback(
    [Output('page-content', 'children'),
    Output('page-1-nav', 'className'),
    Output('page-2-nav', 'className'),
    Output('summary-dataset-dropdown', 'style'),
    Output('data-sliders', 'style')],
    [Input('url', 'pathname'),]
)
def display_page(pathname):
    if pathname == '/page-1':
        return page1_layout, 'active', '', {'display': 'block'}, {'display': 'block'}
    elif pathname == '/page-2':
        return page2_layout, '', 'active', {'display': 'none'}, {'display': 'none'}
    else:
        return html.H3('Nothing here'), '', '',  {'display': 'none'}, {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
