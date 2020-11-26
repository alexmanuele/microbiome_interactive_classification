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
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from components import *
from utils import *

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY],suppress_callback_exceptions=True)

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
    dcc.Store(id='graph-store'),

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

### Landing Page ###
#Adapted from https://getbootstrap.com/docs/4.0/examples/product/
landing_page_layout = [
    html.Div(className="position-relative overflow-hidden p-3 p-md-5 m-md-3 text-center bg-light",children=[
        html.Div(className="col-md-5 p-lg-5 mx-auto my-5", children=[
            html.H1('Vizomics', className="display-4 font-weight-normal"),
            html.P("Explore machine learning over microbiome datasets with no programming.", className='lead font-weight-normal'),
            html.A('Get Started', href='page-1', className='btn btn-outline-secondary'),
        ])
    ]),
    html.Div(className="d-md-flex flex-md-equal w-100 my-md-3 pl-md-3",children=[
        html.Div(className="bg-dark mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center text-white overflow-hidden",children=[
            html.Div(className="my-3 py-3", children=[
                html.H2('Explore several datasets.', className='display-5'),
                html.P(children=["Select one of multiple amplicon sequence datasets from Knight et al.'s ",
                                 html.A("Machine Learning for Microbiome Repo.", href="https://knights-lab.github.io/MLRepo/", className='font-weight-bold text-white'),
                                 ], className='lead'),
            ])
        ]),
        html.Div(className="bg-light mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden",children=[
            html.Div(className="my-3 py-3", children=[
                html.H2("Visually remove outliers.", className="display-5"),
                html.P('Interactive preprocessing allows you to filter out features based on their frequency. See in real time how changes in frequency bounds affect data distributions.', className='lead'),

            ])
        ])
    ]),
    html.Div(className="d-md-flex flex-md-equal w-100 my-md-3 pl-md-3",children=[
        html.Div(className="bg-light mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden",children=[
            html.Div(className="my-3 py-3", children=[
                html.H2('Compare feature representations.', className='display-5'),
                html.P("Investigate the impact of reference database and taxonomic assignment. Automatically find results for both OTUs and Taxa picked from Greengenes and Refseq.", className='lead'),
            ])
        ]),
        html.Div(className="bg-primary mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center text-white overflow-hidden",children=[
            html.Div(className="my-3 py-3", children=[
                html.H2("Automatically find the best model.", className="display-5"),
                html.P('Use the UI to specify which models and parameters to test. Explore model performances for each feature representation using both tables and graphs.', className='lead'),

            ])
        ])
    ]),

]



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
        dbc.Col(children=[
            dbc.Tabs(
                [
                    dbc.Tab(label="Best", tab_id='bar'),
                    dbc.Tab(label='All', tab_id='scatter'),
                    #    dcc.Graph(id='ml-graph-area')],
                ],
                id='tabs',
                active_tab='bar'
            ),
            html.Div(id='tab-content')
            ],
            width=8),
        #Machine Learning Params Area
        dbc.Col(children=[
            dbc.Card([
                dbc.CardHeader('Select Parameters for GridSearch', className='bg-secondary text-white'),
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
                                {'label': "Support Vector Machine", 'value': 'SVM'}
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
                                        [dcc.Dropdown(id='svc-kernel',
                                            options=[
                                                {'label': 'Linear', 'value':'linear'},
                                                {'label': 'Polynomial', 'value': 'poly'},
                                                {'label':'Radial Basis Function', 'value':'rbf'},
                                                {'label':'Sigmoid', 'value': 'sigmoid'}, ],
                                            multi=True,
                                            placeholder='Kernel'
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
                dbc.Button("Run GridSearch", id='submit-button', color="danger", disabled=True),
            ]), #<!-- /card -->
        ], width=4), #<!-- /col -->
    ]),
    dbc.Row(id='result-table-area'),
]),



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
                    'fontWeight': 'bold',
                    'color': 'white',
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
    Input('n-features', 'value'),
    State('summary-dataset-dropdown', 'value')]
)
def enable_submit(model, select, n_features, dataset):
    if not dataset:
        return True
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

### Tab navigation ###
#...navigates tabs.
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
    Input('graph-store', 'data')]
)
def render_tab_content(active_tab, data):
    # render stored graphs in tabs.
    if active_tab and data is not None:
        if active_tab == "bar":
            return dcc.Graph(figure=data['bar'])
        elif active_tab == 'scatter':
            return dcc.Graph(figure=data['scatter'])
    return "No plot selected."
### Compute Classification ###
# Compute classification, create the resultant figures, and serialize them.
@app.callback(
    [#Output('ml-graph-area', 'figure'),
    Output('graph-store', 'data'),
    Output('result-table-area', 'children')],
    [Input('submit-button', 'n_clicks'),
    #models
    State('models-select', 'value'),
    #rf params
    State('rf-criterion', 'value'),
    State('rf-n-estimators', 'value'),
    #svm params
    State('svc-kernel', 'value'),
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
    svc_kernel, svc_c,
    nb_alpha, nb_prior,
    feature_selection, n_features,
    dataset_sel, gg_otu_ff, gg_taxa_ff, rf_otu_ff, rf_taxa_ff):
    #Leave graph empty if no submit button.
    if not click:
        return {k: go.Figure(data=[]) for k in ['bar', 'scatter']}, []
    #cast types for nb_prior, which is currently JS typed.
    if nb_prior:
        nb_prior = [True if e=='true' else False for e in nb_prior]
    #Lookup dict to group inputs to their relevant models.
    input_dict = {'RF': {'criterion': rf_criterion,
                         'n_estimators':rf_n_estimators},
                  'SVM': {'kernel': svc_kernel,
                          'C': svc_c,},
                  'NB': {'alpha': nb_alpha,
                         'fit_prior': nb_prior,}}


    model_dict = {'RF': RandomForestClassifier,
                  'SVM': SVC,
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
    #Filter features based on selections from previous page
    dataset = filter_dataset(dataset, gg_otu_ff, gg_taxa_ff, rf_otu_ff, rf_taxa_ff,
    feature_selection, n_features)
    # Get results for each feature representation.
    results = get_all_results(sel_models, params, dataset)
    #Make tables of the Results
    frames = []
    for key in results.keys():
        df = results[key].score_summary()
        df.insert(0, column='Representation', value=[key]*df.shape[0])
        frames.append(df)
    result_df = pd.concat(frames)
    table = dash_table.DataTable(
        data=result_df.to_dict('records'),
        columns = [{'id':c, 'name':c} for c in result_df.columns],
        sort_action="native",
        sort_mode='multi',
        style_as_list_view = True,
        style_cell={'textAlign': 'left',
                    'height': 'auto',
                    'color': 'white',
        },
        style_header={'fontWeight': 'bold',
                      'color': 'black',
        },
        style_data_conditional=[
            {
                'if':{
                    'filter_query': '{Representation} = "greengenes_otu"',
                },
                'backgroundColor':  ' #636EFA',
            },
            {
                'if':{
                    'filter_query': '{Representation} = "greengenes_taxa"',
                },
                'backgroundColor':  ' #EF553B',
            },
            {
                'if':{
                    'filter_query': '{Representation} = "refseq_otu"',
                },
                'backgroundColor':  ' #00CC96',
            },
            {
                'if':{
                    'filter_query': '{Representation} = "refseq_taxa"',
                },
                'backgroundColor':  ' #AB63FA',
            },
            {
                'if':{
                    'state': 'selected'
                },
                'color':'black',
            }
            ]
    )
    #Send result data.
    bar = bar_plot_best(results)
    scatter = scatter_plot(results)
    #soon: make the params a thing.
    return {'bar':bar, 'scatter':scatter}, table

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
        return landing_page_layout, '', '',  {'display': 'none'}, {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
