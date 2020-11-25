import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

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
        dbc.Col(id='otu-gg-table'),
        dbc.Col(id='taxa-gg-table'),
        dbc.Col(id='otu-refseq-table'),
        dbc.Col(id='taxa-refseq-table'),
    ]),

])
### Machine Learning Layout ###
page2_layout = dbc.Container(fluid=True,children=[
    html.H3('App 2'),
    html.Div(id='app-2-display-value'),
    dcc.Link('Go to App 1', href='/apps/app1')
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

#Callbacks for updating box plot
@app.callback(
    Output('feature-boxplot', 'figure'),
    [Input('gg-otu-slider', 'value'),
    Input('gg-taxa-slider','value'),
    Input('refseq-otu-slider', 'value'),
    Input('refseq-taxa-slider', 'value'),
    State('summary-dataset-dropdown', 'value')]
)
def update_plot(gg_otu, gg_taxa, refseq_otu, refseq_taxa, dataname):
    if dataname:
        dataset = datasets[dataname]
        otu_gg_ff = get_feature_frequencies(dataset['greengenes']['otu'])
        taxa_gg_ff = get_feature_frequencies(dataset['greengenes']['taxa'])
        otu_refseq_ff = get_feature_frequencies(dataset['refseq']['otu'])
        taxa_refseq_ff = get_feature_frequencies(dataset['refseq']['taxa'])
        # get formatted.
        otu_gg_ff = otu_gg_ff[(otu_gg_ff > gg_otu[0])&(otu_gg_ff<gg_otu[1])]
        taxa_gg_ff = taxa_gg_ff[(taxa_gg_ff > gg_taxa[0])&(taxa_gg_ff<gg_taxa[1])]
        otu_refseq_ff = otu_refseq_ff[(otu_refseq_ff > refseq_otu[0])&(otu_refseq_ff < refseq_otu[1])]
        taxa_refseq_ff = taxa_refseq_ff[(taxa_refseq_ff > refseq_taxa[0])&(taxa_refseq_ff<refseq_taxa[1])]
        #Plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=otu_gg_ff, name='OTU Greengenes'))
        fig.add_trace(go.Box(y=taxa_gg_ff, name='Taxa Greengenes'))
        fig.add_trace(go.Box(y=otu_refseq_ff, name='OTU Refseq'))
        fig.add_trace(go.Box(y=taxa_refseq_ff, name='Taxa Refseq'))
        return fig
    return {}

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
