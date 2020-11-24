import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from app import app
from components import *
from utils import *

#Update here if you add more datasets.
data_paths = ['IBD_Gevers', 'plant_v_animal', 'usa_vs_malawi']
datasets = {path: load_dataset(path) for path in data_paths}

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
layout = dbc.Container(fluid=True,children=[
    make_navbar(active=0),
    html.H3('App 1'),
    make_dataset_dropdown(id='summary-dataset-dropdown'),
    html.Div(id='app-1-display-value'),
    dbc.Row([
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
])

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
    Output('refseq-taxa-slider', 'disabled')],
    [Input('summary-dataset-dropdown', 'value')]
)
def update_dataset(value):
    if value:
        dataset = datasets[value]
        otu_gg_ff = get_feature_frequencies(dataset['greengenes']['otu'])
        taxa_gg_ff = get_feature_frequencies(dataset['greengenes']['taxa'])
        otu_refseq_ff = get_feature_frequencies(dataset['refseq']['otu'])
        taxa_refseq_ff = get_feature_frequencies(dataset['refseq']['taxa'])
        """
        fig = go.Figure()
        fig.add_trace(go.Box(y=otu_gg_ff, name='OTU Greengenes'))
        fig.add_trace(go.Box(y=taxa_gg_ff, name='Taxa Greengenes'))
        fig.add_trace(go.Box(y=otu_refseq_ff, name='OTU Refseq'))
        fig.add_trace(go.Box(y=taxa_refseq_ff, name='Taxa Refseq'))
        """

        return (otu_gg_ff.min(), otu_gg_ff.max(), [otu_gg_ff.min(), otu_gg_ff.max(),], False,
                taxa_gg_ff.min(), taxa_gg_ff.max(), [taxa_gg_ff.min(), taxa_gg_ff.max(),], False,
                otu_refseq_ff.min(), otu_refseq_ff.max(), [otu_refseq_ff.min(), otu_refseq_ff.max(),], False,
                taxa_refseq_ff.min(), taxa_refseq_ff.max(), [taxa_refseq_ff.min(), taxa_refseq_ff.max()], False
                )

    return ( 1,10,[1,10], True,
            1,10,[1,10], True,
            1,10,[1,10], True,
            1,10,[1,10], True)

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
