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

layout = dbc.Container(fluid=True,children=[
    make_navbar(active=1),
    html.H3('App 2'),
    make_dataset_dropdown(id='ml-dataset-dropdown'),
    html.Div(id='app-2-display-value'),
    dcc.Link('Go to App 1', href='/apps/app1')
])


@app.callback(
    Output('app-2-display-value', 'children'),
    Input('ml-dataset-dropdown', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)
