# Some components are reused in each app. Put here for easier code readability
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

def make_navbar(active=0):
    classnames = ['', '']
    classnames[active] = "active"

    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Data Summary", href="/page-1"), className=classnames[0]),
            dbc.NavItem(dbc.NavLink("Machine Learning", href="page-2"),className=classnames[1]),
        ],
        brand="Vizomics",
        brand_href="#",
        color="dark",
        dark=True,
    )
    return navbar

def make_dataset_dropdown(id):
        dropdown = dcc.Dropdown(
            id=id,
            options=[
                {'label': 'Gevers 2014: IBD vs Healthy, ileum', 'value': 'IBD_Gevers'},
                {'label': 'David 2014: Animal vs Plant Diet, stool', 'value': 'plant_v_animal'},
                {'label': 'Yatsunenko 2012: USA vs Malawi Adult, stool', 'value':'usa_vs_malawi'},
            ]
        )
        return dropdown
