# app.py

import base64
import datetime
import io
import pandas as pd
import numpy as np
import json
import os
import ast
import pyvinecopulib as pv
from lib.evaluation import empcdf_tail, qqplot, mahalanobis_distance1, count_similar_variables
from lib.preprocessing import PreProcessing
from lib.vinecopsearch import VinecopSearch, plot_structure, all_metric_, u_to_m
from lib.mutual_info import mutual_info, mutual_info_ksg, mutual_info_pairs
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pylab import rcParams
from scipy.stats import ks_2samp

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_interactive_graphviz
import dash_cytoscape as cyto
from networkx.readwrite import json_graph
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)


import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL], suppress_callback_exceptions=True)

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Vinecop App", className="display-4"),
        html.Hr("Contents"),
        html.P(
            "", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Upload your data", href="/page-1", active="exact"),
                dbc.NavLink("Data Exploration", href="/page-2", active="exact"),
                dbc.NavLink("Find structure", href="/page-3", active="exact"),
                dbc.NavLink("Model evaluation", href="/page-4", active="exact"),
                dbc.NavLink("Generate samples and transform", href="/page-5", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        # html.Div([
        # dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        #     dcc.Tab(label='Tab One', value='tab-1-example'),
        #     dcc.Tab(label='Tab Two', value='tab-2-example'),
        # ],  vertical=True, parent_style={'float': 'left'}),
        # html.Div(id='tabs-content-example', style={'float': 'left', 'width': '400'})
    # ])
    ],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    dcc.Store(id='intermediate-value', storage_type='session'), # raw df
    dcc.Store(id='intermediate-value-1', storage_type='session'), # pseudo obs df
    
    dcc.Store(id='intermediate-value-2', storage_type='session'), # json model structure
    dcc.Store(id='t_max', storage_type='session'), # t max
    
    dcc.Store(id='intermediate-value-3', storage_type='session'), # simulated data
    dcc.Store(id='intermediate-value-4', storage_type='session'), # observed data
    dcc.Store(id='intermediate-value-5', storage_type='session'), # model json file
    dcc.Store(id='intermediate-value-6', storage_type='session'), # simulated data to m space
    html.Div(id='page-content')
])
layout_home = html.Div([
    html.H1('Vine copula search dashboard',
                        style={'textAlign':'center'}),
    html.H2('Contents'),
    html.P('This application allows the user to explore Vine copula based on data input.'),
    html.Br(),
    dcc.Markdown('''
                 It includes :
                * Uploading your data with several CSV file and do preprocessing on it
                * For given set of columns and paremeters, find the structure based on Kendall's tau value
                * Given the structure, interactive plot the graph of a given level of tree with information about the tree
    ''')

    
                
                    
                        
            
], style=CONTENT_STYLE)

layout_page_1 = html.Div([
    html.H1('Before starting'),
                html.H2('Upload your data'),
                html.Div([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Not Allow multiple files to be uploaded
                            multiple=True,
                            
                        ),
                        html.Button('Submit files', id='submit-file', n_clicks=0),
                        html.Button("Trigger button", id="trigger_hidden", n_clicks=0, style = { "display" : "none"}),
                        dcc.Loading(
                            id="loading-1",
                            children=[html.Div([html.Div(id="output-data-upload")])],
                            type="circle",
                        ),
                        dcc.Loading(
                            id="loading-2",
                            children=[html.Div([html.Div(id="output-data-obs")])],
                            type="circle",
                        )
                    
                        
                    ])
], style=CONTENT_STYLE)


layout_page_3 = html.Div([
    html.H1('Search structure for the dataset'),
    html.Hr(),
    html.Div([
        html.H3("Select columns"),
        dcc.Dropdown(id='columns-dropdown', options = [], value = [],  multi=True, persistence=True),
        html.Button("SELECT ALL", id="select-all", n_clicks=0),
        html.H3("Select metric"),
        dcc.RadioItems(
                id = 'metric',
                options=[
                    {'label': "Kendall's tau", 'value': "tau"},
                    {'label': 'Mutual information', 'value': "mi"},
                ],
                value = "tau"
        ),
        html.Div(id = "bins_box", children = [dcc.Input(id = "bins_input",
                                                            type='number',
                                                            value=1,
                                                            min = 1,
                                                            step=1)]),
        html.Button("Click to plot", id="button_plot_metric", n_clicks=0),
        html.Hr(),
        html.H4("Plot the sorted values of the chosen metric with the chosen columns "),
        dcc.Loading(
                    id="loading-1",
                    children=[html.Div(id = "plot_metric", children = [dcc.Graph(figure={})])],
                    type="circle",
                    ),
        html.Hr(),
        html.H3("Select parameters"),
        html.H5("Bicop"),
        dcc.RadioItems(
                id = 'bicop',
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False},
                ],
                value = False
        ),
        html.H5("Threshold"),
        dcc.RadioItems(
                id = 'threshold',
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False},
                ],
                value = True
        ),
        html.Div(id = "threshold_box", children = [dcc.Input(id = "threshold_input",
                                                            type='number',
                                                            value=0,
                                                            step=0.001,
                                                            min = 0,
                                                            style={'float': 'left'})]),
        html.Div(id = 'level', children = [
                html.H5("Level of tree to prune"),
                dcc.RadioItems(
                        id = 'level_input',
                        options=[
                            {'label': 'all', 'value': 'all'},
                            {'label': '0', 'value': '0'},
                        ],
                        value = 'all'
        ),
        ]),
        html.Button("Search structure", id="search_structure", n_clicks=0),
        dcc.Loading(
                    id="loading-2",
                    children=[html.Div(id = "alert_search_created")],
                    type="circle",
                    ),
        html.Div([
            html.Hr(),
            html.Button("Download the model structure to JSON file", id="btn_json"),
            dcc.Download(id="download-json-model"),
            html.Button("Download CSV", id="btn_data_cut"),
            dcc.Download(id="download-data-cut-model"),
            
            html.H2("Choose level of tree to print"),
            html.Div([
            dcc.Dropdown(id='level-tree-dropdown', options = [], value = [], multi= False, persistence=True),
            html.Button("Trigger button", id="trigger_hidden", n_clicks=0, style = { "display" : "none"}),
            dcc.Loading(
                    id="loading-3",
                    children=[html.Div(id='output-print-tree')],
                    type="circle",
                    ),
            ])])
    ])

], style=CONTENT_STYLE)

layout_page_2 = html.Div([
    html.H1('Bivariate plot'),
    html.H2('Scartter plot with marginal histogram'),
    html.H3("Select first column"),
    dcc.Dropdown(id='columns-dropdown-scatter-1', options = [], value = [],  multi=False, persistence=True),
    html.H3("Select second column"),
    dcc.Dropdown(id='columns-dropdown-scatter-2', options = [], value = [],  multi=False, persistence=True),
    html.Hr(),
    dcc.Loading(
                id="loading-1",
                children=[html.Div(id ='output_scatter_plot')],
                type="circle",
                ),

], style=CONTENT_STYLE)

layout_page_4 = html.Div([
    html.H1('Model evaluation'),
    dcc.Markdown('''
                 We want to assess whether it is reliable and adequate for future use, i.e., evaluate it in absolute terms.

                Model evaluation therefore proceeds by comparing characteristics of the observed data, which was used for model specification, with simulated observations from the specified R-vine model
                 '''),
    dcc.Tabs(id="tabs-model-evaluation", value='tab-1', children=[
        dcc.Tab(label='Upload observation and model', value='tab-1'),
        dcc.Tab(label='Bivariate plot', value='tab-1-5'),
        dcc.Tab(label='Empirical copula distribution', value='tab-2'),
        dcc.Tab(label='General QQ-plot', value='tab-3'),
        dcc.Tab(label='Mean of copula', value='tab-4'),
        dcc.Tab(label='Mutual Information', value='tab-5'),
        dcc.Tab(label='Mahalanobis distance & KS-test', value='tab-6'),
    ]),
    html.Div(id='tabs-content-model-evaluation')

], style=CONTENT_STYLE)

layout_page_5 = html.Div([
    html.H1('Simulation and transformation'),
    dcc.Markdown('''
                 Simulate samples from model and transform it from copula space to real space
                 '''),
    html.H2('Upload observation and model'),
    html.H2('Upload your model'),
    dcc.Upload(
                            id='upload-model-page5',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select model file')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Not Allow multiple files to be uploaded
                            multiple=False,
                            
                        ),
    html.H2('Upload your real observation data'),
    dcc.Upload(
                            id='upload-data-page5',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select your data file')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Not Allow multiple files to be uploaded
                            multiple=False,
                            
                        ),
    html.Button('Submit files', id='submit-file-sample', n_clicks=0),
    dcc.Loading(
                                id="loading-1",
                                children=[html.Div(id='alert_submit_files_sample')],
                                type="circle",
                        ),
    html.Hr(),
    html.H2("Simulate samples"),
    html.H3("Select number of samples"),
    dcc.Input(id = "bins_input_samples",
                        type='number',
                        value=1,
                        min = 1,
                        step=1),
    dcc.Loading(
                id="loading-2",
                children=[html.Div(id ='output_samples')],
                type="circle",
                ),
    html.Button("Generate samples", id="btn_submit_sample", n_clicks=0),
    html.Br(),
    html.Button("Download CSV", id="btn_data_sample", n_clicks=0),
    dcc.Download(id="download-data-sample"),
    html.Hr(),
    html.H2("Transform samples to real space"),
    html.Button("Transform samples", id="btn_submit_transformation", n_clicks=0),
    dcc.Loading(
                id="loading-3",
                children=[html.Div(id ='output_transformation')],
                type="circle",
                ),
    html.Button("Download CSV", id="btn_data_transformation", n_clicks=0),
    dcc.Download(id="download-data-transformation"),

], style=CONTENT_STYLE)

# "complete" layout
app.validation_layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    layout_page_1,
    layout_page_2,
    layout_page_3,
    layout_page_4,
    layout_page_5
])

# Index callbacks
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/":
        return layout_home
    elif pathname == "/page-1":
        return layout_page_1
    elif pathname == "/page-2":
        return layout_page_2
    elif pathname == "/page-3":
        return layout_page_3
    elif pathname == "/page-4":
        return layout_page_4
    elif pathname == "/page-5":
        return layout_page_5
    else:
        return layout_home

# @app.callback(Output('tabs-content-example', 'children'),
#               [Input('tabs-example', 'value')])
# def render_content(tab):
#     if tab == 'tab-1-example':
#         return layout_home
#     if tab == 'tab-2-example' :
#         return layout_page_1

################################################################################################################################################################################################################################
############# PAGE 1 ######################
 
# UPLOAD A CSV FILE

def read_csv(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


def read_json(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'json' in filename:
            # Assume that the user uploaded a JSON file
            model = decoded.decode('utf-8')
            
            return model
        else : 
            print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

# Home upload file callback
@app.callback(Output('output-data-upload', 'children'),
              Input('intermediate-value', 'data'),
              Input('trigger_hidden', 'n_clicks'), prevent_initial_update = False
              )
def update_output(df, n_clicks):
    df = pd.DataFrame.from_dict(df)
    if not df.empty:
        return  html.Div([
            html.H5("Raw data visualization"),
            dash_table.DataTable(
                df.to_dict(orient = 'records'),
                [{'name': i, 'id': i, 'deletable': True} for i in df.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action='native',
                page_current= 0,
                page_size= 10,
                style_table={'overflowX': 'auto'},
                persistence=True,
                persisted_props=["data"]
            ),

            html.Hr(),
            html.Button('Click to transform data to pseudo observation', id='submit-pseudo-obs', n_clicks=0),
        ])

@app.callback(Output('intermediate-value', 'data'),
              # Output('submit-file', 'n_clicks'),
              Input('submit-file', 'n_clicks'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'), prevent_initial_call=True)
def update_table(n_clicks, list_of_contents, list_of_names):
    if (n_clicks == 0): 
        raise PreventUpdate
    if n_clicks > 0 :
        if list_of_contents is not None:
            children = [
                read_csv(c, n) for c, n in
                zip(list_of_contents, list_of_names)]
            return pd.concat(children).to_dict("records") #, 0
    
# Preprocessing
@app.callback(
    Output('intermediate-value-1', 'data'),
    # Output('submit-pseudo-obs', 'n_clicks'),
    Input('submit-pseudo-obs', 'n_clicks'),
    State('intermediate-value', 'data'), prevent_initial_call=True)
def preprocessing(n_clicks, data):
    if (n_clicks == 0): 
        raise PreventUpdate
    if n_clicks > 0 :
        data = pd.DataFrame.from_dict(data)
        pre = PreProcessing()
        # Cleaning
        try : 
            data = pre.convert_time(data, "time")
            data = pre.drop(data,["Unnamed: 0", "time"])
        except :
            data = pre.drop(data,["Unnamed: 0"])
        data = pre.remove_constant(data)
        return pre.pobs(data).to_dict('records') #, 0

# Update table showed for pseudo obs

@app.callback(Output('output-data-obs', 'children'),
              Input('intermediate-value-1', 'data'))
def update_output_obs(df):
    df = pd.DataFrame.from_dict(df)
    if not df.empty:
        return  html.Div([
            html.H5("Pseudo obs data visualization"),
            dash_table.DataTable(
                df.to_dict('records'),
                [{'name': i, 'id': i, 'deletable': True} for i in df.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action='native',
                page_current= 0,
                page_size= 10,
                style_table={'overflowX': 'auto'},
                persistence=True,
                persisted_props=["data"]
            ),

            html.Hr(),
            html.Button("Download CSV", id="btn_csv"),
            dcc.Download(id="download-dataframe-csv"),
        ])
    
# Home download file transformed
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State("intermediate-value-1", "data"),
    prevent_initial_call=True
)
def func_df(n_clicks, df):
    return dcc.send_data_frame(pd.DataFrame.from_dict(df).to_csv, "pseudo-obs.csv")

##################################################################################################################################################################################################################
############ PAGE 2 ##############

####### SCATTER PLOT ###########
@app.callback(
    Output("columns-dropdown-scatter-1", "options"),
    Output("columns-dropdown-scatter-2", "options"),
    Input("intermediate-value-1", "data")
)
def display_columns_to_choose_scatter(df):
    df = pd.DataFrame.from_dict(df)
    col = df.columns
    return col, col

@app.callback(
    Output("output_scatter_plot", "children"),
    Input("columns-dropdown-scatter-1", "value"),
    Input("columns-dropdown-scatter-2", "value"),
    State("intermediate-value-1", "data"), prevent_initial_call=True
)
def scatter_plot(col1, col2, df):
    df = pd.DataFrame.from_dict(df)
    if not df.empty :
        fig = px.scatter(df, x=col1, y= col2, marginal_x="histogram", marginal_y="histogram", title = "Scartter plot")
        return dcc.Graph(figure=fig)
    else :
        raise PreventUpdate

####### END SCATTER PLOT ##########

##################################################################################################################################################################################################################
############ PAGE 3 ##############

@app.callback(
    Output("columns-dropdown", "options"),
    Input("intermediate-value-1", "data"),
)
def display_columns_to_choose(df):
    df = pd.DataFrame.from_dict(df)
    col = df.columns
    return col

@app.callback(Output("columns-dropdown", "value"), Input("select-all", "n_clicks"),
              State("intermediate-value-1", "data"))
def select_all(n_clicks, df):
    df = pd.DataFrame.from_dict(df)
    col = df.columns
    return [value for value in col]

# Show or hide bins input depending if metric = "mi" or "tau"
@app.callback(Output('bins_input', 'style'), Input('metric', 'value'))
def bins_container(toggle_value):
    if toggle_value == "tau":
        return {'display': 'none'}
    else:
        return {'display': 'block'}

# Plot sorted value of the metric
@app.callback(Output('plot_metric', 'children'),
              # Output("button_plot_metric", "n_clicks"),
              Input("button_plot_metric", "n_clicks"),
              State("columns-dropdown", "value"),
              State('metric', 'value'),
              State('bins_input', 'value'),
              State("intermediate-value-1", "data"))
def plot_metric(n_clicks, columns, metric, bins, df):
    if (n_clicks == 0): 
        raise PreventUpdate
    if n_clicks > 0 :
        df = pd.DataFrame.from_dict(df)
        df = df[columns]
        nb_columns = len(columns)
        metrics = all_metric_(nb_columns, df.to_numpy(), metric, bins)
        metrics = np.sort(np.abs(metrics))
        metrics = pd.DataFrame({"n" : range(len(metrics)), "metric" : metrics})
        fig = px.scatter(metrics, x="n", y="metric")
        
        return dcc.Graph(figure=fig) #, 0
    
# Show or hide threshold input depending if threshold = true or not
@app.callback(Output('threshold_input', 'style'), Output('level', 'style'), Input('threshold', 'value'))
def threshold_container(toggle_value):
    if toggle_value == False:
        return {'display': 'none'}, {'display': 'none'}
    else:
        return {'display': 'block'}, {'display': 'block'}
    
# Search structure tree

@app.callback(
    Output("alert_search_created", "children"),
    Output("t_max", "data"),
    # Output("search_structure", "n_clicks"),
    Input("search_structure", "n_clicks"),
    State("intermediate-value-1", "data"),
    State("bicop", "value"),
    State("threshold", "value"),
    State("threshold_input", "value"),
    State("level_input", "value"),
    State("columns-dropdown", "value"), prevent_initial_call = True
)
def search_structure(n_clicks, df, bicop, threshold, threshold_value, level, columns):
    if n_clicks > 0 : 
        df = pd.DataFrame.from_dict(df)[columns]
        mymodel = VinecopSearch(len(columns))
        mymodel.main(df.to_numpy(), bicop = bicop, threshold=threshold, threshold_value= float(threshold_value), level = level)
        t_max = mymodel.t_max
        mymodel.to_json('./output/model_structure.json')
        return html.P("Structure of the model created"), t_max #, 0
    else :
        raise PreventUpdate

# Page 3 download file model json
@app.callback(
    Output("download-json-model", "data"),
    Input("btn_json", "n_clicks"),
    prevent_initial_call=True
)
def func_json(n_clicks):
    return dcc.send_file('./output/model_structure.json')

# Download data with selected columns

@app.callback(
    Output("download-data-cut-model", "data"),
    Input("btn_data_cut", "n_clicks"),
    State('intermediate-value-1', 'data'),
    State('columns-dropdown', 'value'),
    prevent_initial_call=True
)
def func_data_cut(n_clicks, df, columns):
    return dcc.send_data_frame(pd.DataFrame.from_dict(df)[columns].to_csv, "pseudo-obs-cut.csv")

@app.callback(
    Output("level-tree-dropdown", "options"),
    Output("level-tree-dropdown", "value"),
    Input("t_max", "data"),
    Input("trigger_hidden", "n_clicks"), prevent_initial_call=False
)
def tree_option(t_max, n_clicks):
    if t_max != None:
        return [{"label":str(i),"value":str(i)} for i in range(0,t_max)], "0"
    else :
        raise PreventUpdate

@app.callback(
    Output("intermediate-value-2", "data"),
    Input("level-tree-dropdown", "value")
)
def update_output_tree(t):
    if type(t) != list :
        mymodel = VinecopSearch(filename = './output/model_structure.json')
        t = int(t)
        G, paths,weight = plot_structure(mymodel,t)
        #G = nx.nx_agraph.to_agraph(G).to_undirected().to_string()
        graph_dict = nx.cytoscape_data(G.to_undirected())
        return graph_dict
    else :
        raise PreventUpdate

@app.callback(
    Output("output-print-tree", "children"),
    Input("intermediate-value-2", "data")
)
def update_output_tree(G):
    if G != None :
        return html.Div([
        html.P("Dash Cytoscape:"),
        cyto.Cytoscape(
            id='cytoscape',
            elements=G['elements'],
            layout={'name': 'breadthfirst'},
            stylesheet=[
            # Group selectors
            {
                'selector': 'node',
                'style': {
                    'content': 'data(name)',
                    'text-halign':'center',
                    'text-valign':'center',
                    'font-size' : '20px',
                    "background-color": "#AFDDFF",
                    'width':'70px',
                    'height':'label',
                    'shape':'square',
                    "text-wrap": "ellipsis",
                    "text-max-width": 70,
                    "overflow": "hidden",
                }
            },
            ],
            style={'width': '100%', 'height': '700px'}
        ),
        html.H3("Selected node"),
        html.Div(id = "print-information-node"),
        html.H3("Selected edge"),
        html.Div(id = "print-information-edge")
    ])
    else :
        raise PreventUpdate
    # return html.Div([
    #     dash_interactive_graphviz.DashInteractiveGraphviz(
    #     id="graph",
    #     dot_source=G
    #     ),
    #     html.H3("Selected node"),
    #     html.Div(id = "print-information-node"),
    #     html.H3("Selected edge"),
    #     html.Div(id = "print-information-edge")
    # ])

@app.callback(
    Output('print-information-node', 'children' ),
    Input('cytoscape', 'tapNodeData'),
    State("level-tree-dropdown", "value"),
    State("columns-dropdown", "value")
)
def change_my_view_node(selected, t, columns):
    if selected != None :
        # Do something with selected
        selected = selected["value"]
        t = int(t)
        if t == 0:
            to_print = html.P(str(selected) + ": " + columns[int(selected)-1])
        elif t== 1:
            a,b = selected.split(',',1)
            to_print = html.P([a + ": " + columns[int(a)-1], html.Br() , b + ": " + columns[int(b)-1]])
        else :
            a,c = selected.split(',',1)
            b,c = c.split(';',1)
            c = ast.literal_eval(c)
            tmp = [a + ": " + columns[int(a)-1], html.Br() , b + ": " + columns[int(b)-1], html.Br(), html.Br()]
            tmp.append("Conditionning variables : ")
            tmp.append(html.Br())
            for element in c :
                tmp.append(str(element) + ": " + columns[int(element)-1])
                tmp.append(html.Br())
            to_print = html.P(tmp)
        return to_print
    else :
        raise PreventUpdate

@app.callback(
    Output('print-information-edge', 'children' ),
    Input('cytoscape', 'tapEdgeData'))
def change_my_view_edge(selected):
    if selected != None :
        # Do something with selected
        return html.P("Weight : " + str(selected['weight']))
    else :
        raise PreventUpdate
    
############ PAGE 4 ###################
@app.callback(
    Output('tabs-content-model-evaluation', 'children' ),
    Input('tabs-model-evaluation', 'value'))
def tab_content_model_evaluation(value):
    if value == "tab-1":
        to_print = html.Div([
                        html.H2('Upload your model'),
                        dcc.Upload(
                            id='upload-model',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select model file')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Not Allow multiple files to be uploaded
                            multiple=False,
                            
                        ),
                        html.H2('Upload your pseudo observation data'),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select your data file')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Not Allow multiple files to be uploaded
                            multiple=False,
                            
                        ),
                        html.Button('Submit files and simulate samples', id='submit-file-tab-1', n_clicks=0),
                        dcc.Loading(
                                id="loading-1",
                                children=[html.Div(id='alert_submit_files')],
                                type="circle",
                        ),
                        
                ])
        return to_print
    elif value == "tab-2":
        return html.Div([
            html.H3("Description"),
            dcc.Markdown(r'''
                         If one is particularly interested in an accurate modeling of the joint tail behavior of variables, it might be interesting to consider the empirical copula distribution functions in the tails,

                        $$C_n(u_1, ..., u_d) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{U_{i1} ≤u1,..., U_{id} ≤ ud}$$
                        
                        the pseudo-samples $U_1, ..., U_n$. These can be considered as samples from the underlying copula C.
                        
                        In this case it's interesting to look at bivariate one
                        So we compute $C_n(\alpha, \alpha)$ (lower tail) and $C_n(1-\alpha, 1-\alpha)$  (upper tail) for $\alpha \in [0, 0.1]$
                         ''' , mathjax=True),
            html.H3("Select 2 columns"),
            dcc.Dropdown(id='columns-dropdown-tab-2', options = [], value = [],  multi=True, persistence=True),
            dcc.Loading(
                                id="loading-1",
                                children=[html.Div(id='tail-plot')],
                                type="circle",
                        ),
            
        ])
    elif value == "tab-3":
        return html.Div([
            html.H3("Description"),
            dcc.Markdown(r'''
                        General QQ plots are used to assess the similarity of the distributions of two datasets. 

                        Such plots allow to compare two empirical cdf's, in our case the empirical cdf's of the observed and simulated data, where the observed data comes from the unknown true distribution and we want to investigate if the distribution induced by model(K) is close to this true distribution. Hence we compute :
                        
                        $w_i^K = \frac{1}{n}\sum_{j = 1}^n \mathbf{1}_{u_{jr}^K \leq u_{is}^K, u_{js}^K \leq u_{ir}^K}$ and and  $w_i = \frac{1}{n}\sum_{j = 1}^n \mathbf{1}_{u_{jr} \leq u_{is}, u_{js} \leq u_{ir}}$ for $i \in [|1,n|]$ and $r,s \in [|1,d|]$

                        $w_i^K$ corresponding to the truncated one so the simulated one''' , mathjax=True),
            html.H3("Select 2 columns"),
            dcc.Dropdown(id='columns-dropdown-tab-3', options = [], value = [],  multi=True, persistence=True),
            dcc.Loading(
                                id="loading-2",
                                children=[html.Div(id='qq-plot')],
                                type="circle",
                        ),
            
        ])
    elif value == "tab-4":
        return html.Div([
            html.H3("Description"),
            dcc.Markdown(r'''
                        We need alternative quantities. The most commonly used one is given by the mean of the copula data over its d components
                        $S_i^K = \frac{1}{d}\sum_{r=1}^d u_{ir}^K$ , and $S_i = \frac{1}{d}\sum_{r=1}^d u_{ir}$
                        for all $i \in [|1,n|]$ where n is the number of samples

                        The appropriateness of model(K) can then be assessed by comparing
                        histograms and empirical quantiles based on set of $\{S^K_i, i = 1,...,n\}$ and $\{S_i, i = 1,...,n\}$

                        We can extand the formula by adding personalize weight for each dimension''' , mathjax=True),
            dcc.Loading(
                                id="loading-1",
                                children=[html.Div(id='mean-copula-plot')],
                                type="circle",
                        ),
            
        ])
    elif value == "tab-5":
        return html.Div([
            html.H3("Description"),
            dcc.Markdown(r'''
                        We use different type of mutual information :
                        
                        1. Copula MI based on the copula model
                        
                        2. KSG MI based on the observed data
                        
                        3. Pairwise MI based on the observed data''' , mathjax=True),
            html.H3("Copula MI"),
            dcc.Markdown(r'''
                        We define the mutual information as

                        $I(x) = \int_{x}c(u_x)logc(u_x)du_x$


                        We use the method pdf directly that compute it s density and estimate the integral using Monte Carlo estimation

                        The mutual information (in bits) is 1 when two parties (statistically) share one bit of information.

                        Per default we will use the natural logarithm.''' , mathjax=True),
            dcc.Loading(
                                id="loading-1",
                                children=[html.Div(id='copula-mi-results', children = [html.Div([html.P(html.B("Mutual information of the vine copula model : "), style={'textAlign':'center'})])])],
                                type="circle",
                        ),
            html.H3("KSG MI"),
             dcc.Markdown(r'''
                        KSG estimator is KNN based.
                        
                        Let k be the number of neighbours.
                        
                        $I_{KSG,k} = (d-1)\psi(N) +  \psi(k) -(d-1)/k - \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^d \psi(n_{x_j}(i))$
                        where $n_{x_j}$ is the number of points at a distance less than or equal to $\epsilon_{i,k}^{x_j}/2$ in the $x_j$ subspace and $\psi$ the digamma function''' , mathjax=True),
            html.H4("Choose a number of neighbours"),
            dcc.Input(id = "bins_input_tab_4",
                        type='number',
                        value=2,
                        min = 2,
                        step=1),
            dcc.Loading(
                                id="loading-2",
                                children=[html.Div(id='ksg-mi-results', children = [html.Div([html.P(html.B("The KSG MI estimator of the observed data : "), style={'textAlign':'center'})])] )],
                                type="circle",
                        ),
            html.H3("Pairwise MI"),
            html.H4("Choose a bin"),
            dcc.Input(id = "bins_input_tab_4_",
                        type='number',
                        value=2,
                        min = 1,
                        step=1),
            dcc.Loading(
                                id="loading-3",
                                children=[html.Div(id='pairwise-mi-results', children = [html.Div([html.P(html.B("Pairwise mutual information of the observed data : "), style={'textAlign':'center'})])] )],
                                type="circle",
                        ),
        ])
    elif value == "tab-6":
        return html.Div([
            html.H3("Description"),
            dcc.Markdown(r'''One such approach is the Mahalanobis distance, which measures the distance between two multivariate datasets, taking into account the covariance structure.
                        The Mahalanobis distance provides a quantitative measure of the distance between the observed data and the generated data, taking into account the covariance structure.
                        A smaller Mahalanobis distance suggests a better fit between the generated data and the observed data in terms of their multivariate characteristics.''' , mathjax=True),
            dcc.Loading(
                                id="loading-1",
                                children=[html.Div(id='mahalanobis-distance')],
                                type="circle",
                        ),
            html.H3("Description"),
            dcc.Markdown(r'''The count threshold approach using the KS test allows you to assess the similarity between the generated data and the observed data at the variable level.
                         By counting the number of variables that meet a certain similarity criterion, you can gauge how well the generative model captures the distribution of the observed data across multiple variables.''' , mathjax=True),
            html.H4("Threshold"),
            dcc.Input(id = "threshold-ks",
                        type='number',
                        value=0.05,
                        min = 0,
                        step=0.01,
                        style={'float': 'left'}),
            dcc.Loading(
                                id="loading-2",
                                children=[html.Div(id='count-kstest')],
                                type="circle",
                        ),
            
        ])
    elif value == "tab-1-5":
        return html.Div([
            html.H3("Description"),
            dcc.Markdown(r'''This section is for plotting in copula space the pair of variables. It tells the form of de dependence between them.''' , mathjax=True),
            html.H1('Bivariate plot'),
            html.H3("Select first column"),
            dcc.Dropdown(id='columns-dropdown-scatter-1-eva', options = [], value = [],  multi=False, persistence=True),
            html.H3("Select second column"),
            dcc.Dropdown(id='columns-dropdown-scatter-2-eva', options = [], value = [],  multi=False, persistence=True),
            html.Hr(),
            dcc.Loading(
                        id="loading-1",
                        children=[html.Div(id ='bivariate-eva')],
                        type="circle",
                        ),
            
        ])
    else :
        raise PreventUpdate
    
# Show output for tab 1
@app.callback(Output('intermediate-value-3', 'data',allow_duplicate=True),
              Output('intermediate-value-4', 'data',allow_duplicate=True),
              Output('intermediate-value-5', 'data',allow_duplicate=True),
              Output('alert_submit_files', 'children'),
              Input('submit-file-tab-1', 'n_clicks'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-model', 'contents'),
              State('upload-model', 'filename'),prevent_initial_call=True)
def update_tab_1(n_clicks, contents_csv, filename_csv, contents_model, filename_model):
    if (n_clicks == 0): 
        raise PreventUpdate
    if n_clicks > 0 :
        if contents_csv is not None and contents_model is not None:
            children = [
                read_csv(contents_csv, filename_csv)]
            pseudo_obs =  pd.concat(children).drop(columns = {"Unnamed: 0"})
            
            model = read_json(contents_model, filename_model)
            # write model file and create vinecop object
            
            # json_string = json.dumps(model)
            with open("./output/model_upload.json", 'w') as outfile:
                 outfile.write(model)
            try :
                vmodel = pv.Vinecop("./output/model_upload.json")
                dim = vmodel.dim
            except :
                vmodel = VinecopSearch(filename = "./output/model_upload.json")
                dim = vmodel.d
            try :
                vmodel = pv.Vinecop(vmodel.structure, vmodel.pair_copula)
                dim = vmodel.dim
            except :
                vmodel = VinecopSearch(filename = "./output/model_upload.json")
                dim = vmodel.d
            
            if dim != len(pseudo_obs.columns):
                print(dim)
                print(pseudo_obs.columns)
                return {}, {}, {}, html.P("Dimension of model and data doesn't match")
            
            simulated_data = pd.DataFrame(vmodel.simulate(len(pseudo_obs)))
            simulated_data.columns = pseudo_obs.columns
            
            return simulated_data.to_dict("records"), pseudo_obs.to_dict("records"), model, html.P("Success : creation of simulated data with the given model")
            
        else :
            return {}, {}, {},  html.P("Please insert a model and a data file")
        
# Update columns dropdown
@app.callback(Output('columns-dropdown-tab-2', 'options'),
              Input('intermediate-value-3', 'data'),
              State('intermediate-value-4', 'data'))
def update_columns_tab_2(df,df1):
    df1 = pd.DataFrame.from_dict(df1)
    df = pd.DataFrame.from_dict(df)
    if not df1.empty and not df.empty :
        col1 = df1.columns
        col = df.columns
        assert (col == col1).all()
        return col1
    else :
        raise PreventUpdate

# Update columns dropdown
@app.callback(
              Output('columns-dropdown-tab-3', 'options'),
              Input('intermediate-value-3', 'data'),
              State('intermediate-value-4', 'data'))
def update_columns_tab_3(df,df1):
    df1 = pd.DataFrame.from_dict(df1)
    df = pd.DataFrame.from_dict(df)
    if not df1.empty and not df.empty :
        col1 = df1.columns
        col = df.columns
        assert (col == col1).all()
        return col1
    else :
        raise PreventUpdate

# Show output for tab 2

@app.callback(Output('tail-plot', 'children'),
              Input('columns-dropdown-tab-2', 'value'),
              State('intermediate-value-3', 'data'),
              State('intermediate-value-4', 'data'))
def update_tab_2_output(columns, df_simulated, df_observed):
    if len(columns) == 2 :
        df_simulated = pd.DataFrame.from_dict(df_simulated)
        df_observed = pd.DataFrame.from_dict(df_observed)
        c_n_observed_lower , c_n_observed_upper = empcdf_tail(df_observed[columns].to_numpy())
        c_n_simulated_lower , c_n_simulated_upper = empcdf_tail(df_simulated[columns].to_numpy())
        c_n_lower = pd.DataFrame({"observed" : c_n_observed_lower, "simulated" : c_n_simulated_lower, "alpha" : np.linspace(0, 0.1, 1000)})
        c_n_upper = pd.DataFrame({"observed" : c_n_observed_upper, "simulated" : c_n_simulated_upper, "alpha" : 1-np.linspace(0, 0.1, 1000)})
        fig1 = px.line(c_n_lower, x='alpha', y=c_n_lower.columns, title = "Empirical copula distribution functions in the lower tail")
        fig2 = px.line(c_n_upper, x='alpha', y=c_n_lower.columns, title = "Empirical copula distribution functions in the upper tail")
        return html.Div([
            dcc.Graph(figure = fig1),
            dcc.Graph(figure = fig2)
        ])
    elif len(columns) == 0 :
        raise PreventUpdate
    else :
        return html.P("Please select 2 columns")

# SHow output for tab 3

@app.callback(Output('qq-plot', 'children'),
              Input('columns-dropdown-tab-3', 'value'),
              State('intermediate-value-3', 'data'),
              State('intermediate-value-4', 'data'))
def update_tab_2_output(columns, df_simulated, df_observed):
    if len(columns) == 2 :
        df_simulated = pd.DataFrame.from_dict(df_simulated)
        df_observed = pd.DataFrame.from_dict(df_observed)
        w_k_observed = qqplot(df_observed[columns].to_numpy())
        w_k_simulated = qqplot(df_simulated[columns].to_numpy())
        w_k = pd.DataFrame({"observed" : w_k_observed, "simulated" : w_k_simulated})
        fig = px.scatter(x=w_k['observed'].sort_values(),y= w_k['simulated'].sort_values(), title = "QQ plot simulated depending of observed", labels=dict(x="w", y="w^k"))
        fig.add_shape(type="line",
            xref="x", yref="y",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(
                color="LightSeaGreen",
                width=1,
            ),
        )
        # Perform the KS test
        # ks_statistic, p_value = ks_2samp(df_observed[columns].to_numpy(), df_simulated[columns].to_numpy())
        return html.Div([
            dcc.Graph(figure = fig)
            #html.P(" KS-test p value :" + str(p_value)),
            #html.P(" KS-test ks_statistic :" + str(ks_statistic))
        ])
    elif len(columns) == 0 :
        raise PreventUpdate
    else :
        return html.P("Please select 2 columns")
    
# SHOW RESULTS TAB 4
@app.callback(Output('mean-copula-plot', 'children'),
              Input('intermediate-value-3', 'data'),
              Input('intermediate-value-4', 'data'))
def update_tab_3_output(df_simulated, df_observed):
    df_simulated = pd.DataFrame.from_dict(df_simulated)
    df_observed = pd.DataFrame.from_dict(df_observed)
    if not df_observed.empty and not df_simulated.empty :
        s_observed = df_observed.sum(axis = 1)/len(df_observed.columns)
        s_simulated = df_simulated.sum(axis = 1)/len(df_simulated.columns)
        return html.Div([
            dcc.Graph(figure = px.histogram(s_observed)),
            dcc.Graph(figure = px.histogram(s_simulated))
        ])
    else :
        raise PreventUpdate
    
# SHOW RESULTS TAB 5
@app.callback(Output('copula-mi-results', 'children'),
              Input('intermediate-value-5', 'data'))
def update_tab_5_output_copula(model):
    if model != None :
        try :
            vmodel = pv.Vinecop("./output/model_upload.json")
        except :
            try :
                vmodel = VinecopSearch(filename = "./output/model_upload.json")
                vmodel = pv.Vinecop(vmodel.structure, vmodel.pair_copula)
                res = mutual_info(vmodel)
                return html.Div([html.P(html.B("Mutual information of the vine copula model : " + str(res)), style={'textAlign':'center'})])
            except : 
                vmodel = VinecopSearch(filename = "./output/model_upload.json")
                return html.Div([html.P(html.B("Mutual information of the vine copula model : " + "Not implemented yet for VinecopSearch"), style={'textAlign':'center'})])
            
    else :
        raise PreventUpdate
    
@app.callback(Output('ksg-mi-results', 'children'),
              Input('intermediate-value-4', 'data'),
              Input('bins_input_tab_4', 'value')
              )
def update_tab_5_output_ksg(df, bins):
    df = pd.DataFrame.from_dict(df)
    if not df.empty :
        res = mutual_info_ksg(df.to_numpy(), bins)
        return html.Div([html.P(html.B("The KSG MI estimator of the observed data : " + str(res)), style={'textAlign':'center'})])
    else :
        raise PreventUpdate

@app.callback(Output('pairwise-mi-results', 'children'),
              Input('intermediate-value-4', 'data'),
              Input('bins_input_tab_4_', 'value')
              )
def update_tab_5_output_ksg(df, bins):
    df = pd.DataFrame.from_dict(df)
    if not df.empty :
        res = mutual_info_pairs(df.to_numpy(), bins)
        res = pd.DataFrame(res)
        res.index = df.columns
        res.columns = df.columns
        res = res.reset_index()
        
        return html.Div([html.P(html.B("Pairwise mutual information of the observed data : "), style={'textAlign':'center'}),
                         dash_table.DataTable(
                res.to_dict(orient = 'records'),
                [{'name': i, 'id': i} for i in res.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action='native',
                page_current= 0,
                page_size= 10,
                style_table={'overflowX': 'auto'},
                persistence=True,
                persisted_props=["data"]
            ),])
    else :
        raise PreventUpdate

@app.callback(Output('mahalanobis-distance', 'children'),
              Input('intermediate-value-4', 'data'),
              Input('intermediate-value-3', 'data'),
              )
def mahalanobis_distance_output(df1, df2):
    df1 = pd.DataFrame.from_dict(df1) # observed
    df2 = pd.DataFrame.from_dict(df2) # simulated
    res = mahalanobis_distance1(df2, df1)
    fig = px.box(res, labels={'value': 'Mahalanobis Distance'})
    fig.update_layout(
        title='Box Plot of Mahalanobis Distances',
        xaxis_title='Dataset',
        yaxis_title='Mahalanobis Distance'
    )
    return html.Div([dcc.Graph(figure=fig)])
#html.Div([html.P(html.B("Mahalanobis distance : "+str(res)), style={'textAlign':'center'})])

@app.callback(Output('count-kstest', 'children'),
              Input('intermediate-value-4', 'data'),
              Input('intermediate-value-3', 'data'),
              Input('threshold-ks', 'value'),
              )
def count_similar_variables_output(df1, df2, threshold):
    df1 = pd.DataFrame.from_dict(df1) # observed
    df2 = pd.DataFrame.from_dict(df2) # simulated
    res = count_similar_variables(df1, df2, threshold)

    return html.Div([
        html.P(html.B("Count variables where p-value < " +str(threshold) + " : "+str(res) + " / " + str(len(df1.columns)) ), style={'textAlign':'center'})
        ])

####### SCATTER PLOT ###########
@app.callback(
    Output("columns-dropdown-scatter-1-eva", "options"),
    Output("columns-dropdown-scatter-2-eva", "options"),
    Input("intermediate-value-4", "data")
)
def display_columns_to_choose_scatter_eva(df):
    df = pd.DataFrame.from_dict(df)
    col = df.columns
    return col, col

@app.callback(
    Output("bivariate-eva", "children"),
    Input("columns-dropdown-scatter-1-eva", "value"),
    Input("columns-dropdown-scatter-2-eva", "value"),
    State("intermediate-value-4", "data"),
    State("intermediate-value-3", "data"), prevent_initial_call=True
)
def scatter_plot_eva(col1, col2, df1, df2):
    df1 = pd.DataFrame.from_dict(df1)
    df2 = pd.DataFrame.from_dict(df2)
    if (not df1.empty) and (not df2.empty) :
        fig1 = px.scatter(df1, x=col1, y= col2, marginal_x="histogram", marginal_y="histogram", title = "Scartter plot for observed data")
        fig2 = px.scatter(df2, x=col1, y= col2, marginal_x="histogram", marginal_y="histogram", title = "Scartter plot for simulated data")
        return html.Div([dcc.Graph(figure=fig1),
                         dcc.Graph(figure=fig2)
        ])
    else :
        raise PreventUpdate
    
#### PAGE 5
@app.callback(
              Output('intermediate-value-4', 'data',allow_duplicate=True),
              Output('intermediate-value-5', 'data',allow_duplicate=True),
              Output('alert_submit_files_sample', 'children'),
              Input('submit-file-sample', 'n_clicks'),
              State('upload-data-page5', 'contents'),
              State('upload-data-page5', 'filename'),
              State('upload-model-page5', 'contents'),
              State('upload-model-page5', 'filename'),prevent_initial_call=True)
def update_page5(n_clicks, contents_csv, filename_csv, contents_model, filename_model):
    if (n_clicks == 0): 
        raise PreventUpdate
    if n_clicks > 0 :
        if contents_csv is not None and contents_model is not None:
            children = [
                read_csv(contents_csv, filename_csv)]
            real_obs =  pd.concat(children).drop(columns = {"Unnamed: 0"})
            
            model = read_json(contents_model, filename_model)
            # write model file and create vinecop object
            
            # json_string = json.dumps(model)
            with open("./output/model_upload.json", 'w') as outfile:
                 outfile.write(model)
            try :
                vmodel = pv.Vinecop("./output/model_upload.json")
                dim = vmodel.dim

            except :
                vmodel = VinecopSearch(filename = "./output/model_upload.json")
                dim = vmodel.d

            if dim != len(real_obs.columns):
                return {}, {},  html.P("Dimension of model and data doesn't match")
            
            
            return real_obs.to_dict("records"), model, html.P("Success : creation of simulated data with the given model")
            
        else :
            return {}, {},  html.P("Please insert a model and a data file")
@app.callback(
              Output('intermediate-value-3', 'data',allow_duplicate=True),
              Output('output_samples', 'children'),
              Input('btn_submit_sample', 'n_clicks'),
              State('bins_input_samples', 'value'),
              State('intermediate-value-4', 'data'),
              State('intermediate-value-5', 'data'),prevent_initial_call=True)
def generate_sample(n_clicks, n, df1, df2):
    if (n_clicks == 0): 
        raise PreventUpdate
    if n_clicks > 0 :
        if (df2 == {}) or (df2 == None):
            return {}, html.P("Please upload the model first")
        else :
            df1 = pd.DataFrame.from_dict(df1)
            try :
                vmodel = pv.Vinecop("./output/model_upload.json")
                dim = vmodel.dim
            except :
                vmodel = VinecopSearch(filename = "./output/model_upload.json")
                dim = vmodel.d
            try :
                vmodel = pv.Vinecop(vmodel.structure, vmodel.pair_copula)
                dim = vmodel.dim
            except :
                vmodel = VinecopSearch(filename = "./output/model_upload.json")
                dim = vmodel.d
            
            if dim != len(df1.columns):
                return {}, html.P("Dimension of model and data doesn't match")
            
            simulated_data = pd.DataFrame(vmodel.simulate(n))
            simulated_data.columns = df1.columns
            
            return simulated_data.to_dict("records"), html.P("Success : Generate data with the given model")
@app.callback(
              Output('intermediate-value-6', 'data',allow_duplicate=True),
              Output('output_transformation', 'children'),
              Input('btn_submit_transformation', 'n_clicks'),
              State('intermediate-value-3', 'data'),
              State('intermediate-value-4', 'data'),prevent_initial_call=True)
def u_to_m_output(n_clicks, simulated_data, observed_data):
    if (n_clicks == 0): 
        raise PreventUpdate
    if n_clicks > 0 :
        simulated_data = pd.DataFrame.from_dict(simulated_data)
        observed_data = pd.DataFrame.from_dict(observed_data)
        if simulated_data.empty:
            return {}, html.P("Please generate samples first")
        else :
            pseudo_data = PreProcessing().pobs(observed_data)
            simulated_data = pd.DataFrame(u_to_m(pseudo_data.to_numpy(), observed_data.to_numpy(),simulated_data.to_numpy()))
            simulated_data.columns = observed_data.columns
            
            return simulated_data.to_dict("records"), html.P("Success : Transform data from copula space to real space")

# Download button btn_data_sample
@app.callback(
    Output("download-data-sample", "data"),
    Input("btn_data_sample", "n_clicks"),
    State("intermediate-value-3", "data"),
    prevent_initial_call=True
)
def func_df_sample(n_clicks, df):
    return dcc.send_data_frame(pd.DataFrame.from_dict(df).to_csv, "samples_df.csv")

@app.callback(
    Output("download-data-transformation", "data"),
    Input("btn_data_transformation", "n_clicks"),
    State("intermediate-value-6", "data"),
    prevent_initial_call=True
)
def func_df_sample_to_m(n_clicks, df):
    return dcc.send_data_frame(pd.DataFrame.from_dict(df).to_csv, "samples_to_m_df.csv")

if __name__ == "__main__":
    app.run_server(debug=True)