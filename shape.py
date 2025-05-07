import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime

# ---- CONFIG ----
DARK_THEME = dbc.themes.CYBORG

# ---- DATA LOADING ----
def load_data(path='rbond_data_with_maturity.csv'):
    df = pd.read_csv(path, parse_dates=['report_date'])
    # Only keep REPO, SATB, ZARGB bonds
    df = df[df['bond'].str.startswith(('REPO', 'SATB', 'ZARGB'))].copy()
    df['bond_yield'] = df['bond_yield'].astype(float)
    df['years_to_maturity'] = df['years_to_maturity'].astype(float)
    return df

def get_unique_dates_bonds(df):
    unique_dates = np.sort(df['report_date'].unique())
    unique_bonds = np.sort(df['bond'].unique())
    return unique_dates, unique_bonds

# ---- SHAPE ANALYTICS ----
def build_shape_analytics(df, unique_dates, unique_bonds):
    """
    Build a comprehensive set of yield curve shape analytics, including:
    - All possible gradients/slopes between bond pairs
    - Curvature, butterfly, and hump metrics
    - Rolling statistics (mean, volatility)
    - PCA (principal component analysis) for shape decomposition
    - Correlation matrices
    - Summary statistics tables
    Returns: shape_df (analytics), pivot (yields), extra dict of tables/metrics
    """
    import itertools
    from sklearn.decomposition import PCA
    # Remove duplicate (date, bond)
    df = df.drop_duplicates(subset=['report_date', 'bond'], keep='last')
    # Exclude matured bonds: only include rows where maturity >= report_date
    bond_warned = set()
    def not_matured(row):
        val = row.get('exact_maturity', None)
        if not isinstance(val, str) or pd.isnull(val):
            b = row.get('bond', 'UNKNOWN')
            if b not in bond_warned:
                print(f"[ShapeAnalytics][WARNING] Bond {b} has invalid or missing maturity, excluded from analytics.")
                bond_warned.add(b)
            return False
        try:
            mat = datetime.strptime(val, '%d/%m/%Y')
            return mat >= row['report_date']
        except Exception as e:
            b = row.get('bond', 'UNKNOWN')
            if b not in bond_warned:
                print(f"[ShapeAnalytics][WARNING] Could not parse maturity for bond {b} on {row['report_date']}: {e}")
                bond_warned.add(b)
            return False
    if 'exact_maturity' in df.columns:
        df = df[df.apply(not_matured, axis=1)]
    # Pivot for fast lookup
    pivot = df.pivot(index='report_date', columns='bond', values='bond_yield')
    
    # Fill missing values with last known rate (forward-fill)
    print(f"[ShapeAnalytics] Initial pivot shape: {pivot.shape}, missing values: {pivot.isna().sum().sum()}")
    pivot_filled = pivot.ffill()  # Forward-fill (use last known rate)
    missing_after_ffill = pivot_filled.isna().sum().sum()
    print(f"[ShapeAnalytics] After forward-fill, still missing: {missing_after_ffill} values")
    
    # Use the filled pivot for all analytics
    pivot = pivot_filled
    # --- Slope/Gradient for all pairs ---
    slopes = {}
    for b1, b2 in itertools.combinations(unique_bonds, 2):
        if b1 in pivot.columns and b2 in pivot.columns:
            name = f"{b2}-{b1}"  # ascending by maturity assumed
            slopes[name] = (pivot[b2] - pivot[b1]) * 100
    slopes_df = pd.DataFrame(slopes)
    # --- Curvature, Butterfly, Hump ---
    curvature = {}
    butterfly = {}
    for b1, b2, b3 in itertools.combinations(unique_bonds, 3):
        if all(x in pivot.columns for x in [b1, b2, b3]):
            curv_name = f"curv_{b1}_{b2}_{b3}"
            curvature[curv_name] = ((pivot[b1] + pivot[b3])/2 - pivot[b2]) * 100
            bf_name = f"bfly_{b1}_{b2}_{b3}"
            butterfly[bf_name] = (pivot[b2] - (pivot[b1] + pivot[b3])/2) * 100
    curvature_df = pd.DataFrame(curvature)
    butterfly_df = pd.DataFrame(butterfly)
    # --- Rolling stats ---
    rolling_mean = pivot.rolling(21, min_periods=5).mean()
    rolling_std = pivot.rolling(21, min_periods=5).std()
    # --- PCA (first 3 components) ---
    pca = PCA(n_components=min(3, len(pivot.columns)))
    pca_data = pivot.dropna(axis=1, how='any').ffill().bfill()
    pca_result = np.full((len(pivot), 3), np.nan)
    if pca_data.shape[1] >= 3:
        pca_fit = pca.fit_transform(pca_data)
        pca_result[:pca_fit.shape[0], :pca_fit.shape[1]] = pca_fit
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(3)], index=pivot.index)
    # --- Correlation matrix ---
    corr_matrix = pivot.corr()
    # --- Summary stats ---
    summary_table = pivot.describe().T
    # --- Assemble shape_df efficiently (avoid fragmentation) ---
    # Start with a base dataframe containing report_date
    shape_df = pd.DataFrame({'report_date': pivot.index})
    
    # Create a list of dataframes to concatenate
    dfs_to_concat = [shape_df]
    
    # Add slopes dataframe if it exists
    if slopes:
        dfs_to_concat.append(pd.DataFrame(slopes))
        
    # Add curvature dataframe if it exists
    if curvature:
        dfs_to_concat.append(pd.DataFrame(curvature))
        
    # Concatenate all dataframes at once (much more efficient)
    if len(dfs_to_concat) > 1:
        shape_df = pd.concat(dfs_to_concat, axis=1)
        # Remove duplicate report_date column if it exists
        shape_df = shape_df.loc[:,~shape_df.columns.duplicated()]
    # --- Collect all analytics for UI ---
    analytics = {
        'slopes': slopes_df,
        'curvature': curvature_df,
        'butterfly': butterfly_df,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'pca': pca_df,
        'corr': corr_matrix,
        'summary': summary_table,
        'pivot': pivot
    }
    return shape_df, analytics

# ---- DASH APP ----
def make_app(df, shape_df, analytics, unique_dates, unique_bonds):
    """
    Build a comprehensive Dash app with tabs for:
    - Shape Metrics (time series)
    - Heatmap (correlation)
    - Rolling Stats (mean, std)
    - PCA (principal components)
    - Summary Table
    - Raw Yields Table
    """
    app = Dash(__name__, external_stylesheets=[DARK_THEME], title="Yield Curve Shape Analytics")
    # Dropdowns for dynamic metrics
    metric_options = [{'label': c, 'value': c} for c in shape_df.columns if c != 'report_date']
    slope_options = [{'label': c, 'value': c} for c in analytics['slopes'].columns]
    curvature_options = [{'label': c, 'value': c} for c in analytics['curvature'].columns]
    butterfly_options = [{'label': c, 'value': c} for c in analytics['butterfly'].columns]
    # Layout with tabs
    app.layout = dbc.Container([
        html.H2("Yield Curve Shape Analytics", className='mt-4'),
        dbc.Tabs([
            dbc.Tab(label="Shape Metrics", children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='shape-metric', options=metric_options, value=metric_options[0]['value'] if metric_options else None, className='mb-2'),
                    ], md=4),
                ]),
                dcc.Graph(id='shape-metric-graph'),
            ]),
            dbc.Tab(label="Correlation Heatmap", children=[
                dcc.Graph(id='corr-heatmap', figure=go.Figure(data=go.Heatmap(z=analytics['corr'].values, x=analytics['corr'].columns, y=analytics['corr'].index, colorscale='Viridis')).update_layout(template='plotly_dark', title='Bond Yield Correlation Matrix')),
            ]),
            dbc.Tab(label="Rolling Stats", children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='rolling-bond', options=[{'label': b, 'value': b} for b in analytics['pivot'].columns], value=analytics['pivot'].columns[0] if len(analytics['pivot'].columns)>0 else None, className='mb-2'),
                    ], md=4),
                ]),
                dcc.Graph(id='rolling-mean-graph'),
                dcc.Graph(id='rolling-std-graph'),
            ]),
            dbc.Tab(label="PCA", children=[
                dcc.Graph(id='pca-graph', figure=go.Figure(data=[go.Scatter(x=analytics['pca'].index, y=analytics['pca'][c], mode='lines', name=c) for c in analytics['pca'].columns]).update_layout(template='plotly_dark', title='Yield Curve PCA Components', xaxis_title='Date', yaxis_title='Component Value')),
            ]),
            dbc.Tab(label="Summary Table", children=[
                html.Div([
                    html.H5("Summary Statistics (by Bond)"),
                    dbc.Table.from_dataframe(analytics['summary'].reset_index(), striped=True, bordered=True, hover=True, dark=True)
                ])
            ]),
            dbc.Tab(label="Raw Yields", children=[
                html.Div([
                    html.H5("Raw Bond Yields (Pivot Table)"),
                    dbc.Table.from_dataframe(analytics['pivot'].reset_index(), striped=True, bordered=True, hover=True, dark=True, size='sm')
                ])
            ]),
            dbc.Tab(label="Curvature/Butterfly", children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='curvature-metric', options=curvature_options, value=curvature_options[0]['value'] if curvature_options else None, className='mb-2'),
                    ], md=4),
                    dbc.Col([
                        dcc.Dropdown(id='butterfly-metric', options=butterfly_options, value=butterfly_options[0]['value'] if butterfly_options else None, className='mb-2'),
                    ], md=4),
                ]),
                dcc.Graph(id='curvature-graph'),
                dcc.Graph(id='butterfly-graph'),
            ]),
        ]),
        html.Hr(),
        html.H5("Single Bond Yield History"),
        dbc.Row([
            dbc.Col([
                dcc.DatePickerSingle(
                    id='date-picker',
                    min_date_allowed=pd.to_datetime(unique_dates.min()).strftime('%Y-%m-%d'),
                    max_date_allowed=pd.to_datetime(unique_dates.max()).strftime('%Y-%m-%d'),
                    date=pd.to_datetime(unique_dates.max()).strftime('%Y-%m-%d'),
                    display_format='YYYY-MM-DD',
                    className='mb-2'
                ),
                dcc.Dropdown(
                    id='bond-dropdown',
                    options=[{'label': b, 'value': b} for b in unique_bonds],
                    value=unique_bonds[0] if len(unique_bonds) > 0 else None,
                    className='mb-2'
                ),
            ], md=4),
            dbc.Col([
                dcc.Graph(id='bond-yield-graph'),
            ], md=8),
        ]),
    ], fluid=True)

    @app.callback(
        Output('shape-metric-graph', 'figure'),
        Input('shape-metric', 'value')
    )
    def update_shape_metric_graph(metric):
        fig = go.Figure()
        if metric and metric in shape_df.columns:
            fig.add_trace(go.Scatter(x=shape_df['report_date'], y=shape_df[metric], mode='lines+markers', name=metric))
            fig.update_layout(template='plotly_dark', title=f'{metric} Time Series', xaxis_title='Date', yaxis_title=metric)
        return fig

    @app.callback(
        Output('rolling-mean-graph', 'figure'),
        Input('rolling-bond', 'value')
    )
    def update_rolling_mean_graph(bond):
        fig = go.Figure()
        if bond in analytics['rolling_mean'].columns:
            fig.add_trace(go.Scatter(x=analytics['rolling_mean'].index, y=analytics['rolling_mean'][bond], mode='lines', name=f"Mean {bond}"))
            fig.update_layout(template='plotly_dark', title=f'21D Rolling Mean: {bond}', xaxis_title='Date', yaxis_title='Mean Yield')
        return fig

    @app.callback(
        Output('rolling-std-graph', 'figure'),
        Input('rolling-bond', 'value')
    )
    def update_rolling_std_graph(bond):
        fig = go.Figure()
        if bond in analytics['rolling_std'].columns:
            fig.add_trace(go.Scatter(x=analytics['rolling_std'].index, y=analytics['rolling_std'][bond], mode='lines', name=f"Std {bond}"))
            fig.update_layout(template='plotly_dark', title=f'21D Rolling Std: {bond}', xaxis_title='Date', yaxis_title='Std Dev')
        return fig

    @app.callback(
        Output('curvature-graph', 'figure'),
        Input('curvature-metric', 'value')
    )
    def update_curvature_graph(metric):
        fig = go.Figure()
        if metric and metric in analytics['curvature'].columns:
            fig.add_trace(go.Scatter(x=analytics['curvature'].index, y=analytics['curvature'][metric], mode='lines+markers', name=metric))
            fig.update_layout(template='plotly_dark', title=f'{metric} (Curvature)', xaxis_title='Date', yaxis_title=metric)
        return fig

    @app.callback(
        Output('butterfly-graph', 'figure'),
        Input('butterfly-metric', 'value')
    )
    def update_butterfly_graph(metric):
        fig = go.Figure()
        if metric and metric in analytics['butterfly'].columns:
            fig.add_trace(go.Scatter(x=analytics['butterfly'].index, y=analytics['butterfly'][metric], mode='lines+markers', name=metric))
            fig.update_layout(template='plotly_dark', title=f'{metric} (Butterfly)', xaxis_title='Date', yaxis_title=metric)
        return fig

    @app.callback(
        Output('bond-yield-graph', 'figure'),
        [Input('bond-dropdown', 'value'), Input('date-picker', 'date')]
    )
    def update_bond_yield_graph(bond, date):
        fig = go.Figure()
        if bond in analytics['pivot'].columns:
            fig.add_trace(go.Scatter(x=analytics['pivot'].index, y=analytics['pivot'][bond], mode='lines+markers', name=bond))
            if date:
                fig.add_vline(x=date, line_dash='dash', line_color='orange')
            fig.update_layout(template='plotly_dark', title=f'Yield History: {bond}', xaxis_title='Date', yaxis_title='Yield (%)')
        return fig

    return app

# ---- MAIN ----
if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    unique_dates, unique_bonds = get_unique_dates_bonds(df)
    print(f"Found {len(unique_dates)} unique dates and {len(unique_bonds)} unique bonds")
    
    print("Building shape analytics...")
    shape_df, analytics = build_shape_analytics(df, unique_dates, unique_bonds)
    print("Analytics completed")
    
    print("Creating Dash app...")
    app = make_app(df, shape_df, analytics, unique_dates, unique_bonds)
    print("Starting server at http://0.0.0.0:8050")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
