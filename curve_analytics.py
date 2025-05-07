"""
Professional Yield Curve Analytics
==================================
Advanced analytics for fixed income yield curve shape and evolution analysis.
Focuses on complete data integrity and professional financial metrics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Constants
DARK_THEME = dbc.themes.DARKLY
DATE_FORMAT = '%d/%m/%Y'
COLORS = px.colors.qualitative.Dark24


class YieldCurveData:
    """Professional yield curve data management with strict data integrity."""
    
    def __init__(self, csv_path='rbond_data_with_maturity.csv'):
        """Initialize with raw bond data."""
        self.raw_df = pd.read_csv(csv_path, parse_dates=['report_date'])
        self.filter_bonds()
        self.unique_dates = np.sort(self.df['report_date'].unique())
        self.unique_bonds = np.sort(self.df['bond'].unique())
        self.pivot = None
        self.complete_data_mask = None
        self.prepare_pivot_table()
        
    def filter_bonds(self):
        """Filter to relevant bonds and ensure data quality."""
        # Focus on REPO, SATB, ZARGB bonds
        self.df = self.raw_df[self.raw_df['bond'].str.startswith(('REPO', 'SATB', 'ZARGB'))].copy()
        self.df['bond_yield'] = self.df['bond_yield'].astype(float)
        self.df['years_to_maturity'] = self.df['years_to_maturity'].astype(float)
        # Remove duplicates and ensure data quality
        self.df = self.df.drop_duplicates(subset=['report_date', 'bond'], keep='last')
        # Filter out matured bonds
        if 'exact_maturity' in self.df.columns:
            self.df = self.filter_matured_bonds(self.df)
    
    def filter_matured_bonds(self, df):
        """Remove bonds that have matured based on exact_maturity date."""
        def not_matured(row):
            val = row.get('exact_maturity', None)
            if not isinstance(val, str) or pd.isnull(val):
                return False
            try:
                mat = datetime.strptime(val, DATE_FORMAT)
                return mat >= row['report_date']
            except Exception:
                return False
        return df[df.apply(not_matured, axis=1)]
    
    def prepare_pivot_table(self):
        """Create pivot table and identify dates with complete data."""
        # Create pivot table
        self.pivot = self.df.pivot(index='report_date', columns='bond', values='bond_yield')
        
        # Identify dates with complete data for each bond pair
        self.complete_data_mask = {}
        for bond1 in self.unique_bonds:
            for bond2 in self.unique_bonds:
                if bond1 != bond2:
                    pair_key = f"{bond1}_{bond2}"
                    # Only include dates where both bonds have data
                    mask = (~self.pivot[bond1].isna()) & (~self.pivot[bond2].isna())
                    self.complete_data_mask[pair_key] = mask
        
        # Log data completeness
        total_dates = len(self.unique_dates)
        print(f"Total dates in dataset: {total_dates}")
        for pair, mask in self.complete_data_mask.items():
            complete_count = mask.sum()
            if complete_count > 0:
                print(f"Pair {pair}: {complete_count}/{total_dates} complete dates ({complete_count/total_dates:.1%})")


class CurveAnalytics:
    """Professional yield curve analytics with strict data integrity."""
    
    def __init__(self, curve_data):
        """Initialize with curve data object."""
        self.data = curve_data
        self.spreads = {}
        self.curvature = {}
        self.level_slope_curvature = None
        self.calculate_analytics()
    
    def calculate_analytics(self):
        """Calculate all yield curve analytics."""
        self.calculate_spreads()
        self.calculate_curvature()
        self.calculate_pca_factors()
    
    def calculate_spreads(self):
        """Calculate spreads between all bond pairs, ensuring data integrity."""
        self.spreads = {}
        
        # Define key spread pairs (can be customized)
        key_pairs = [
            ('ZARGB2', 'ZARGB10'),  # 2s10s
            ('ZARGB5', 'ZARGB30'),  # 5s30s
            ('ZARGB10', 'ZARGB30'), # 10s30s
            ('REPO', 'ZARGB2'),     # Repo-2y
            ('REPO', 'ZARGB10'),    # Repo-10y
        ]
        
        # Calculate spreads only for dates with complete data
        for bond1, bond2 in key_pairs:
            pair_key = f"{bond1}_{bond2}"
            if bond1 in self.data.pivot.columns and bond2 in self.data.pivot.columns:
                # Only calculate for dates where both bonds have data
                mask = self.data.complete_data_mask.get(pair_key, pd.Series(False, index=self.data.pivot.index))
                if mask.any():
                    # Create spread series with proper name
                    spread_name = f"{bond2}-{bond1}"
                    spread_values = (self.data.pivot[bond2] - self.data.pivot[bond1]) * 100  # in basis points
                    self.spreads[spread_name] = pd.Series(np.nan, index=self.data.pivot.index)
                    self.spreads[spread_name][mask] = spread_values[mask]
    
    def calculate_curvature(self):
        """Calculate curvature metrics, ensuring data integrity."""
        self.curvature = {}
        
        # Define key curvature triplets (can be customized)
        key_triplets = [
            ('ZARGB2', 'ZARGB10', 'ZARGB30'),  # Standard curvature
            ('ZARGB5', 'ZARGB10', 'ZARGB20'),  # Mid-curve curvature
        ]
        
        # Calculate curvature only for dates with complete data
        for bond1, bond2, bond3 in key_triplets:
            if all(bond in self.data.pivot.columns for bond in [bond1, bond2, bond3]):
                # Only calculate for dates where all three bonds have data
                mask1 = self.data.complete_data_mask.get(f"{bond1}_{bond2}", pd.Series(False, index=self.data.pivot.index))
                mask2 = self.data.complete_data_mask.get(f"{bond2}_{bond3}", pd.Series(False, index=self.data.pivot.index))
                mask = mask1 & mask2
                
                if mask.any():
                    # Create curvature series with proper name
                    curv_name = f"Curv_{bond1}_{bond2}_{bond3}"
                    curv_values = ((self.data.pivot[bond1] + self.data.pivot[bond3])/2 - self.data.pivot[bond2]) * 100  # in basis points
                    self.curvature[curv_name] = pd.Series(np.nan, index=self.data.pivot.index)
                    self.curvature[curv_name][mask] = curv_values[mask]
    
    def calculate_pca_factors(self):
        """Calculate PCA factors (level, slope, curvature) from yield curve."""
        # Store PCA metadata for display in the dashboard
        self.pca_metadata = {
            'components': None,
            'explained_variance': None,
            'bonds_used': None,
            'data_points': 0,
            'missing_data_pct': 0,
            'method': 'Standard PCA with forward-backward filling of missing values'
        }
        
        # More lenient approach: select bonds that have at least 70% of data points
        min_data_threshold = 0.7  # At least 70% of dates must have data
        total_dates = len(self.data.pivot.index)
        
        # Count non-NaN values for each bond
        non_nan_counts = self.data.pivot.count()
        valid_bonds = non_nan_counts[non_nan_counts/total_dates >= min_data_threshold].index.tolist()
        
        # Need at least 3 bonds for PCA
        if len(valid_bonds) >= 3:
            # Select only the valid bonds
            pca_data = self.data.pivot[valid_bonds].copy()
            
            # Calculate missing data percentage before filling
            missing_pct = pca_data.isna().sum().sum() / (pca_data.shape[0] * pca_data.shape[1]) * 100
            self.pca_metadata['missing_data_pct'] = missing_pct
            
            # Fill missing values: forward fill then backward fill
            pca_data = pca_data.ffill().bfill()
            
            # Store metadata
            self.pca_metadata['bonds_used'] = valid_bonds
            self.pca_metadata['data_points'] = pca_data.shape[0]
            
            # Standardize the data (important for PCA)
            pca_data_std = (pca_data - pca_data.mean()) / pca_data.std()
            
            # Perform PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(pca_data_std)
            
            # Store component loadings (eigenvectors)
            self.pca_metadata['components'] = pd.DataFrame(
                pca.components_,
                columns=valid_bonds,
                index=['Level', 'Slope', 'Curvature']
            )
            
            # Create DataFrame with results
            self.level_slope_curvature = pd.DataFrame(
                pca_result, 
                index=pca_data.index,
                columns=['Level', 'Slope', 'Curvature']
            )
            
            # Calculate explained variance
            explained_var = pca.explained_variance_ratio_
            self.pca_metadata['explained_variance'] = explained_var
            
            print(f"PCA analysis complete using {len(valid_bonds)} bonds")
            print(f"PCA explained variance: Level={explained_var[0]:.2%}, Slope={explained_var[1]:.2%}, Curvature={explained_var[2]:.2%}")
        else:
            print(f"Insufficient data for PCA: only {len(valid_bonds)} bonds have sufficient data (need at least 3)")
            self.level_slope_curvature = None


class CurveDashboard:
    """Professional yield curve dashboard with advanced visualizations."""
    
    def __init__(self, curve_data, analytics):
        """Initialize with curve data and analytics objects."""
        self.data = curve_data
        self.analytics = analytics
        self.app = Dash(__name__, external_stylesheets=[DARK_THEME], title="Professional Yield Curve Analytics")
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up professional dashboard layout with tabs."""
        # Prepare dropdown options
        spread_options = [{'label': k, 'value': k} for k in self.analytics.spreads.keys()]
        curvature_options = [{'label': k, 'value': k} for k in self.analytics.curvature.keys()]
        bond_options = [{'label': b, 'value': b} for b in self.data.unique_bonds]
        
        # Dashboard layout with tabs
        self.app.layout = dbc.Container([
            html.H2("Professional Yield Curve Analytics", className='mt-4 mb-4'),
            dbc.Tabs([
                # Tab 1: Spread Analysis
                dbc.Tab(label="Spread Analysis", children=[
                    html.H4("Yield Curve Spreads", className='mt-3 mb-3'),
                    html.P("Analysis of key yield spreads with complete data integrity", className='text-muted'),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Spread:"),
                            dcc.Dropdown(
                                id='spread-selector',
                                options=spread_options,
                                value=spread_options[0]['value'] if spread_options else None,
                                className='mb-3'
                            ),
                        ], md=4),
                    ]),
                    dcc.Graph(id='spread-chart'),
                    html.Div(id='spread-stats', className='mt-3')
                ]),
                
                # Tab 2: Curvature Analysis
                dbc.Tab(label="Curvature Analysis", children=[
                    html.H4("Yield Curve Curvature", className='mt-3 mb-3'),
                    html.P("Analysis of yield curve shape and convexity", className='text-muted'),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Curvature Metric:"),
                            dcc.Dropdown(
                                id='curvature-selector',
                                options=curvature_options,
                                value=curvature_options[0]['value'] if curvature_options else None,
                                className='mb-3'
                            ),
                        ], md=4),
                    ]),
                    dcc.Graph(id='curvature-chart'),
                    html.Div(id='curvature-stats', className='mt-3')
                ]),
                
                # Tab 3: PCA Factors
                dbc.Tab(label="PCA Factors", children=[
                    html.H4("Level, Slope, Curvature Decomposition", className='mt-3 mb-3'),
                    html.P("Principal component analysis of yield curve movements", className='text-muted'),
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert(
                                [
                                    html.H5("PCA Methodology", className="alert-heading"),
                                    html.P([
                                        "Principal Component Analysis (PCA) is a statistical technique that identifies the main ",
                                        "patterns of variation in yield curves. The first three components typically correspond to:"
                                    ]),
                                    html.Ul([
                                        html.Li([html.Strong("Level (PC1): "), "Parallel shifts in the entire yield curve"]),
                                        html.Li([html.Strong("Slope (PC2): "), "Changes in the steepness of the yield curve"]),
                                        html.Li([html.Strong("Curvature (PC3): "), "Changes in the curvature (belly) of the yield curve"])
                                    ]),
                                    html.P("These three factors typically explain 95-99% of all yield curve movements.")
                                ],
                                color="info",
                                className="mb-3"
                            )
                        ], md=12),
                    ]),
                    dcc.Graph(id='pca-chart'),
                    html.Div(id='pca-stats', className='mt-3'),
                    html.Div(id='pca-details', className='mt-3')
                ]),
                
                # Tab 4: Yield Curve Evolution
                dbc.Tab(label="Curve Evolution", children=[
                    html.H4("Yield Curve Evolution", className='mt-3 mb-3'),
                    html.P("Analysis of yield curve shape changes over time", className='text-muted'),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Date Range:"),
                            dcc.DatePickerRange(
                                id='date-range',
                                min_date_allowed=pd.Timestamp(self.data.unique_dates[0]).strftime('%Y-%m-%d'),
                                max_date_allowed=pd.Timestamp(self.data.unique_dates[-1]).strftime('%Y-%m-%d'),
                                start_date=pd.Timestamp(self.data.unique_dates[0]).strftime('%Y-%m-%d'),
                                end_date=pd.Timestamp(self.data.unique_dates[-1]).strftime('%Y-%m-%d'),
                                className='mb-3'
                            ),
                        ], md=6),
                    ]),
                    dcc.Graph(id='evolution-chart'),
                ]),
                
                # Tab 5: Individual Bond Analysis
                dbc.Tab(label="Bond Analysis", children=[
                    html.H4("Individual Bond Analysis", className='mt-3 mb-3'),
                    html.P("Detailed analysis of individual bond yields", className='text-muted'),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Bond:"),
                            dcc.Dropdown(
                                id='bond-selector',
                                options=bond_options,
                                value=bond_options[0]['value'] if bond_options else None,
                                className='mb-3'
                            ),
                        ], md=4),
                    ]),
                    dcc.Graph(id='bond-chart'),
                    html.Div(id='bond-stats', className='mt-3')
                ]),
            ]),
            html.Hr(),
            html.Footer([
                html.P("Professional Yield Curve Analytics Dashboard", className='text-muted text-center'),
                html.P("Data integrity enforced: analytics shown only for dates with complete data", className='text-muted text-center small')
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Set up interactive callbacks for the dashboard."""
        # Spread Analysis callback
        @self.app.callback(
            [Output('spread-chart', 'figure'),
             Output('spread-stats', 'children')],
            [Input('spread-selector', 'value')]
        )
        def update_spread_analysis(spread_name):
            if not spread_name or spread_name not in self.analytics.spreads:
                return go.Figure(), html.Div("No data available")
            
            # Get spread data
            spread_data = self.analytics.spreads[spread_name]
            valid_data = spread_data.dropna()
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=valid_data.index, 
                y=valid_data.values,
                mode='lines+markers',
                name=spread_name,
                line=dict(color=COLORS[0], width=2),
                marker=dict(size=6)
            ))
            
            # Add reference lines
            mean_val = valid_data.mean()
            fig.add_hline(y=mean_val, line_dash="dash", line_color="white", 
                         annotation_text=f"Mean: {mean_val:.1f}bp")
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                title=f'Spread Analysis: {spread_name}',
                xaxis_title='Date',
                yaxis_title='Spread (basis points)',
                height=500,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            # Statistics
            stats = dbc.Card([
                dbc.CardHeader("Spread Statistics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Mean: {valid_data.mean():.2f}bp"),
                            html.P(f"Median: {valid_data.median():.2f}bp"),
                        ], md=4),
                        dbc.Col([
                            html.P(f"Min: {valid_data.min():.2f}bp"),
                            html.P(f"Max: {valid_data.max():.2f}bp"),
                        ], md=4),
                        dbc.Col([
                            html.P(f"Std Dev: {valid_data.std():.2f}bp"),
                            html.P(f"Current: {valid_data.iloc[-1]:.2f}bp"),
                        ], md=4),
                    ])
                ])
            ])
            
            return fig, stats
        
        # Curvature Analysis callback
        @self.app.callback(
            [Output('curvature-chart', 'figure'),
             Output('curvature-stats', 'children')],
            [Input('curvature-selector', 'value')]
        )
        def update_curvature_analysis(curvature_name):
            if not curvature_name or curvature_name not in self.analytics.curvature:
                return go.Figure(), html.Div("No data available")
            
            # Get curvature data
            curvature_data = self.analytics.curvature[curvature_name]
            valid_data = curvature_data.dropna()
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=valid_data.index, 
                y=valid_data.values,
                mode='lines+markers',
                name=curvature_name,
                line=dict(color=COLORS[1], width=2),
                marker=dict(size=6)
            ))
            
            # Add reference lines
            mean_val = valid_data.mean()
            fig.add_hline(y=mean_val, line_dash="dash", line_color="white", 
                         annotation_text=f"Mean: {mean_val:.1f}bp")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                title=f'Curvature Analysis: {curvature_name}',
                xaxis_title='Date',
                yaxis_title='Curvature (basis points)',
                height=500,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            # Statistics
            stats = dbc.Card([
                dbc.CardHeader("Curvature Statistics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Mean: {valid_data.mean():.2f}bp"),
                            html.P(f"Median: {valid_data.median():.2f}bp"),
                        ], md=4),
                        dbc.Col([
                            html.P(f"Min: {valid_data.min():.2f}bp"),
                            html.P(f"Max: {valid_data.max():.2f}bp"),
                        ], md=4),
                        dbc.Col([
                            html.P(f"Std Dev: {valid_data.std():.2f}bp"),
                            html.P(f"Current: {valid_data.iloc[-1]:.2f}bp"),
                        ], md=4),
                    ])
                ])
            ])
            
            return fig, stats
        
        # PCA Factors callback
        @self.app.callback(
            [Output('pca-chart', 'figure'),
             Output('pca-stats', 'children'),
             Output('pca-details', 'children')],
            [Input('pca-chart', 'id')]  # Dummy input to trigger callback
        )
        def update_pca_analysis(_):
            if self.analytics.level_slope_curvature is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    template='plotly_dark',
                    title='PCA Decomposition: Insufficient Data',
                    annotations=[dict(
                        text="Insufficient data for PCA analysis. Need at least 3 bonds with sufficient data coverage.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        font=dict(size=16)
                    )],
                    height=400
                )
                error_message = dbc.Alert(
                    [
                        html.H5("Insufficient Data for PCA", className="alert-heading"),
                        html.P("PCA requires at least 3 bonds with sufficient data coverage (70% or more of dates)."),
                        html.P("Please check your data or adjust the data filtering criteria.")
                    ],
                    color="danger"
                )
                return empty_fig, error_message, html.Div()
            
            # Create figure with subplots
            fig = make_subplots(rows=3, cols=1, 
                               subplot_titles=("Level", "Slope", "Curvature"),
                               shared_xaxes=True,
                               vertical_spacing=0.1)
            
            # Add traces for each factor
            for i, factor in enumerate(['Level', 'Slope', 'Curvature']):
                fig.add_trace(
                    go.Scatter(
                        x=self.analytics.level_slope_curvature.index,
                        y=self.analytics.level_slope_curvature[factor],
                        mode='lines',
                        name=factor,
                        line=dict(color=COLORS[i], width=2)
                    ),
                    row=i+1, col=1
                )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                title='PCA Decomposition: Level, Slope, Curvature',
                height=800,
                showlegend=True,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            # Basic statistics
            stats = dbc.Card([
                dbc.CardHeader("PCA Factor Statistics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Explained Variance"),
                            html.P(f"Level: {self.analytics.pca_metadata['explained_variance'][0]:.2%}"),
                            html.P(f"Slope: {self.analytics.pca_metadata['explained_variance'][1]:.2%}"),
                            html.P(f"Curvature: {self.analytics.pca_metadata['explained_variance'][2]:.2%}"),
                            html.P(f"Total: {sum(self.analytics.pca_metadata['explained_variance']):.2%}"),
                        ], md=4),
                        dbc.Col([
                            html.H5("Data Coverage"),
                            html.P(f"Bonds used: {len(self.analytics.pca_metadata['bonds_used'])}"),
                            html.P(f"Data points: {self.analytics.pca_metadata['data_points']}"),
                            html.P(f"Missing data: {self.analytics.pca_metadata['missing_data_pct']:.2f}%"),
                        ], md=4),
                        dbc.Col([
                            html.H5("Interpretation"),
                            html.P("Level: Parallel shifts in the entire curve"),
                            html.P("Slope: Changes in the steepness of the curve"),
                            html.P("Curvature: Changes in the middle of the curve relative to the ends"),
                        ], md=4),
                    ])
                ])
            ])
            
            # Create component loadings visualization if available
            details = html.Div()
            if self.analytics.pca_metadata['components'] is not None:
                # Convert component loadings to a format suitable for visualization
                components_df = self.analytics.pca_metadata['components']
                
                # Create a heatmap of component loadings
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=components_df.values,
                    x=components_df.columns,
                    y=components_df.index,
                    colorscale='Viridis',
                    colorbar=dict(title='Loading')
                ))
                
                heatmap_fig.update_layout(
                    template='plotly_dark',
                    title='PCA Component Loadings (Eigenvectors)',
                    xaxis_title='Bond',
                    yaxis_title='Component',
                    height=400,
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                
                # Create a table of bonds used
                bonds_used = self.analytics.pca_metadata['bonds_used']
                bonds_table = dash_table.DataTable(
                    id='bonds-table',
                    columns=[{'name': 'Bonds Used in PCA', 'id': 'bond'}],
                    data=[{'bond': bond} for bond in bonds_used],
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '10px'
                    },
                    style_table={
                        'maxHeight': '300px',
                        'overflowY': 'auto'
                    }
                )
                
                # Combine the heatmap and table into a detailed view
                details = dbc.Card([
                    dbc.CardHeader("PCA Technical Details"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("PCA Component Loadings"),
                                html.P("Shows how each bond contributes to each principal component"),
                                dcc.Graph(figure=heatmap_fig)
                            ], md=8),
                            dbc.Col([
                                html.H5("Bonds Used in Analysis"),
                                html.P(f"Total: {len(bonds_used)} bonds with ≥70% data coverage"),
                                bonds_table,
                                html.Hr(),
                                html.H5("PCA Assumptions"),
                                html.Ul([
                                    html.Li("Data is standardized (mean=0, std=1)"),
                                    html.Li("Missing values are filled using forward/backward fill"),
                                    html.Li("Bonds require at least 70% data coverage"),
                                    html.Li("Components are orthogonal (uncorrelated)"),
                                ])
                            ], md=4),
                        ])
                    ])
                ])
            
            return fig, stats, details
        
        # Yield Curve Evolution callback
        @self.app.callback(
            Output('evolution-chart', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_evolution_chart(start_date, end_date):
            if not start_date or not end_date:
                return go.Figure()
            
            # Convert string dates to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Filter pivot table by date range
            mask = (self.data.pivot.index >= start_date) & (self.data.pivot.index <= end_date)
            filtered_pivot = self.data.pivot[mask]
            
            # Get unique dates in range
            dates_in_range = filtered_pivot.index.unique()
            
            # Create 3D surface plot
            fig = go.Figure()
            
            # Get all bonds including REPO and SATB levels
            # First, filter to ensure we have only valid bonds with data
            valid_bonds = [col for col in self.data.pivot.columns if not filtered_pivot[col].isna().all()]
            
            # Define a custom sorting function to order bonds by maturity
            def get_maturity_value(bond):
                # REPO is treated as the shortest maturity (0)
                if bond == 'REPO':
                    return 0
                # SATB bonds (e.g., SATB3M, SATB6M, SATB12M)
                elif bond.startswith('SATB') and bond[4:-1].isdigit():
                    return float(bond[4:-1])/12  # Convert months to years
                # ZARGB bonds (e.g., ZARGB2, ZARGB5, ZARGB10)
                elif bond.startswith('ZARGB') and bond[5:].isdigit():
                    return float(bond[5:])  # Already in years
                # Any other bonds
                else:
                    return 999  # Place at the end
            
            # Sort bonds by maturity
            bonds = sorted(valid_bonds, key=get_maturity_value)
            
            # Create a surface plot if we have enough data
            if len(dates_in_range) >= 3 and len(bonds) >= 3:
                # Sample dates if there are too many (for better visualization)
                max_dates_to_show = 15  # Maximum number of dates to display for better readability
                if len(dates_in_range) > max_dates_to_show:
                    # Use evenly spaced dates
                    indices = np.round(np.linspace(0, len(dates_in_range) - 1, max_dates_to_show)).astype(int)
                    sampled_dates = dates_in_range[indices]
                else:
                    sampled_dates = dates_in_range
                
                # Create meshgrid for 3D surface
                x_mesh, y_mesh = np.meshgrid(
                    np.arange(len(bonds)), 
                    np.arange(len(sampled_dates))
                )
                
                # Create z values (yields)
                z_values = np.zeros((len(sampled_dates), len(bonds)))
                for i, date in enumerate(sampled_dates):
                    for j, bond in enumerate(bonds):
                        z_values[i, j] = filtered_pivot.loc[date, bond] if date in filtered_pivot.index and bond in filtered_pivot.columns and not pd.isna(filtered_pivot.loc[date, bond]) else np.nan
                
                # Format bond labels for better readability
                formatted_bond_labels = []
                for bond in bonds:
                    if bond == 'REPO':
                        formatted_bond_labels.append('REPO')
                    elif bond.startswith('ZARGB') and bond[5:].isdigit():
                        formatted_bond_labels.append(f"{bond[5:]}Y")
                    elif bond.startswith('SATB') and bond[4:-1].isdigit():
                        months = int(bond[4:-1])
                        formatted_bond_labels.append(f"{months}M")
                    else:
                        formatted_bond_labels.append(bond)
                
                # Format date labels - use a more readable format
                formatted_date_labels = [d.strftime('%d %b %Y') for d in sampled_dates]
                
                # Add surface plot with improved styling
                fig.add_trace(go.Surface(
                    z=z_values,
                    x=x_mesh,
                    y=y_mesh,
                    colorscale='Viridis',
                    opacity=0.9,  # Slightly increased opacity
                    showscale=True,
                    colorbar=dict(
                        title='Yield (%)',
                        titlefont=dict(size=14),
                        tickfont=dict(size=12)
                    ),
                    lighting=dict(
                        ambient=0.6,  # Increase ambient light for better visibility
                        diffuse=0.8,
                        specular=0.2,
                        roughness=0.5,
                        fresnel=0.2
                    ),
                    contours=dict(
                        z=dict(show=True, usecolormap=True, project_z=True, width=2)
                    )
                ))
                
                # Update layout with improved styling
                fig.update_layout(
                    template='plotly_dark',
                    title={
                        'text': 'Yield Curve Evolution',
                        'font': {'size': 24, 'color': 'white'},
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    scene=dict(
                        xaxis=dict(
                            title={
                                'text': 'Bond Maturity',
                                'font': {'size': 14}
                            },
                            tickvals=list(range(len(bonds))),
                            ticktext=formatted_bond_labels,
                            tickfont=dict(size=11),
                            gridcolor='rgba(255, 255, 255, 0.1)',
                            showbackground=True,
                            backgroundcolor='rgba(0, 0, 0, 0.3)'
                        ),
                        yaxis=dict(
                            title={
                                'text': 'Date',
                                'font': {'size': 14}
                            },
                            tickvals=list(range(len(sampled_dates))),
                            ticktext=formatted_date_labels,
                            tickfont=dict(size=11),
                            gridcolor='rgba(255, 255, 255, 0.1)',
                            showbackground=True,
                            backgroundcolor='rgba(0, 0, 0, 0.3)'
                        ),
                        zaxis=dict(
                            title={
                                'text': 'Yield (%)',
                                'font': {'size': 14}
                            },
                            tickfont=dict(size=11),
                            gridcolor='rgba(255, 255, 255, 0.1)',
                            showbackground=True,
                            backgroundcolor='rgba(0, 0, 0, 0.3)'
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=-1.5, z=0.8)  # Adjusted camera angle for better view
                        ),
                        aspectratio=dict(x=1.2, y=1.5, z=0.8)  # Adjusted aspect ratio
                    ),
                    height=700,
                    margin=dict(l=40, r=40, t=80, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            else:
                # Fallback to 2D plot if not enough data
                for bond in bonds:
                    if bond in filtered_pivot.columns:
                        fig.add_trace(go.Scatter(
                            x=filtered_pivot.index,
                            y=filtered_pivot[bond],
                            mode='lines+markers',
                            name=bond
                        ))
                
                fig.update_layout(
                    template='plotly_dark',
                    title='Yield Curve Evolution (2D View)',
                    xaxis_title='Date',
                    yaxis_title='Yield (%)',
                    height=500,
                    margin=dict(l=40, r=40, t=80, b=40)
                )
            
            return fig
        
        # Individual Bond Analysis callback
        @self.app.callback(
            [Output('bond-chart', 'figure'),
             Output('bond-stats', 'children')],
            [Input('bond-selector', 'value')]
        )
        def update_bond_analysis(bond_name):
            if not bond_name or bond_name not in self.data.pivot.columns:
                return go.Figure(), html.Div("No data available")
            
            # Get bond data
            bond_data = self.data.pivot[bond_name].dropna()
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bond_data.index, 
                y=bond_data.values,
                mode='lines+markers',
                name=bond_name,
                line=dict(color=COLORS[3], width=2),
                marker=dict(size=6)
            ))
            
            # Add rolling mean
            rolling_mean = bond_data.rolling(window=20, min_periods=5).mean()
            fig.add_trace(go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean.values,
                mode='lines',
                name='20-day MA',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash')
            ))
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                title=f'Bond Analysis: {bond_name}',
                xaxis_title='Date',
                yaxis_title='Yield (%)',
                height=500,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            # Statistics
            stats = dbc.Card([
                dbc.CardHeader(f"Bond Statistics: {bond_name}"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Mean: {bond_data.mean():.2f}%"),
                            html.P(f"Median: {bond_data.median():.2f}%"),
                        ], md=4),
                        dbc.Col([
                            html.P(f"Min: {bond_data.min():.2f}%"),
                            html.P(f"Max: {bond_data.max():.2f}%"),
                        ], md=4),
                        dbc.Col([
                            html.P(f"Std Dev: {bond_data.std():.2f}%"),
                            html.P(f"Current: {bond_data.iloc[-1]:.2f}%"),
                        ], md=4),
                    ])
                ])
            ])
            
            return fig, stats
    
    def run_server(self, debug=True, host='0.0.0.0', port=8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, host=host, port=port)


# Main execution
if __name__ == '__main__':
    print("Loading yield curve data...")
    curve_data = YieldCurveData()
    
    print("Calculating analytics...")
    analytics = CurveAnalytics(curve_data)
    
    print("Setting up dashboard...")
    dashboard = CurveDashboard(curve_data, analytics)
    
    print("Starting server at http://0.0.0.0:8050")
    dashboard.run_server()
