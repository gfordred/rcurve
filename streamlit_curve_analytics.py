"""
Professional Yield Curve Analytics - Streamlit Version
=====================================================
Advanced analytics for fixed income yield curve shape and evolution analysis.
Focuses on complete data integrity and professional financial metrics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Constants
DATE_FORMAT = '%d/%m/%Y'
COLORS = px.colors.qualitative.Dark24

# Set page config
st.set_page_config(
    page_title="Professional Yield Curve Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme styling
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e2e2e;
        border-bottom: 2px solid #4CAF50;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    .stAlert {
        background-color: #2e2e2e;
        color: white;
    }
    .stDataFrame {
        background-color: #2e2e2e;
    }
</style>
""", unsafe_allow_html=True)


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
        st.sidebar.write(f"Total dates in dataset: {total_dates}")
        data_completeness = []
        for pair, mask in self.complete_data_mask.items():
            complete_count = mask.sum()
            if complete_count > 0:
                data_completeness.append(f"Pair {pair}: {complete_count}/{total_dates} complete dates ({complete_count/total_dates:.1%})")
        
        if st.sidebar.checkbox("Show Data Completeness", False):
            st.sidebar.write("Data Completeness:")
            for item in data_completeness[:10]:  # Show only first 10 to avoid clutter
                st.sidebar.write(item)
            if len(data_completeness) > 10:
                st.sidebar.write(f"... and {len(data_completeness)-10} more pairs")


class CurveAnalytics:
    """Professional yield curve analytics with strict data integrity."""
    
    def __init__(self, curve_data):
        """Initialize with curve data object."""
        self.data = curve_data
        self.spreads = {}
        self.curvature = {}
        self.level_slope_curvature = None
        self.pca_metadata = None
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
            
            st.sidebar.success(f"PCA analysis complete using {len(valid_bonds)} bonds")
            st.sidebar.info(f"PCA explained variance: Level={explained_var[0]:.2%}, Slope={explained_var[1]:.2%}, Curvature={explained_var[2]:.2%}")
        else:
            st.sidebar.warning(f"Insufficient data for PCA: only {len(valid_bonds)} bonds have sufficient data (need at least 3)")
            self.level_slope_curvature = None


# Main Streamlit Application
def main():
    st.title("Professional Yield Curve Analytics")
    st.markdown("Advanced analytics for fixed income yield curve shape and evolution analysis")
    
    # Load data
    with st.spinner("Loading yield curve data..."):
        curve_data = YieldCurveData()
        analytics = CurveAnalytics(curve_data)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Spread Analysis", "Curvature Analysis", "PCA Factors", "Curve Evolution", "Bond Analysis"])
    
    # Tab 1: Spread Analysis
    with tab1:
        st.header("Yield Curve Spreads")
        st.markdown("Analysis of key yield spreads with complete data integrity")
        
        # Spread selector
        spread_options = list(analytics.spreads.keys())
        if spread_options:
            selected_spread = st.selectbox("Select Spread:", spread_options)
            spread_data = analytics.spreads[selected_spread].dropna()
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spread_data.index, 
                y=spread_data.values,
                mode='lines+markers',
                name=selected_spread,
                line=dict(color=COLORS[0], width=2),
                marker=dict(size=6)
            ))
            
            # Add reference lines
            mean_val = spread_data.mean()
            fig.add_hline(y=mean_val, line_dash="dash", line_color="white", 
                         annotation_text=f"Mean: {mean_val:.1f}bp")
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                title=f'Spread Analysis: {selected_spread}',
                xaxis_title='Date',
                yaxis_title='Spread (basis points)',
                height=500,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{spread_data.mean():.2f}bp")
                st.metric("Median", f"{spread_data.median():.2f}bp")
            with col2:
                st.metric("Min", f"{spread_data.min():.2f}bp")
                st.metric("Max", f"{spread_data.max():.2f}bp")
            with col3:
                st.metric("Std Dev", f"{spread_data.std():.2f}bp")
                st.metric("Current", f"{spread_data.iloc[-1]:.2f}bp")
        else:
            st.warning("No spread data available")
    
    # Tab 2: Curvature Analysis
    with tab2:
        st.header("Yield Curve Curvature")
        st.markdown("Analysis of yield curve shape and convexity")
        
        # Curvature selector
        curvature_options = list(analytics.curvature.keys())
        if curvature_options:
            selected_curvature = st.selectbox("Select Curvature Metric:", curvature_options)
            curvature_data = analytics.curvature[selected_curvature].dropna()
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curvature_data.index, 
                y=curvature_data.values,
                mode='lines+markers',
                name=selected_curvature,
                line=dict(color=COLORS[1], width=2),
                marker=dict(size=6)
            ))
            
            # Add reference lines
            mean_val = curvature_data.mean()
            fig.add_hline(y=mean_val, line_dash="dash", line_color="white", 
                         annotation_text=f"Mean: {mean_val:.1f}bp")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                title=f'Curvature Analysis: {selected_curvature}',
                xaxis_title='Date',
                yaxis_title='Curvature (basis points)',
                height=500,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{curvature_data.mean():.2f}bp")
                st.metric("Median", f"{curvature_data.median():.2f}bp")
            with col2:
                st.metric("Min", f"{curvature_data.min():.2f}bp")
                st.metric("Max", f"{curvature_data.max():.2f}bp")
            with col3:
                st.metric("Std Dev", f"{curvature_data.std():.2f}bp")
                st.metric("Current", f"{curvature_data.iloc[-1]:.2f}bp")
        else:
            st.warning("No curvature data available")
    
    # Tab 3: PCA Factors
    with tab3:
        st.header("Level, Slope, Curvature Decomposition")
        st.markdown("Principal component analysis of yield curve movements")
        
        # PCA methodology explanation
        with st.expander("PCA Methodology", expanded=True):
            st.markdown("""
            **Principal Component Analysis (PCA)** is a statistical technique that identifies the main patterns of variation in yield curves. 
            The first three components typically correspond to:
            
            - **Level (PC1)**: Parallel shifts in the entire yield curve
            - **Slope (PC2)**: Changes in the steepness of the yield curve
            - **Curvature (PC3)**: Changes in the curvature (belly) of the yield curve
            
            These three factors typically explain 95-99% of all yield curve movements.
            """)
        
        if analytics.level_slope_curvature is not None:
            # Create figure with subplots
            fig = make_subplots(rows=3, cols=1, 
                               subplot_titles=("Level", "Slope", "Curvature"),
                               shared_xaxes=True,
                               vertical_spacing=0.1)
            
            # Add traces for each factor
            for i, factor in enumerate(['Level', 'Slope', 'Curvature']):
                fig.add_trace(
                    go.Scatter(
                        x=analytics.level_slope_curvature.index,
                        y=analytics.level_slope_curvature[factor],
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
            
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA Statistics
            st.subheader("PCA Factor Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Explained Variance**")
                st.write(f"Level: {analytics.pca_metadata['explained_variance'][0]:.2%}")
                st.write(f"Slope: {analytics.pca_metadata['explained_variance'][1]:.2%}")
                st.write(f"Curvature: {analytics.pca_metadata['explained_variance'][2]:.2%}")
                st.write(f"Total: {sum(analytics.pca_metadata['explained_variance']):.2%}")
            
            with col2:
                st.markdown("**Data Coverage**")
                st.write(f"Bonds used: {len(analytics.pca_metadata['bonds_used'])}")
                st.write(f"Data points: {analytics.pca_metadata['data_points']}")
                st.write(f"Missing data: {analytics.pca_metadata['missing_data_pct']:.2f}%")
            
            with col3:
                st.markdown("**Interpretation**")
                st.write("Level: Parallel shifts in the entire curve")
                st.write("Slope: Changes in the steepness of the curve")
                st.write("Curvature: Changes in the middle of the curve relative to the ends")
            
            # Component loadings visualization
            if analytics.pca_metadata['components'] is not None:
                st.subheader("PCA Technical Details")
                
                # Convert component loadings to a format suitable for visualization
                components_df = analytics.pca_metadata['components']
                
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
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**PCA Component Loadings**")
                    st.markdown("Shows how each bond contributes to each principal component")
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Bonds Used in Analysis**")
                    st.markdown(f"Total: {len(analytics.pca_metadata['bonds_used'])} bonds with ≥70% data coverage")
                    st.dataframe(pd.DataFrame({"Bond": analytics.pca_metadata['bonds_used']}))
                    
                    st.markdown("**PCA Assumptions**")
                    st.markdown("""
                    - Data is standardized (mean=0, std=1)
                    - Missing values are filled using forward/backward fill
                    - Bonds require at least 70% data coverage
                    - Components are orthogonal (uncorrelated)
                    """)
        else:
            st.error("Insufficient data for PCA analysis. Need at least 3 bonds with sufficient data coverage.")
            st.warning("PCA requires at least 3 bonds with sufficient data coverage (70% or more of dates).")
            st.info("Please check your data or adjust the data filtering criteria.")
    
    # Tab 4: Yield Curve Evolution
    with tab4:
        st.header("Yield Curve Evolution")
        st.markdown("Analysis of yield curve shape changes over time")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                pd.Timestamp(curve_data.unique_dates[0]).date(),
                min_value=pd.Timestamp(curve_data.unique_dates[0]).date(),
                max_value=pd.Timestamp(curve_data.unique_dates[-1]).date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                pd.Timestamp(curve_data.unique_dates[-1]).date(),
                min_value=pd.Timestamp(curve_data.unique_dates[0]).date(),
                max_value=pd.Timestamp(curve_data.unique_dates[-1]).date()
            )
        
        # Convert to pandas Timestamp
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Filter pivot table by date range
        mask = (curve_data.pivot.index >= start_date) & (curve_data.pivot.index <= end_date)
        filtered_pivot = curve_data.pivot[mask]
        
        # Get unique dates in range
        dates_in_range = filtered_pivot.index.unique()
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Get all bonds including REPO and SATB levels
        # First, filter to ensure we have only valid bonds with data
        valid_bonds = [col for col in curve_data.pivot.columns if not filtered_pivot[col].isna().all()]
        
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
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Individual Bond Analysis
    with tab5:
        st.header("Individual Bond Analysis")
        st.markdown("Detailed analysis of individual bond yields")
        
        # Bond selector
        bond_options = list(curve_data.pivot.columns)
        if bond_options:
            selected_bond = st.selectbox("Select Bond:", bond_options)
            bond_data = curve_data.pivot[selected_bond].dropna()
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bond_data.index, 
                y=bond_data.values,
                mode='lines+markers',
                name=selected_bond,
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
                title=f'Bond Analysis: {selected_bond}',
                xaxis_title='Date',
                yaxis_title='Yield (%)',
                height=500,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{bond_data.mean():.2f}%")
                st.metric("Median", f"{bond_data.median():.2f}%")
            with col2:
                st.metric("Min", f"{bond_data.min():.2f}%")
                st.metric("Max", f"{bond_data.max():.2f}%")
            with col3:
                st.metric("Std Dev", f"{bond_data.std():.2f}%")
                st.metric("Current", f"{bond_data.iloc[-1]:.2f}%")
        else:
            st.warning("No bond data available")


if __name__ == '__main__':
    main()
