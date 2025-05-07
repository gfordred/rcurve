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
