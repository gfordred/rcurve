
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from scipy.interpolate import interp1d, CubicSpline
import QuantLib as ql
from datetime import datetime

# ---- R-BOND METADATA: Maturity (dd/mm/yyyy) and Coupon ----
R_BOND_META = {
    'R207': {'maturity': '15/01/2020', 'coupon': 7.25},
    'R208': {'maturity': '31/03/2021', 'coupon': 6.75},
    'R2023': {'maturity': '28/02/2023', 'coupon': 7.75},
    'R186': {'maturity': '21/12/2026', 'coupon': 10.5},
    'R2030': {'maturity': '31/01/2030', 'coupon': 8.0},
    'R213': {'maturity': '28/02/2031', 'coupon': 7.0},
    'R209': {'maturity': '31/03/2036', 'coupon': 6.25},
    'R2032': {'maturity': '31/03/2032', 'coupon': 8.25},
    'R2035': {'maturity': '28/02/2035', 'coupon': 8.875},
    'R2037': {'maturity': '31/01/2037', 'coupon': 8.5},
    'R2040': {'maturity': '31/01/2040', 'coupon': 9.0},
    'R214': {'maturity': '28/02/2041', 'coupon': 6.5},
    'R2044': {'maturity': '31/01/2044', 'coupon': 8.75},
    'R2048': {'maturity': '28/02/2048', 'coupon': 8.75},
    'R2053': {'maturity': '31/03/2053', 'coupon': 11.625},
    'R2033': {'maturity': '31/03/2033', 'coupon': 8.5},
    # Add ZARGB and SATB as needed if metadata available
}

ZA_CALENDAR = ql.SouthAfrica()


# ----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING
# ----------------------------------------------------------------------------

def load_data(path: str = 'rbond_data_with_maturity.csv') -> pd.DataFrame:
    """Load SA bond data, merge R_BOND_META, and compute QuantLib analytics columns."""
    import time
    start_total = time.time()
    print(f"[Startup] Loading data from {path} ...")
    t0 = time.time()
    df = pd.read_csv(path, parse_dates=['report_date'])
    print(f"[Startup] CSV loaded in {time.time() - t0:.2f} seconds (shape={df.shape})")
    t1 = time.time()
    df['bond_yield'] = df['bond_yield'].astype(float)
    df['years_to_maturity'] = df['years_to_maturity'].astype(float)

    # Merge R_BOND_META: maturity and coupon
    # Dynamic metadata for ZARGB, SATC3M, REPO
    import re
    def dynamic_meta(row):
        b = row['bond']
        report_date = row['report_date']
        if b.startswith('ZARGB') and b[5:].isdigit():
            n = int(b[5:])
            maturity_dt = report_date + pd.DateOffset(years=n)
            coupon = row['bond_yield']  # par coupon = yield
            return pd.Series({'exact_maturity': maturity_dt.strftime('%d/%m/%Y'), 'coupon': coupon})
        # SATB{n}M: n months from report_date, zero coupon
        m = re.match(r'SATB(\d+)M', b)
        if m:
            n_months = int(m.group(1))
            maturity_dt = report_date + pd.DateOffset(months=n_months)
            return pd.Series({'exact_maturity': maturity_dt.strftime('%d/%m/%Y'), 'coupon': 0.0})
        elif b.startswith('SATC3M'):
            maturity_dt = report_date + pd.DateOffset(months=3)
            return pd.Series({'exact_maturity': maturity_dt.strftime('%d/%m/%Y'), 'coupon': 0.0})
        elif b == 'REPO':
            maturity_dt = report_date + pd.DateOffset(days=1)
            return pd.Series({'exact_maturity': maturity_dt.strftime('%d/%m/%Y'), 'coupon': 0.0})
        else:
            meta = R_BOND_META.get(b, {})
            return pd.Series({'exact_maturity': meta.get('maturity', None), 'coupon': meta.get('coupon', np.nan)})
    t2 = time.time()
    df[['exact_maturity', 'coupon']] = df.apply(dynamic_meta, axis=1)
    print(f"[Startup] Metadata merged in {time.time() - t2:.2f} seconds")

    # Simple approximations for all rows
    def approx_duration(row):
        b = row['bond']
        # SATB{n}M: duration = years_to_maturity (zero coupon)
        if re.match(r'SATB(\d+)M', b):
            return row['years_to_maturity']
        # REPO: duration = 0
        if b == 'REPO':
            return 0.0
        # Default
        return row['years_to_maturity'] / (1 + row['bond_yield'] / 100)
    t3 = time.time()
    df['mod_duration'] = df.apply(approx_duration, axis=1)
    df['dv01'] = df['mod_duration'] * 100 * 0.0001  # assuming price ~100
    print(f"[Startup] Duration/DV01 calculated in {time.time() - t3:.2f} seconds")
    df['ql_price'] = np.nan
    df['ql_mod_duration'] = np.nan
    df['ql_dv01'] = np.nan

    # QuantLib only for latest date per bond
    t4 = time.time()
    latest_idx = df.sort_values('report_date').groupby('bond')['report_date'].idxmax()
    latest_rows = df.loc[latest_idx].copy()
    def quantlib_risk(row):
        if pd.isnull(row['exact_maturity']) or pd.isnull(row['coupon']) or row['years_to_maturity'] < 0.01:
            return pd.Series({'ql_price': np.nan, 'ql_mod_duration': np.nan, 'ql_dv01': np.nan})
        try:
            settle = ql.Date(row['report_date'].day, row['report_date'].month, row['report_date'].year)
            maturity_dt = datetime.strptime(row['exact_maturity'], '%d/%m/%Y')
            maturity = ql.Date(maturity_dt.day, maturity_dt.month, maturity_dt.year)
            freq = ql.Annual if row['bond'].startswith('ZARGB') else ql.Annual
            schedule = ql.Schedule(settle, maturity, ql.Period(freq), ZA_CALENDAR, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
            bond = ql.FixedRateBond(0, 100, schedule, [row['coupon']/100.0], ql.Actual365Fixed())
            ytm = row['bond_yield']/100.0
            price = ql.BondFunctions.cleanPrice(bond, ytm, ql.Actual365Fixed(), ql.Compounded, ql.Annual)
            mod_duration = ql.BondFunctions.duration(bond, ytm, ql.Actual365Fixed(), ql.Compounded, ql.Annual, ql.Duration.Modified)
            dv01 = mod_duration * price * 0.0001
            return pd.Series({'ql_price': price, 'ql_mod_duration': mod_duration, 'ql_dv01': dv01})
        except Exception as e:
            print(f"[Startup][QuantLib] Error: {e}")
            return pd.Series({'ql_price': np.nan, 'ql_mod_duration': np.nan, 'ql_dv01': np.nan})
    ql_risk = latest_rows.apply(quantlib_risk, axis=1)
    df.loc[latest_idx, ['ql_price', 'ql_mod_duration', 'ql_dv01']] = ql_risk.values
    print(f"[Startup] QuantLib risk calculated in {time.time() - t4:.2f} seconds")
    print(f"[Startup] TOTAL data load time: {time.time() - start_total:.2f} seconds")
    return df


import time
startup_start = time.time()
try:
    df_bonds = load_data()
    print(f"[Startup] df_bonds loaded. Shape: {df_bonds.shape}")
    unique_dates = np.sort(df_bonds['report_date'].unique())
    unique_bonds = np.sort(df_bonds['bond'].unique())
    print(f"[Startup] unique_dates: {len(unique_dates)}, unique_bonds: {len(unique_bonds)}")
    print(f"[Startup] unique_bonds sample: {unique_bonds[:10]}")
except Exception as e:
    print(f"[Startup][ERROR] Exception during data loading: {e}")
    raise
print(f"[Startup] Data loading and preprocessing finished in {time.time() - startup_start:.2f} seconds")

# ----------------------------------------------------------------------------
# APP INITIALISATION – Dash + Bootstrap (dark theme)
# ----------------------------------------------------------------------------
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets, title="SA Fixed Income Dashboard")
server = app.server  # for gunicorn / deployment

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def interpolate_curve(x: np.ndarray, y: np.ndarray, method: str = 'Cubic Spline'):
    """Return dense x_new, y_new according to chosen interpolation method. Returns empty if insufficient or non-finite data."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.array([]), np.array([])
    x_sorted_idx = np.argsort(x)
    x, y = x[x_sorted_idx], y[x_sorted_idx]
    x_new = np.linspace(x.min(), x.max(), 200)
    try:
        if method == 'Linear':
            f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        else:
            f = CubicSpline(x, y, extrapolate=True)
        y_new = f(x_new)
    except Exception as e:
        x_new, y_new = np.array([]), np.array([])
    return x_new, y_new



def build_yield_curve_fig(date: np.datetime64, method: str = 'Cubic Spline') -> go.Figure:
    """Create yield curve scatter + interpolation line for a single date."""
    daily = df_bonds[df_bonds['report_date'] == date]
    x = daily['years_to_maturity'].values
    y = daily['bond_yield'].values

    x_new, y_new = interpolate_curve(x, y, method)

    fig = go.Figure()
    # Bond name labels
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers+text', name='Observed',
        marker=dict(size=10, color='orange', line=dict(width=1, color='black')),
        text=daily['bond'], textposition='top center',
        hovertemplate='<b>%{text}</b><br>Maturity: %{x:.2f}y<br>Yield: %{y:.2f}%'
    ))
    fig.add_trace(go.Scatter(
        x=x_new, y=y_new, mode='lines', name=f'{method} Fit',
        line=dict(width=3, color='cyan')
    ))
    fig.update_layout(
        template='plotly_dark',
        title=f'<b>South Africa Govt Yield Curve</b> – {pd.to_datetime(date).date()}',
        xaxis_title='Years to Maturity',
        yaxis_title='Yield (%)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def build_bond_history_fig(bond: str) -> go.Figure:
    """Yield history for selected bond."""
    series = df_bonds[df_bonds['bond'] == bond].sort_values('report_date')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series['report_date'], y=series['bond_yield'], mode='lines+markers+text',
        text=series['bond'], textposition='top center',
        marker=dict(size=8, color='orange', line=dict(width=1, color='black')),
        line=dict(width=2, color='cyan'),
        hovertemplate='<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>Yield: %{y:.2f}%'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=f'<b>Historical Yield</b> – {bond}',
        xaxis_title='Date',
        yaxis_title='Yield (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def build_curve_evolution_fig(method: str = 'Cubic Spline') -> go.Figure:
    """Animated yield curve evolution across all dates. Handles missing/non-finite data robustly."""
    # Build initial data
    first_date = unique_dates[0]
    x_init, y_init = interpolate_curve(
        df_bonds[df_bonds['report_date'] == first_date]['years_to_maturity'].values,
        df_bonds[df_bonds['report_date'] == first_date]['bond_yield'].values,
        method,
    )
    fig = go.Figure(
        data=[go.Scatter(x=x_init, y=y_init, mode='lines', name=str(first_date)[:10])],
        layout=go.Layout(
            template='plotly_dark',
            xaxis=dict(title='Years to Maturity'),
            yaxis=dict(title='Yield (%)'),
            title='Yield Curve Evolution',
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=1.05,
                xanchor='right',
                yanchor='top',
                buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}])]
            )]
        ),
        frames=[
            go.Frame(
                data=[go.Scatter(
                    x=interpolate_curve(
                        df_bonds[df_bonds['report_date'] == d]['years_to_maturity'].values,
                        df_bonds[df_bonds['report_date'] == d]['bond_yield'].values,
                        method,
                    )[0],
                    y=interpolate_curve(
                        df_bonds[df_bonds['report_date'] == d]['years_to_maturity'].values,
                        df_bonds[df_bonds['report_date'] == d]['bond_yield'].values,
                        method,
                    )[1],
                    mode='lines')],
                name=str(d)
            ) for d in unique_dates if len(interpolate_curve(
                df_bonds[df_bonds['report_date'] == d]['years_to_maturity'].values,
                df_bonds[df_bonds['report_date'] == d]['bond_yield'].values,
                method,
            )[0]) > 1
        ]
    )
    return fig

# ----------------------------------------------------------------------------
# UI COMPONENTS
# ----------------------------------------------------------------------------

curve_controls = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Label("Report Date"), md=3),
                dbc.Col(dcc.DatePickerSingle(
                    id='date-picker',
                    min_date_allowed=df_bonds['report_date'].min(),
                    max_date_allowed=df_bonds['report_date'].max(),
                    date=df_bonds['report_date'].max(),
                    display_format='YYYY-MM-DD'
                ), md=9),
            ], align='center', className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label("Interpolation"), md=3),
                dbc.Col(dcc.RadioItems(
                    id='interp-method',
                    options=[{'label': m, 'value': m} for m in ['Linear', 'Cubic Spline']],
                    value='Cubic Spline',
                    inline=True,
                    inputStyle={'margin-right': '6px'}
                ), md=9)
            ]
        )
    ], body=True, className='mb-4'
)

yield_curve_tab = dbc.Container(
    [curve_controls, dcc.Graph(id='yield-curve-graph')],
    fluid=True
)

print("[Startup] Building bond_analytics_tab ...")
bond_analytics_tab = dbc.Container(
    [
        dbc.Row([
            dbc.Col(dbc.Label("Select Bond"), md=2),
            dbc.Col(dcc.Dropdown(
                id='bond-dropdown',
                options=[{'label': b, 'value': b} for b in unique_bonds[:20]],  # limit for debug
                value=unique_bonds[0]
            ), md=10)
        ], className='mb-4'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(id='latest-yield', className='card-title'),
                html.Small("Latest Yield (%)")
            ]), color='dark', inverse=True), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(id='latest-duration', className='card-title'),
                html.Small("Mod. Duration (yrs)")
            ]), color='dark', inverse=True), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(id='latest-dv01', className='card-title'),
                html.Small("DV01")
            ]), color='dark', inverse=True), md=4),
        ], className='mb-4'),
        dcc.Graph(id='bond-history-graph')
    ],
    fluid=True
)
print("[Startup] bond_analytics_tab built.")

print("[Startup] Building curve_evolution_tab ...")
curve_evolution_tab = dbc.Container([
    dcc.Graph(id='curve-evolution-graph', figure=build_curve_evolution_fig())
], fluid=True)
print("[Startup] curve_evolution_tab built.")

from dash import dash_table

# ---- Curve Shape Analytics Tab ----
def build_curve_shape_analytics():
    import time
    start = time.time()
    print("[ShapeAnalytics] Starting build_curve_shape_analytics...")
    # Required ZARGB bonds
    required_bonds = ["ZARGB2","ZARGB5","ZARGB10","ZARGB20","ZARGB25","ZARGB30"]
    present_bonds = set(df_bonds['bond'].unique())
    missing_bonds = [b for b in required_bonds if b not in present_bonds]
    print(f"[ShapeAnalytics] Required bonds missing: {missing_bonds}")
    print(f"[ShapeAnalytics] Number of unique dates: {len(unique_dates)}")
    # Drop duplicate (report_date, bond) pairs before pivot
    t_pivot = time.time()
    n_before = len(df_bonds)
    df_bonds_nodup = df_bonds.drop_duplicates(subset=['report_date', 'bond'], keep='last')
    n_after = len(df_bonds_nodup)
    if n_after < n_before:
        print(f"[ShapeAnalytics][WARNING] Removed {n_before - n_after} duplicate (report_date, bond) rows before pivoting.")
    # Exclude matured bonds: only include rows where maturity >= report_date
    bond_warned = set()
    def not_matured_curve(row):
        val = row.get('exact_maturity', None)
        if not isinstance(val, str) or pd.isnull(val):
            # Only log once per bond
            b = row.get('bond', 'UNKNOWN')
            if b not in bond_warned:
                print(f"[ShapeAnalytics][WARNING] Bond {b} has invalid or missing maturity, excluded from analytics.")
                bond_warned.add(b)
            return False  # Exclude if no valid string maturity
        try:
            mat = datetime.strptime(val, '%d/%m/%Y')
            return mat >= row['report_date']
        except Exception as e:
            b = row.get('bond', 'UNKNOWN')
            if b not in bond_warned:
                print(f"[ShapeAnalytics][WARNING] Could not parse maturity for bond {b} on {row['report_date']}: {e}")
                bond_warned.add(b)
            return False  # Exclude if cannot parse
    n_before_mature = len(df_bonds_nodup)
    df_bonds_nodup = df_bonds_nodup[df_bonds_nodup.apply(not_matured_curve, axis=1)]
    n_after_mature = len(df_bonds_nodup)
    if n_after_mature < n_before_mature:
        print(f"[ShapeAnalytics][WARNING] Excluded {n_before_mature - n_after_mature} matured bond/date pairs from analytics.")
    pivot = df_bonds_nodup.pivot(index='report_date', columns='bond', values='bond_yield')
    print(f"[ShapeAnalytics] Pivot table created in {time.time()-t_pivot:.2f}s. Shape: {pivot.shape}")
    # Slope
    shape_df = pd.DataFrame({'report_date': unique_dates})
    for name, s1, s2 in [("2s10s", "ZARGB2", "ZARGB10"), ("5s30s", "ZARGB5", "ZARGB30"), ("10s30s", "ZARGB10", "ZARGB30")]:
        shape_df[name] = (pivot[s2] - pivot[s1]) * 100 if s1 in pivot.columns and s2 in pivot.columns else np.nan
    # Curvature
    if all(b in pivot.columns for b in ["ZARGB2","ZARGB10","ZARGB30"]):
        shape_df["2s10s30s_curv"] = ((pivot["ZARGB30"] + pivot["ZARGB2"]) / 2 - pivot["ZARGB10"]) * 100
    else:
        shape_df["2s10s30s_curv"] = np.nan
    if all(b in pivot.columns for b in ["ZARGB5","ZARGB10","ZARGB20"]):
        shape_df["5s10s20s_bfly"] = (pivot["ZARGB10"] - (pivot["ZARGB5"] + pivot["ZARGB20"]) / 2) * 100
    else:
        shape_df["5s10s20s_bfly"] = np.nan
    # Humps
    shape_df['max_slope'] = shape_df[['2s10s','5s30s','10s30s']].max(axis=1)
    shape_df['min_slope'] = shape_df[['2s10s','5s30s','10s30s']].min(axis=1)
    shape_df['max_curv'] = shape_df[['2s10s30s_curv','5s10s20s_bfly']].max(axis=1)
    shape_df['min_curv'] = shape_df[['2s10s30s_curv','5s10s20s_bfly']].min(axis=1)
    # Plots
    fig_slope = go.Figure()
    for col in ['2s10s','5s30s','10s30s']:
        fig_slope.add_trace(go.Scatter(x=shape_df['report_date'], y=shape_df[col], mode='lines', name=col))
    fig_slope.update_layout(template='plotly_dark', title='Curve Slope Time Series', xaxis_title='Date', yaxis_title='Slope (bp)')
    fig_curv = go.Figure()
    for col in ['2s10s30s_curv','5s10s20s_bfly']:
        fig_curv.add_trace(go.Scatter(x=shape_df['report_date'], y=shape_df[col], mode='lines', name=col))
    fig_curv.update_layout(template='plotly_dark', title='Curve Curvature/Butterfly Time Series', xaxis_title='Date', yaxis_title='Curvature (bp)')
    # Table
    table_cols = ['report_date','2s10s','5s30s','10s30s','2s10s30s_curv','5s10s20s_bfly','max_slope','min_slope','max_curv','min_curv']
    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in table_cols],
        data=shape_df[table_cols].round(2).to_dict('records'),
        style_table={'overflowX': 'auto','minWidth':'100%'},
        style_header={'backgroundColor': '#111','color':'#fff','fontWeight':'bold'},
        style_cell={'backgroundColor': '#222','color':'#fff','fontSize':13,'padding':'5px'},
        page_size=20
    )
    print(f"[ShapeAnalytics] Finished in {time.time()-start:.2f}s.")
    return dbc.Container([
        html.H4("Curve Shape Analytics: Slope, Curvature, Butterflies, Humps"),
        dcc.Graph(figure=fig_slope),
        dcc.Graph(figure=fig_curv),
        html.H5("Shape Table (bp)"),
        table
    ], fluid=True)


print("[Startup] Building curve_shape_analytics_tab ...")
curve_shape_analytics_tab = build_curve_shape_analytics()
print("[Startup] curve_shape_analytics_tab built.")

# ---- Bond Summary Table ----
def build_bond_summary_table():
    # Use most recent date as analysis date
    analysis_date = df_bonds['report_date'].max()
    latest = df_bonds[df_bonds['report_date'] == analysis_date].copy()
    # Exclude matured bonds (maturity before analysis_date), even if years_to_maturity is NaN
    n_before = len(latest)
    def not_matured(row):
        try:
            if 'exact_maturity' in row and pd.notnull(row['exact_maturity']):
                mat = datetime.strptime(row['exact_maturity'], '%d/%m/%Y')
                return mat >= analysis_date
            else:
                # If no maturity info, keep by default
                return True
        except Exception as e:
            print(f"[BondSummary][WARNING] Could not parse maturity for bond {row['bond']} on {row['report_date']}: {e}")
            return True
    latest = latest[latest.apply(not_matured, axis=1)]
    n_after = len(latest)
    if n_after < n_before:
        print(f"[BondSummary][INFO] Excluded {n_before-n_after} matured bonds from summary table (maturity before {analysis_date.date()}).")
    # Sort: REPO first, then by years_to_maturity ascending
    latest['sorter'] = np.where(latest['bond']=='REPO', -1, latest['years_to_maturity'])
    latest = latest.sort_values('sorter')
    # Professional column order and formatting
    cols = [
        'bond', 'coupon', 'exact_maturity', 'years_to_maturity', 'bond_yield',
        'ql_price', 'ql_mod_duration', 'ql_dv01', 'mod_duration', 'dv01'
    ]
    colnames = {
        'bond':'Bond', 'coupon':'Coupon', 'exact_maturity':'Maturity', 'years_to_maturity':'Yrs to Maturity',
        'bond_yield':'Yield', 'ql_price':'QL Price', 'ql_mod_duration':'QL Mod Duration', 'ql_dv01':'QL DV01',
        'mod_duration':'Approx Mod Duration', 'dv01':'Approx DV01'
    }
    latest = latest[cols].rename(columns=colnames)
    # Format
    return html.Div([
        html.H5(f"Bond Risk Summary (QuantLib, Analysis Date: {analysis_date.date()})"),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in latest.columns],
            data=latest.round(4).to_dict('records'),
            style_table={'overflowX': 'auto','minWidth':'100%'},
            style_header={'backgroundColor': '#111','color':'#fff','fontWeight':'bold'},
            style_cell={'backgroundColor': '#222','color':'#fff','fontSize':14,'padding':'6px'},
            style_data_conditional=[
                {"if": {"column_id": 'Bond'}, "fontWeight": "bold", "color": "#0ff"},
                {"if": {"column_id": 'Maturity'}, "color": "#ff0"},
                {"if": {"column_id": 'Yrs to Maturity'}, "color": "#0f0"},
            ],
            page_size=30
        )
    ])

print("[Startup] Building bond_summary_tab ...")
bond_summary_tab = dbc.Container([
    html.H4("Bond Risk Summary (QuantLib)"),
    build_bond_summary_table()
], fluid=True)
print("[Startup] bond_summary_tab built.")

# ---- Spread/Shape Section ----
def build_spread_df():
    # Only use ZARGB bonds for spreads, fallback to R if not present
    spread_defs = [
        ("2s10s", "ZARGB2", "ZARGB10"),
        ("5s20s", "ZARGB5", "ZARGB20"),
        ("5s25s", "ZARGB5", "ZARGB25"),
        ("5s30s", "ZARGB5", "ZARGB30"),
        ("10s30s", "ZARGB10", "ZARGB30"),
    ]
    rows = []
    for d in unique_dates:
        row = {'report_date': d}
        for name, s1, s2 in spread_defs:
            y1 = df_bonds[(df_bonds['report_date']==d)&(df_bonds['bond']==s1)]['bond_yield']
            y2 = df_bonds[(df_bonds['report_date']==d)&(df_bonds['bond']==s2)]['bond_yield']
            if not y1.empty and not y2.empty:
                row[name] = (y2.values[0] - y1.values[0])*100
            else:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

spread_df = build_spread_df()

def build_spread_table():
    latest = spread_df.sort_values('report_date').tail(1)
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in latest.columns if i != 'report_date'],
        data=latest.drop(columns=['report_date']).round(2).to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#222','color':'#fff'},
        style_cell={'backgroundColor': '#222','color':'#fff'},
        page_size=1
    )

def build_spread_timeseries():
    fig = go.Figure()
    for col in spread_df.columns:
        if col != 'report_date':
            fig.add_trace(go.Scatter(x=spread_df['report_date'], y=spread_df[col], mode='lines+markers', name=col))
    fig.update_layout(template='plotly_dark', title='Key Curve Spreads (bp)', xaxis_title='Date', yaxis_title='Spread (bp)')
    return dcc.Graph(figure=fig)

print("[Startup] Building spread_tab ...")
spread_tab = dbc.Container([
    html.H4("Key Curve Spreads (bp)"),
    build_spread_table(),
    build_spread_timeseries()
], fluid=True)
print("[Startup] spread_tab built.")

# ---- Enhanced Curve Evolution Tab ----
def build_curve_evolution_fig_enhanced(method: str = 'Cubic Spline') -> go.Figure:
    y_min = df_bonds['bond_yield'].min() - 0.5
    y_max = df_bonds['bond_yield'].max() + 0.5
    frames = []

print("[Startup] Building app.layout ... (only Yield Curve tab enabled for debug)")
app.layout = dbc.Container([
    html.H1("South African Fixed Income Dashboard", className='mt-4'),
    dbc.Tabs([
        dbc.Tab(label='Yield Curve', tab_id='tab-curve', children=[yield_curve_tab]),
        dbc.Tab(label='Bond Analytics', tab_id='tab-bond', children=[bond_analytics_tab]),
        dbc.Tab(label='Curve Evolution', tab_id='tab-evolution', children=[curve_evolution_tab]),
        dbc.Tab(label='Bond Summary', tab_id='tab-summary', children=[bond_summary_tab]),
        dbc.Tab(label='Curve Spreads/Shape', tab_id='tab-spreads', children=[spread_tab]),
        dbc.Tab(label='Curve Shape Analytics', tab_id='tab-shape', children=[curve_shape_analytics_tab]),
    ])
], fluid=True)
print("[Startup] app.layout built.")

# ----------------------------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------------------------

@app.callback(
    Output('yield-curve-graph', 'figure'),
    Input('date-picker', 'date'),
    Input('interp-method', 'value')
)
def update_yield_curve(date, method):
    date = pd.to_datetime(date)
    return build_yield_curve_fig(date, method)


@app.callback(
    Output('bond-history-graph', 'figure'),
    Output('latest-yield', 'children'),
    Output('latest-duration', 'children'),
    Output('latest-dv01', 'children'),
    Input('bond-dropdown', 'value')
)
def update_bond_analytics(bond):
    fig = build_bond_history_fig(bond)
    latest = df_bonds[df_bonds['bond'] == bond].sort_values('report_date').iloc[-1]
    return (
        fig,
        f"{latest['bond_yield']:.2f}",
        f"{latest['modified_duration']:.2f}",
        f"{latest['dv01']:.4f}"
    )


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    print("[Startup] Starting Dash app...")
    app_start = time.time()
    try:
        app.run_server(debug=True, host='0.0.0.0', port=8050)
    except Exception as e:
        print(f"[Startup][ERROR] Dash app failed to start: {e}")
        raise
    print(f"[Startup] Dash app exited after {time.time() - app_start:.2f} seconds")