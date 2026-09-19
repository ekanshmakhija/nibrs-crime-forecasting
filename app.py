import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================================
# NIBRS CRIME INTELLIGENCE DASHBOARD
# Power BI-style research dashboard v3
# ============================================================

st.set_page_config(
    page_title="NIBRS Crime Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent / "data"

DATA_FILE = BASE / "forecasting_dataset.csv"
FORECAST_FILE = BASE / "final_forecasts_jan_mar_2025.csv"
MODEL_FILE = BASE / "final_sarima_selected_models.csv"
STATE_PERF_FILE = BASE / "model_comparison_by_state.csv"
HORIZON_FILE = BASE / "model_comparison_by_horizon.csv"
ORIGIN_FILE = BASE / "model_comparison_by_origin.csv"

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: #0b0f17;
    }

    [data-testid="stSidebar"] {
        background: #171b25;
        border-right: 1px solid #292f3d;
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 285px;
    }

    section[data-testid="stSidebar"] {
        width: 285px !important;
    }

    .chart-card {
        background: linear-gradient(145deg, #121824, #0f141d);
        border: 1px solid #252f40;
        border-radius: 14px;
        padding: 4px;
    }

    .insight-box {
        background: linear-gradient(145deg, #111c2b, #0e1723);
        border: 1px solid #294866;
        border-radius: 14px;
        padding: 16px 18px;
        color: #b9c9dd;
        margin-top: 10px;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .main-title {
        font-size: 2.55rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #9ca6b7;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 0.7rem;
        margin-bottom: 0.8rem;
    }

    .eyebrow {
        color: #8f9aab;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 700;
    }

    .metric-card {
        background: linear-gradient(145deg, #151b26, #111620);
        border: 1px solid #293241;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 112px;
    }

    .metric-label {
        color: #8f9aab;
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 780;
        color: #f3f6fb;
    }

    .metric-sub {
        color: #7f8a9d;
        font-size: 0.75rem;
        margin-top: 5px;
    }

    .forecast-card {
        background: linear-gradient(145deg, #151d2a, #111722);
        border: 1px solid #2b3b51;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .forecast-month {
        color: #9ca6b7;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .forecast-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .forecast-range {
        color: #7f8a9d;
        font-size: 0.74rem;
        margin-top: 5px;
    }

    .info-box {
        background: #132235;
        border: 1px solid #274463;
        border-radius: 12px;
        padding: 15px 18px;
        color: #b8c8dc;
        font-size: 0.9rem;
    }

    .warning-box {
        background: #292513;
        border: 1px solid #5c4d20;
        border-radius: 12px;
        padding: 14px 18px;
        color: #d8c98f;
        font-size: 0.86rem;
    }

    .insight-box {
        background: linear-gradient(145deg, #17263a, #111a28);
        border: 1px solid #36577c;
        border-radius: 14px;
        padding: 16px 18px;
        color: #d9e7f7;
        font-size: 0.9rem;
        margin: 8px 0 16px 0;
    }

    div[data-testid="stMetric"] {
        background: #151b26;
        border: 1px solid #293241;
        padding: 14px 16px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8f9aab;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #292f3d;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 18px;
        color: #aab3c2;
    }

    .stTabs [aria-selected="true"] {
        color: #f1f5fb;
    }

    .footer {
        color: #667184;
        font-size: 0.75rem;
        padding-top: 25px;
        padding-bottom: 10px;
        border-top: 1px solid #242b37;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    fc = pd.read_csv(FORECAST_FILE, parse_dates=["forecast_date", "training_end"])
    models = pd.read_csv(MODEL_FILE)

    state_perf = pd.read_csv(STATE_PERF_FILE)
    horizon_perf = pd.read_csv(HORIZON_FILE)
    origin_perf = pd.read_csv(ORIGIN_FILE, parse_dates=["origin"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")

    return df, fc, models, state_perf, horizon_perf, origin_perf


# -----------------------------
# Helpers
# -----------------------------
def fmt_num(x):
    if pd.isna(x):
        return "—"
    return f"{int(round(x)):,}"


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def forecast_card(month, value, lower, upper):
    st.markdown(
        f"""
        <div class="forecast-card">
            <div class="forecast-month">{month}</div>
            <div class="forecast-value">{fmt_num(value)}</div>
            <div class="forecast-range">95% interval: {fmt_num(lower)} – {fmt_num(upper)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def base_layout(fig, height=390):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0b0f17",
        plot_bgcolor="#0b0f17",
        font=dict(color="#dce3ed"),
        margin=dict(l=55, r=25, t=20, b=50),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="#242b37",
            zerolinecolor="#242b37",
        ),
        yaxis=dict(
            gridcolor="#242b37",
            zerolinecolor="#242b37",
        ),
    )
    return fig


# -----------------------------
# Load
# -----------------------------
try:
    df, forecasts, models, state_perf, horizon_perf, origin_perf = load_data()
except Exception as e:
    st.error("Dashboard data could not be loaded.")
    st.code(str(e))
    st.info(
        "Make sure the required CSV files are present in the repository data/ folder and that "
        "the application is being run from the project directory."
    )
    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## 🎛️ Filters")
st.sidebar.caption("Explore historical NIBRS observations")

jurisdictions = sorted(df["state_abb"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Jurisdiction", jurisdictions, index=jurisdictions.index("AL") if "AL" in jurisdictions else 0)

years = sorted(df["year"].unique())
year_options = ["All Years"] + [str(y) for y in years]
selected_year = st.sidebar.selectbox("Historical Year", year_options)

month_options = ["All Months"] + list(range(1, 13))
month_labels = {i: pd.Timestamp(2000, i, 1).strftime("%B") for i in range(1, 13)}
selected_month = st.sidebar.selectbox(
    "Historical Month",
    ["All Months"] + [month_labels[i] for i in range(1, 13)]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.markdown("**Source:** FBI NIBRS")
st.sidebar.markdown("**Period:** 2015–2024")
st.sidebar.markdown(f"**Jurisdictions:** {df['state_abb'].nunique()}")
st.sidebar.markdown("**Forecast horizon:** 3 months")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Counts represent recorded NIBRS incidents. Changes over time may partly "
    "reflect agency participation and reporting coverage."
)

# -----------------------------
# Filter historical data
# -----------------------------
state_df = df[df["state_abb"] == selected_state].copy()

filtered = state_df.copy()

if selected_year != "All Years":
    filtered = filtered[filtered["year"] == int(selected_year)]

if selected_month != "All Months":
    month_num = list(month_labels.keys())[list(month_labels.values()).index(selected_month)]
    filtered = filtered[filtered["month"] == month_num]


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">📊 NIBRS Crime Forecasting</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Jurisdiction-level monthly crime incident analysis and three-month SARIMA forecasting using FBI NIBRS data.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="insight-box"><b>LIVE ANALYTICS VIEW</b> &nbsp; • &nbsp; Historical NIBRS intelligence + 3-month SARIMA forecast &nbsp; • &nbsp; 41 eligible jurisdictions</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_history, tab_forecast, tab_performance = st.tabs(
    ["📌 Overview", "📈 Historical Analysis", "🔮 Forecast", "📊 Model Performance"]
)


# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:

    st.markdown(
        f'<div class="section-title">Overview — {selected_state}</div>',
        unsafe_allow_html=True,
    )

    total_incidents = filtered["incidents"].sum()
    avg_month = filtered["incidents"].mean() if len(filtered) else np.nan

    y2024 = state_df[state_df["year"] == 2024]["incidents"].sum()
    dec2024 = state_df[
        (state_df["year"] == 2024) & (state_df["month"] == 12)
    ]["incidents"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("RECORDED INCIDENTS", fmt_num(total_incidents), "Current filter")
    with c2:
        metric_card("MONTHLY AVERAGE", fmt_num(avg_month), "Across selected period")
    with c3:
        metric_card("2024 INCIDENTS", fmt_num(y2024), "Full year")
    with c4:
        metric_card("DECEMBER 2024", fmt_num(dec2024), "Latest observed month")

    st.markdown("")
    st.markdown("### Recent Incident Trend")

    recent = state_df[state_df["date"] >= "2022-01-01"].sort_values("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["incidents"],
            mode="lines",
            name="Recorded incidents",
            line=dict(width=2.8),
        )
    )
    fig.update_layout(title="Monthly recorded incidents — 2022 to 2024", yaxis_title="Recorded incidents", xaxis_title="")
    base_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    left, right = st.columns(2)

    with left:
        st.markdown("### Yearly Incident Volume")
        yearly = state_df.groupby("year", as_index=False)["incidents"].sum().sort_values("year")
        fig_year = px.bar(yearly, x="year", y="incidents", labels={"year": "Year", "incidents": "Recorded incidents"})
        base_layout(fig_year, 340)
        st.plotly_chart(fig_year, use_container_width=True, config={"displaylogo": False})

    with right:
        st.markdown("### Seasonal Pattern")
        season = state_df.groupby("month", as_index=False)["incidents"].mean().sort_values("month")
        season["month_name"] = season["month"].map(month_labels)
        fig_season = px.bar(season, x="month_name", y="incidents", labels={"month_name": "", "incidents": "Average recorded incidents"})
        base_layout(fig_season, 340)
        fig_season.update_xaxes(categoryorder="array", categoryarray=list(month_labels.values()))
        st.plotly_chart(fig_season, use_container_width=True, config={"displaylogo": False})

    st.markdown("### Visual Historical Breakdown")
    show_insights = st.checkbox("Show visual historical database", value=False)

    if show_insights:
        # Power BI-style compact breakdowns based only on the cleaned jurisdiction-month dataset.
        qdf = state_df.copy()
        qdf["quarter"] = qdf["month"].map({1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"})

        q1, q2 = st.columns(2)
        with q1:
            st.markdown("#### Quarterly distribution — 2024")
            q24 = qdf[qdf["year"] == 2024].groupby("quarter", as_index=False)["incidents"].sum()
            fig_q = go.Figure(data=[go.Pie(labels=q24["quarter"], values=q24["incidents"], hole=0.58, textinfo="label+percent", hovertemplate="%{label}: %{value:,}<extra></extra>")])
            fig_q.update_layout(title="Share of recorded incidents", height=340, paper_bgcolor="#0b0f17", plot_bgcolor="#0b0f17", font=dict(color="#dce3ed"), margin=dict(l=20,r=20,t=55,b=20), showlegend=True)
            st.plotly_chart(fig_q, use_container_width=True, config={"displaylogo": False})

        with q2:
            st.markdown("#### Recent-year comparison")
            recent_years = state_df[state_df["year"].isin([2022, 2023, 2024])].groupby("year", as_index=False)["incidents"].sum()
            fig_ry = go.Figure()
            fig_ry.add_trace(go.Bar(x=recent_years["year"].astype(str), y=recent_years["incidents"], name="Recorded incidents"))
            fig_ry.update_layout(title="2022–2024 annual volume", yaxis_title="Recorded incidents", xaxis_title="Year")
            base_layout(fig_ry, 340)
            st.plotly_chart(fig_ry, use_container_width=True, config={"displaylogo": False})

        h1, h2 = st.columns(2)
        with h1:
            st.markdown("#### Monthly profile")
            monthly_profile = state_df.groupby("month", as_index=False)["incidents"].mean()
            monthly_profile["month_name"] = monthly_profile["month"].map(month_labels)
            fig_mp = go.Figure()
            fig_mp.add_trace(go.Bar(x=monthly_profile["month_name"], y=monthly_profile["incidents"], name="Average incidents"))
            fig_mp.update_layout(title="Average incidents by calendar month", yaxis_title="Average recorded incidents", xaxis_title="Month")
            base_layout(fig_mp, 340)
            fig_mp.update_xaxes(categoryorder="array", categoryarray=list(month_labels.values()))
            st.plotly_chart(fig_mp, use_container_width=True, config={"displaylogo": False})

        with h2:
            st.markdown("#### Recent 12-month movement")
            rolling = state_df.sort_values("date").copy()
            rolling["rolling_12"] = rolling["incidents"].rolling(12, min_periods=6).mean()
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=rolling["date"], y=rolling["incidents"], mode="lines", name="Monthly", line=dict(width=1.4)))
            fig_roll.add_trace(go.Scatter(x=rolling["date"], y=rolling["rolling_12"], mode="lines", name="12-month average", line=dict(width=3)))
            fig_roll.update_layout(title="Monthly incidents and rolling average", yaxis_title="Recorded incidents", xaxis_title="Date")
            base_layout(fig_roll, 340)
            st.plotly_chart(fig_roll, use_container_width=True, config={"displaylogo": False})

        csv_data = state_df[["date", "year", "month", "incidents"]].to_csv(index=False).encode("utf-8")
        st.download_button(
               "⬇ Download historical data",
                csv_data,
                file_name=f"{selected_state}_historical_nibrs.csv",
                mime="text/csv",
                key=f"overview_historical_download_{selected_state}"
        )


    

    st.markdown(
        '<div class="warning-box">⚠ Coverage note: historical NIBRS counts should be interpreted as recorded incidents. Variation may partly reflect changes in participating agencies and reporting coverage.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================
with tab_history:

    st.markdown(
        f'<div class="section-title">Historical Intelligence — {selected_state}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Explore the selected jurisdiction as a visual historical database rather than a raw table.")

    start_date = state_df["date"].min()
    end_date = state_df["date"].max()

    view_mode = st.radio(
        "Historical database view",
        ["📊 Visual Dashboard", "📋 Data Table"],
        horizontal=True,
        index=0,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("FILTERED INCIDENTS", fmt_num(filtered["incidents"].sum()), "Current filters")
    with c2:
        metric_card("MONTHLY AVERAGE", fmt_num(filtered["incidents"].mean()), "Recorded incidents")
    with c3:
        metric_card("PEAK MONTH", fmt_num(filtered["incidents"].max()), "Highest observed month")
    with c4:
        metric_card("OBSERVATIONS", f"{len(filtered):,}", f"{start_date:%b %Y} – {end_date:%b %Y}")

    if view_mode == "📋 Data Table":
        table = filtered[["date", "year", "month", "incidents"]].copy().sort_values("date", ascending=False)
        table["date"] = table["date"].dt.strftime("%B %Y")
        table.columns = ["Date", "Year", "Month", "Recorded Incidents"]
        st.dataframe(table, use_container_width=True, hide_index=True, height=520)
    else:
        hist = filtered.sort_values("date")

        # Main historical trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["incidents"], mode="lines",
            name="Recorded incidents", line=dict(width=2.5)
        ))
        fig.update_layout(
            title="Monthly recorded incidents",
            xaxis_title="Date", yaxis_title="Recorded incidents"
        )
        base_layout(fig, 430)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        # Power-BI style visual cards/charts
        left, right = st.columns(2)
        with left:
            st.markdown("#### Annual incident volume")
            yearly = state_df.groupby("year", as_index=False)["incidents"].sum()
            fig_y = px.bar(yearly, x="year", y="incidents", labels={"year":"Year", "incidents":"Recorded incidents"})
            fig_y.update_layout(title="Year-over-year volume")
            base_layout(fig_y, 330)
            st.plotly_chart(fig_y, use_container_width=True, config={"displaylogo": False})

        with right:
            st.markdown("#### 2024 quarterly distribution")
            q24 = state_df[state_df["year"] == 2024].copy()
            q24["quarter"] = q24["date"].dt.quarter.map(lambda q: f"Q{q}")
            q24 = q24.groupby("quarter", as_index=False)["incidents"].sum()
            fig_q = go.Figure(data=[go.Pie(
                labels=q24["quarter"], values=q24["incidents"], hole=0.58,
                textinfo="label+percent", hovertemplate="%{label}: %{value:,}<extra></extra>"
            )])
            fig_q.update_layout(title="Share of 2024 incidents", height=330,
                                paper_bgcolor="#0b0f17", plot_bgcolor="#0b0f17",
                                font=dict(color="#dce3ed"), margin=dict(l=20,r=20,t=55,b=20))
            st.plotly_chart(fig_q, use_container_width=True, config={"displaylogo": False})

        left, right = st.columns(2)
        with left:
            st.markdown("#### Calendar-month seasonality")
            season = state_df.groupby("month", as_index=False)["incidents"].mean()
            season["month_name"] = season["month"].map(month_labels)
            fig_s = go.Figure(go.Bar(x=season["month_name"], y=season["incidents"], name="Average incidents"))
            fig_s.update_layout(title="Average incidents by month", yaxis_title="Average recorded incidents")
            fig_s.update_xaxes(categoryorder="array", categoryarray=list(month_labels.values()))
            base_layout(fig_s, 330)
            st.plotly_chart(fig_s, use_container_width=True, config={"displaylogo": False})

        with right:
            st.markdown("#### Trend momentum")
            roll = state_df.sort_values("date").copy()
            roll["rolling_12"] = roll["incidents"].rolling(12, min_periods=6).mean()
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=roll["date"], y=roll["incidents"], mode="lines", name="Monthly", line=dict(width=1.3)))
            fig_r.add_trace(go.Scatter(x=roll["date"], y=roll["rolling_12"], mode="lines", name="12-month average", line=dict(width=3)))
            fig_r.update_layout(title="Recorded incidents vs 12-month average", yaxis_title="Incidents")
            base_layout(fig_r, 330)
            st.plotly_chart(fig_r, use_container_width=True, config={"displaylogo": False})

        # Monthly contribution donut for the selected filter, useful when a year/month filter is applied
        st.markdown("#### Distribution by year")
        year_dist = filtered.groupby("year", as_index=False)["incidents"].sum()
        if len(year_dist) >= 2:
            fig_d = go.Figure(data=[go.Pie(
                labels=year_dist["year"].astype(str), values=year_dist["incidents"], hole=0.55,
                textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>"
            )])
            fig_d.update_layout(title="Share of filtered recorded incidents", height=360,
                                paper_bgcolor="#0b0f17", plot_bgcolor="#0b0f17",
                                font=dict(color="#dce3ed"), margin=dict(l=20,r=20,t=55,b=20))
            st.plotly_chart(fig_d, use_container_width=True, config={"displaylogo": False})

        csv_data = filtered[["date", "year", "month", "incidents"]].sort_values("date").to_csv(index=False).encode("utf-8")
        st.download_button(
               "⬇ Download filtered historical data",
                csv_data,
                file_name=f"{selected_state}_historical_nibrs.csv",
                mime="text/csv",
                key=f"history_historical_download_{selected_state}"
        )

                                                                                                                   

    st.markdown(
        '<div class="warning-box">⚠ Coverage note: historical NIBRS counts should be interpreted as recorded incidents. Variation may partly reflect changes in participating agencies and reporting coverage.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST
# ============================================================
with tab_forecast:

    st.markdown(
        f'<div class="section-title">🔮 Three-Month Forecast — {selected_state}</div>',
        unsafe_allow_html=True,
    )

    fc_state = forecasts[forecasts["state_abb"] == selected_state].sort_values("forecast_date").copy()

    if fc_state.empty:
        st.error("No forecast is available for the selected jurisdiction.")
    else:
        model_row = models[models["state_abb"] == selected_state]

        if not model_row.empty:
            order = model_row.iloc[0]["order"]
            seasonal = model_row.iloc[0]["seasonal_order"]
            model_text = f"SARIMA{order} × {seasonal}"
        else:
            model_text = str(fc_state.iloc[0]["model"])

        st.markdown(
            f"""
            <div class="info-box">
            Forecast origin: <b>December 2024</b> &nbsp; | &nbsp;
            Training period: <b>2015–2024</b> &nbsp; | &nbsp;
            Model: <b>{model_text}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        cards = st.columns(3)
        for col, (_, row) in zip(cards, fc_state.iterrows()):
            with col:
                forecast_card(
                    row["forecast_date"].strftime("%B %Y"),
                    row["forecast"],
                    row["lower_95"],
                    row["upper_95"],
                )

        st.markdown("### Actual → Forecast")
        st.caption("The forecast line starts at the December 2024 observed value, so the historical series continues smoothly into the SARIMA forecast.")

        history_plot = state_df[state_df["date"] >= "2022-01-01"].sort_values("date")
        last_actual = history_plot.iloc[-1]

        # Include the final observed point in the forecast trace so the line changes
        # styling at the boundary instead of appearing to break.
        forecast_line = pd.concat([
            pd.DataFrame({"forecast_date": [last_actual["date"]], "forecast": [last_actual["incidents"]]}),
            fc_state[["forecast_date", "forecast"]],
        ], ignore_index=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_plot["date"], y=history_plot["incidents"], mode="lines",
            name="Actual", line=dict(width=2.4, color="#6f8fb3")
        ))
        fig.add_trace(go.Scatter(
            x=forecast_line["forecast_date"], y=forecast_line["forecast"],
            mode="lines+markers", name="SARIMA Forecast",
            line=dict(width=3, color="#4cc9f0"), marker=dict(size=7, color="#4cc9f0")
        ))

        fig.add_trace(go.Scatter(
            x=fc_state["forecast_date"], y=fc_state["upper_95"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=fc_state["forecast_date"], y=fc_state["lower_95"],
            mode="lines", fill="tonexty", fillcolor="rgba(80, 130, 190, 0.16)",
            line=dict(width=0), name="95% Prediction Interval", hoverinfo="skip"
        ))

        fig.add_vline(x=pd.Timestamp("2024-12-01"), line_width=1.5, line_dash="dash")
        fig.add_annotation(x=pd.Timestamp("2024-12-01"), y=1.02, yref="paper", text="Forecast begins", showarrow=False, font=dict(size=11))
        fig.update_layout(title="Historical observations and three-month SARIMA forecast", xaxis_title="Date", yaxis_title="Recorded incidents")
        base_layout(fig, 500)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.markdown("### Forecast Details")
        detail = fc_state[["forecast_date", "forecast", "lower_95", "upper_95"]].copy()
        detail["forecast_date"] = detail["forecast_date"].dt.strftime("%B %Y")
        detail["forecast"] = detail["forecast"].round().astype(int)
        detail["lower_95"] = detail["lower_95"].round().astype(int)
        detail["upper_95"] = detail["upper_95"].round().astype(int)

        dcols = st.columns(3)
        for col, (_, row) in zip(dcols, detail.iterrows()):
            with col:
                metric_card(row["forecast_date"].upper(), fmt_num(row["forecast"]), f"95% interval: {fmt_num(row['lower_95'])} – {fmt_num(row['upper_95'])}")

        st.caption("Forecasts are generated using SARIMA specifications selected during the development period and fitted using observations available through December 2024. Intervals are model-based 95% prediction intervals.")


# ============================================================
# MODEL PERFORMANCE
# ============================================================
with tab_performance:

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">Leakage-controlled rolling-origin validation evaluated SARIMA against a Seasonal Naive benchmark across 41 jurisdictions and 1,107 forecast-month observations from 2022–2024.</div>',
        unsafe_allow_html=True,
    )

    # Overall values from final validation
    sarima_mae = 900.770
    naive_mae = 1499.291
    sarima_rmse = 1578.331
    naive_rmse = 2650.674
    sarima_smape = 5.087
    naive_smape = 8.911

    mae_imp = (naive_mae - sarima_mae) / naive_mae * 100
    rmse_imp = (naive_rmse - sarima_rmse) / naive_rmse * 100
    smape_imp = (naive_smape - sarima_smape) / naive_smape * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("SARIMA MAE", f"{sarima_mae:,.2f}", f"{mae_imp:.2f}% lower than benchmark")
    with c2:
        metric_card("SARIMA RMSE", f"{sarima_rmse:,.2f}", f"{rmse_imp:.2f}% lower than benchmark")
    with c3:
        metric_card("SARIMA sMAPE", f"{sarima_smape:.2f}%", f"{smape_imp:.2f}% lower than benchmark")

    st.markdown("### SARIMA vs Seasonal Naive")

    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "sMAPE"],
        "SARIMA": [sarima_mae, sarima_rmse, sarima_smape],
        "Seasonal Naive": [naive_mae, naive_rmse, naive_smape],
        "Improvement": [mae_imp, rmse_imp, smape_imp],
    })

    comparison["Improvement"] = comparison["Improvement"].map(lambda x: f"{x:.2f}%")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("### Performance by Forecast Horizon")

    hp = horizon_perf.copy()

    fig_h = go.Figure()
    fig_h.add_trace(
        go.Bar(
            x=hp["horizon"].astype(str).map(lambda x: f"Month {x}"),
            y=hp["sarima_MAE"],
            name="SARIMA MAE",
        )
    )
    fig_h.add_trace(
        go.Bar(
            x=hp["horizon"].astype(str).map(lambda x: f"Month {x}"),
            y=hp["naive_MAE"],
            name="Seasonal Naive MAE",
        )
    )
    fig_h.update_layout(
        title="MAE by forecast horizon",
        barmode="group",
        yaxis_title="MAE",
        xaxis_title="Forecast horizon",
    )
    base_layout(fig_h, 360)
    st.plotly_chart(fig_h, use_container_width=True, config={"displaylogo": False})

    hcols = st.columns(3)
    for i, (_, row) in enumerate(hp.iterrows()):
        with hcols[i]:
            metric_card(
                f"MONTH {int(row['horizon'])}",
                f"{row['MAE_improvement_pct']:.2f}%",
                f"SARIMA MAE improvement • win rate {row['sarima_win_rate_pct']:.2f}%",
            )

    st.markdown("### Model Win Distribution")
    win1, win2 = st.columns([1, 2])
    with win1:
        wins = pd.DataFrame({"model": ["SARIMA", "Seasonal Naive"], "wins": [36, 5]})
        fig_w = go.Figure(data=[go.Pie(labels=wins["model"], values=wins["wins"], hole=0.62, textinfo="label+percent", hovertemplate="%{label}: %{value} jurisdictions<extra></extra>")])
        fig_w.update_layout(title="Jurisdiction wins", height=320, paper_bgcolor="#0b0f17", plot_bgcolor="#0b0f17", font=dict(color="#dce3ed"), margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig_w, use_container_width=True, config={"displaylogo": False})
    with win2:
        st.markdown("#### Where SARIMA helps most")
        best = state_perf.sort_values("MAE_improvement_pct", ascending=False).head(10).sort_values("MAE_improvement_pct")
        fig_best = go.Figure()
        fig_best.add_trace(go.Bar(y=best["state_abb"], x=best["MAE_improvement_pct"], orientation="h", name="MAE improvement"))
        fig_best.update_layout(title="Top 10 jurisdictions by MAE improvement", xaxis_title="MAE improvement (%)", yaxis_title="Jurisdiction")
        base_layout(fig_best, 320)
        st.plotly_chart(fig_best, use_container_width=True, config={"displaylogo": False})

    st.markdown("### Where SARIMA struggles")
    worst = state_perf.sort_values("MAE_improvement_pct").head(10).sort_values("MAE_improvement_pct", ascending=False)
    fig_worst = go.Figure()
    fig_worst.add_trace(go.Bar(y=worst["state_abb"], x=worst["MAE_improvement_pct"], orientation="h", name="MAE improvement"))
    fig_worst.update_layout(title="Bottom 10 jurisdictions by MAE improvement", xaxis_title="MAE improvement (%)", yaxis_title="Jurisdiction")
    base_layout(fig_worst, 360)
    st.plotly_chart(fig_worst, use_container_width=True, config={"displaylogo": False})

    st.markdown(
        '<div class="insight-box"><b>Key finding:</b> SARIMA produced lower errors than the Seasonal Naive benchmark in 36 of 41 jurisdictions under the leakage-controlled 2022–2024 rolling-origin evaluation.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box"><b>Statistical significance:</b> Wilcoxon signed-rank test on paired absolute forecast errors yielded p = 2.61 × 10⁻³⁵, indicating significantly lower SARIMA errors under the study evaluation design.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="warning-box">Model performance is specific to the NIBRS dataset and evaluation design used in this study and should not be interpreted as universal superiority of SARIMA.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">NIBRS Crime Forecasting • FBI NIBRS 2015–2024 • 41 eligible jurisdictions • Three-month SARIMA forecasting framework</div>',
    unsafe_allow_html=True,
)
