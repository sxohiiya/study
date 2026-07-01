import streamlit as st
import pandas as pd
import joblib

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="KOSPI AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# DATA LOAD
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/kospi_5years_combined.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.zfill(6)
    )

    # Delete weekend data (Saturday = 5, Sunday = 6)
    df = df[df["Date"].dt.dayofweek < 5]

    return df

# =====================================================
# MODEL LOAD
# =====================================================

@st.cache_resource
def load_models():

    return {

        "Linear Regression":
        joblib.load(
            "models/lr_model.pkl"
        ),

        "Random Forest":
        joblib.load(
            "models/rf_model.pkl"
        ),

        "XGBoost":
        joblib.load(
            "models/xgb_model.pkl"
        ),

        "LightGBM":
        joblib.load(
            "models/lgbm_model.pkl"
        )
    }

# =====================================================
# LOAD
# =====================================================

df = load_data()

models = load_models()

# =====================================================
# STOCK LIST
# =====================================================

ticker_map = {

    "005930":"삼성전자",
    "000660":"SK하이닉스",
    "402340":"SK스퀘어",
    "005935":"삼성전자우",
    "009150":"삼성전기",
    "005380":"현대차",
    "373220":"LG에너지솔루션",
    "032830":"삼성생명",
    "028260":"삼성물산",
    "329180":"HD현대중공업"

}

# =====================================================
# DEFAULT
# =====================================================

if "symbol" not in st.session_state:

    st.session_state.symbol = "005930"

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📊 종목")

    for code, name in ticker_map.items():

        if st.button(
            name,
            use_container_width=True
        ):
            st.session_state.symbol = code

# =====================================================
# SELECTED STOCK
# =====================================================

selected_symbol = (
    st.session_state.symbol
)

stock_df = (

    df[
        df["Symbol"]
        ==
        selected_symbol
    ]

    .copy()

    .sort_values("Date")

)

# =====================================================
# STOCK INFO
# =====================================================

stock_name = (
    ticker_map.get(
        selected_symbol,
        selected_symbol
    )
)

latest_row = (
    stock_df.iloc[-1]
)

current_close = (
    latest_row["Close"]
)

# =====================================================
# FEATURE
# =====================================================

features = [

    "Open",
    "High",
    "Low",
    "Close",
    "Volume"

]

X_latest = (
    stock_df[features]
    .tail(1)
)

# =====================================================
# PREDICTION
# =====================================================

predictions = {}

for model_name, model in models.items():

    pred = (
        model.predict(
            X_latest
        )[0]
    )

    predictions[
        model_name
    ] = pred

# =====================================================
# TITLE
# =====================================================

st.title(
    f"📈 {stock_name}"
)

st.caption(
    f"종목코드 : {selected_symbol}"
)

# =====================================================
# CHART CONTROL
# =====================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col1:

    timeframe = st.radio(
        "봉",
        ["일봉", "주봉", "월봉"],
        horizontal=True
    )

with col2:

    period = st.radio(
        "기간",
        ["3개월", "6개월", "1년", "3년", "전체"],
        horizontal=True,
        index=2
    )

with col3:

    show_volume = st.toggle(
        "거래량",
        value=False
    )

# =====================================================
# RESAMPLE
# =====================================================

if timeframe == "주봉":

    stock_df = (
        stock_df
        .set_index("Date")
        .resample("W-FRI")  # Group by week, ending on Friday to avoid weekend dates
        .agg({
            "Open":"first",
            "High":"max",
            "Low":"min",
            "Close":"last",
            "Volume":"sum"
        })
        .dropna()
        .reset_index()
    )

elif timeframe == "월봉":

    stock_df = (
        stock_df
        .set_index("Date")
        .resample("ME")
        .agg({
            "Open":"first",
            "High":"max",
            "Low":"min",
            "Close":"last",
            "Volume":"sum"
        })
        .dropna()
        .reset_index()
    )
    # If the month ends on a weekend, shift it to Friday
    weekday = stock_df["Date"].dt.dayofweek
    stock_df.loc[weekday == 5, "Date"] -= pd.Timedelta(days=1)
    stock_df.loc[weekday == 6, "Date"] -= pd.Timedelta(days=2)

# =====================================================
# PERIOD FILTER
# =====================================================

latest_date = stock_df["Date"].max()

if period == "3개월":

    stock_df = stock_df[
        stock_df["Date"]
        >= latest_date - pd.DateOffset(months=3)
    ]

elif period == "6개월":

    stock_df = stock_df[
        stock_df["Date"]
        >= latest_date - pd.DateOffset(months=6)
    ]

elif period == "1년":

    stock_df = stock_df[
        stock_df["Date"]
        >= latest_date - pd.DateOffset(years=1)
    ]

elif period == "3년":

    stock_df = stock_df[
        stock_df["Date"]
        >= latest_date - pd.DateOffset(years=3)
    ]

if stock_df.empty:

    st.error(
        "종목 데이터 없음"
    )

    st.stop()

# =====================================================
# FUTURE SPACE
# =====================================================

last_date = (
    stock_df["Date"]
    .max()
)

# Set dynamic spacing based on timeframe to make predictions visually spaced out ("좀 멀리")
if timeframe == "주봉":
    future_date = last_date + pd.Timedelta(weeks=4)
    chart_end = last_date + pd.Timedelta(weeks=6)
    default_visible_count = min(35, len(stock_df))
elif timeframe == "월봉":
    future_date = last_date + pd.DateOffset(months=4)
    chart_end = last_date + pd.DateOffset(months=6)
    default_visible_count = min(24, len(stock_df))
else:  # 일봉
    future_date = last_date + pd.Timedelta(days=15)
    chart_end = last_date + pd.Timedelta(days=25)
    default_visible_count = min(60, len(stock_df))

# Shift future dates if they land on a weekend to avoid hidden gaps
def to_weekday(dt):
    if dt.dayofweek == 5:  # Saturday
        return dt + pd.Timedelta(days=2)
    elif dt.dayofweek == 6:  # Sunday
        return dt + pd.Timedelta(days=1)
    return dt

future_date = to_weekday(future_date)
chart_end = to_weekday(chart_end)

# Compute starting date for the initial view range so candles are not squashed
if default_visible_count > 0:
    chart_start = stock_df["Date"].iloc[-default_visible_count]
else:
    chart_start = stock_df["Date"].min()

# =====================================================
# CHART
# =====================================================

if show_volume:

    fig = make_subplots(

        rows=2,
        cols=1,

        shared_xaxes=True,

        row_heights=[
            0.8,
            0.2
        ],

        vertical_spacing=0.03
    )

else:

    fig = make_subplots(
        rows=1,
        cols=1
    )

# =====================================================
# CANDLE
# =====================================================

fig.add_trace(

    go.Candlestick(

        x=stock_df["Date"],

        open=stock_df["Open"],

        high=stock_df["High"],

        low=stock_df["Low"],

        close=stock_df["Close"],

        name=stock_name,

        # Standard Korean colors: Red for up, Blue for down
        increasing_line_color='#FF3B30',
        decreasing_line_color='#007AFF'

    ),

    row=1,
    col=1

)

# =====================================================
# PREDICTION LINE
# =====================================================

label_positions = [
    "top right",
    "top left",
    "bottom right",
    "bottom left"
]

for i, (model_name, pred) in enumerate(predictions.items()):

    # Slightly offset the end point so labels do not overlap when several
    # models produce the same predicted value.
    offset_days = i * pd.Timedelta(days=0.4)
    offset_x = future_date + offset_days

    fig.add_trace(

        go.Scatter(

            x=[
                last_date,
                offset_x
            ],

            y=[
                current_close,
                pred
            ],

            mode="lines+markers+text",

            marker=dict(
                size=10
            ),

            line=dict(
                width=3,
                dash="dash"
            ),

            text=[
                "",
                f"{model_name} ({pred:,.0f})"
            ],

            textposition=label_positions[i % len(label_positions)],

            name=model_name

        ),

        row=1,
        col=1

    )

# =====================================================
# VOLUME
# =====================================================

if show_volume:

    fig.add_trace(

        go.Bar(

            x=stock_df["Date"],

            y=stock_df["Volume"],

            name="Volume"

        ),

        row=2,
        col=1

    )

# =====================================================
# LAYOUT
# =====================================================

fig.update_xaxes(

    range=[
        chart_start,
        chart_end
    ],
    # Remove weekend gaps from the chart visualization
    rangebreaks=[
        dict(bounds=["sat", "mon"])
    ]

)

fig.update_layout(

    template="plotly_dark",

    # Reduced height from 1500 to 700 for a much more comfortable scrolling experience
    height=700,

    hovermode="x unified",

    dragmode="zoom",

    xaxis_rangeslider_visible=False,

    legend=dict(
        orientation="h",
        y=1.02,
        x=0
    ),

    margin=dict(
        l=5,
        r=5,
        t=20,
        b=5
    )
)


# =====================================================
# SHOW
# =====================================================

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# RESULT TABLE
# =====================================================

result_df = pd.DataFrame({

    "Model":
    list(
        predictions.keys()
    ),

    "Predicted Close":
    [
        round(v, 2)
        for v
        in predictions.values()
    ]

})


st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)