import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import kruskal

st.set_page_config(page_title="ClimateScope Global Weather Intelligence System", layout="wide")

st.title("🌍 ClimateScope Global Weather Intelligence System")

with st.expander("📘 How to use this dashboard", expanded=False):
    st.markdown(
        """
        - **Title header**: Shows the ClimateScope Global Weather Intelligence System.
        - **Sidebar**: Contains filters, sliders, and feature menu tabs.
        - **Dynamic analysis panels**: Display interactive charts and maps.
        - **Plotly visualizations**: Support zoom, hover, and tooltips.
        - **Multiple chart types**: Maps, heatmaps, histograms, line graphs, scatter plots, and bar charts.
        - **Real-time updates**: Graphs respond instantly to filter changes.
        - **Dividers**: Separate each analysis module for clarity.
        - **Layout**: Uses Streamlit wide mode for a clean, responsive experience.
        """
    )

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("processed_weather_data.csv")
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df["year"] = df["last_updated"].dt.year
    df["month"] = df["last_updated"].dt.month
    return df

df = load_data()

temp_col = "temperature_celsius"

# -------------------------------------------------------
# FEATURE FUNCTIONS
# -------------------------------------------------------

def show_key_metrics(df):
    st.subheader("📊 Key Temperature Metrics")
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Temp", f"{df[temp_col].mean():.1f}°C")
        col2.metric("Max Temp", f"{df[temp_col].max():.1f}°C")
        col3.metric("Min Temp", f"{df[temp_col].min():.1f}°C")
        col4.metric("Std Dev", f"{df[temp_col].std():.1f}°C")
    else:
        st.write("No data available.")

def hottest_countries(df):
    st.subheader("🔥 Hottest Countries")
    if not df.empty:
        top = df.groupby("country")[temp_col].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x":"Avg Temp","y":"Country"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def show_distribution(df):
    st.subheader("📊 Temperature Distribution")
    if not df.empty:
        fig = px.histogram(df, x=temp_col, nbins=40)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def seasonal_heatmap(df):
    st.subheader("📅 Seasonal Temperature Heatmap")
    if not df.empty:
        season = df.groupby(["country","month"])[temp_col].mean().reset_index()
        fig = px.density_heatmap(season, x="month", y="country", z=temp_col, color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def long_term_trend(df):
    st.subheader("📈 Long-term Temperature Trend")
    if not df.empty:
        trend = df.groupby("last_updated")[temp_col].mean().reset_index()
        fig = px.line(trend, x="last_updated", y=temp_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def rolling_avg(df):
    st.subheader("📊 Rolling Average Trend")
    if not df.empty:
        window = st.slider("Rolling Window (days)", 3, 30, 7, key="rolling")
        trend = df.groupby("last_updated")[temp_col].mean().reset_index()
        trend["rolling"] = trend[temp_col].rolling(window).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend[temp_col], name="Original"))
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["rolling"], name="Rolling Average"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def extreme_events(df):
    st.subheader("⚠ Extreme Weather Events")
    if not df.empty:
        threshold = df[temp_col].quantile(0.95)
        extreme = df[df[temp_col] > threshold]
        if not extreme.empty:
            fig = px.scatter(extreme, x="last_updated", y=temp_col, color="country")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No extreme events detected.")
    else:
        st.write("No data available.")

def similarity(df):
    st.subheader("🌍 Climate Similarity Matrix")
    if not df.empty:
        features = [temp_col]
        if "humidity" in df.columns:
            features.append("humidity")
        if "precip_mm" in df.columns:
            features.append("precip_mm")
        cluster = df.groupby("country")[features].mean()
        if len(cluster) > 1:
            sim = cosine_similarity(cluster)
            fig = px.imshow(sim, x=cluster.index, y=cluster.index, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Need more countries for similarity matrix.")
    else:
        st.write("No data available.")

def kruskal_test(df):
    st.subheader("📈 Kruskal-Wallis Test")
    if not df.empty and len(df["country"].unique()) > 1:
        groups = [df[df["country"] == c][temp_col].dropna() for c in df["country"].unique()]
        if all(len(g) > 0 for g in groups):
            stat, p = kruskal(*groups)
            st.write(f"Statistic: {stat:.2f}, p-value: {p:.4f}")
            if p < 0.05:
                st.write("Significant differences between countries.")
            else:
                st.write("No significant differences.")
        else:
            st.write("Insufficient data for test.")
    else:
        st.write("Need multiple countries for Kruskal-Wallis test.")

def multi_param(df):
    st.subheader("🌡️ Multi-parameter Analysis")
    if not df.empty:
        params = [temp_col]
        if "humidity" in df.columns:
            params.append("humidity")
        if "precip_mm" in df.columns:
            params.append("precip_mm")
        if "wind_kph" in df.columns:
            params.append("wind_kph")
        if len(params) > 1:
            fig = px.scatter_matrix(df[params])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Need more parameters.")
    else:
        st.write("No data available.")

def yoy(df):
    st.subheader("📅 Year-over-Year Comparison")
    if not df.empty:
        yoy_data = df.groupby("year")[temp_col].mean().reset_index()
        yoy_data["yoy_change"] = yoy_data[temp_col].pct_change() * 100
        fig = px.bar(yoy_data, x="year", y="yoy_change", labels={"yoy_change":"% Change"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def humidity_wind(df):
    st.subheader("💨 Humidity & Wind Trends")
    if not df.empty and "humidity" in df.columns and "wind_kph" in df.columns:
        trend = df.groupby("last_updated")[["humidity", "wind_kph"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["humidity"], name="Humidity"))
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["wind_kph"], name="Wind Speed"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Humidity or wind data not available.")

def dbscan_anomaly(df):
    st.subheader("🔍 DBSCAN Anomaly Detection")
    if not df.empty:
        features = [temp_col]
        if "humidity" in df.columns:
            features.append("humidity")
        data = df[features].dropna()
        if len(data) > 10:
            scaler = StandardScaler()
            X = scaler.fit_transform(data)
            db = DBSCAN(eps=0.5, min_samples=5)
            labels = db.fit_predict(X)
            df_anom = data.copy()
            df_anom["anomaly"] = labels == -1
            fig = px.scatter(df_anom, x=features[0], y=features[1] if len(features)>1 else features[0], color="anomaly")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for anomaly detection.")
    else:
        st.write("No data available.")

def monthly_dist(df):
    st.subheader("📅 Monthly Temperature Distribution")
    if not df.empty:
        fig = px.box(df, x="month", y=temp_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def percentile(df):
    st.subheader("📊 Temperature Percentiles")
    if not df.empty:
        perc = df[temp_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        st.write(perc)
        fig = px.line(x=perc.index, y=perc.values, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def conditions(df):
    st.subheader("🌤️ Weather Conditions Distribution")
    if not df.empty and "condition" in df.columns:
        cond = df["condition"].value_counts()
        fig = px.pie(values=cond.values, names=cond.index)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Weather condition data not available.")

def forecast(df):
    st.subheader("🔮 Temperature Forecast")
    if not df.empty:
        year_temp = df.groupby("year")[temp_col].mean().reset_index()
        if len(year_temp) > 1:
            X = year_temp[["year"]]
            y = year_temp[temp_col]
            model = LinearRegression()
            model.fit(X, y)
            future = pd.DataFrame({"year": np.arange(year_temp["year"].max()+1, year_temp["year"].max()+6)})
            future["prediction"] = model.predict(future)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=year_temp["year"], y=year_temp[temp_col], mode="lines+markers", name="Historical"))
            fig.add_trace(go.Scatter(x=future["year"], y=future["prediction"], mode="lines+markers", name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for forecasting.")
    else:
        st.write("No data available.")

def corr_vars(df):
    st.subheader("🔗 Correlation Analysis")
    if not df.empty:
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            corr = numeric.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No numeric data.")
    else:
        st.write("No data available.")

def precipitation_rainfall(df):
    st.subheader("🌧 Precipitation Analysis")
    if not df.empty and "precip_mm" in df.columns:
        rain = df.groupby("country")["precip_mm"].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=rain.values, y=rain.index, orientation="h", labels={"x":"Avg Precip","y":"Country"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Precipitation data not available.")

def hottest_on_globe(df):
    # Already covered in hottest_countries
    pass

def latitude_longitude_map(df):
    st.subheader("🌍 Lat-Long Temperature Map")
    if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
        fig = px.scatter_geo(df, lat="latitude", lon="longitude", color=temp_col, hover_name="country", color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Location data not available.")

def top_countries_comparison(df):
    st.subheader("🏆 Top Countries Comparison")
    if not df.empty:
        top_countries = df.groupby("country")[temp_col].mean().sort_values(ascending=False).head(10).index
        comp = df[df["country"].isin(top_countries)].groupby("country")[temp_col].agg(["mean", "max", "min"]).reset_index()
        st.dataframe(comp)
        fig = px.bar(comp, x="country", y="mean", title="Average Temperature")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def global_insight_summary_panel(df):
    st.subheader("🌐 Global Insights")
    if not df.empty:
        st.write(f"Total Countries: {df['country'].nunique()}")
        st.write(f"Average Global Temp: {df[temp_col].mean():.1f}°C")
        st.write(f"Data Points: {len(df)}")
    else:
        st.write("No data available.")

def interactive_feature_explanations(df):
    st.subheader("ℹ️ Feature Explanations")
    st.write("This dashboard provides various analyses of global weather data.")
    st.write("- **Key Metrics**: Basic statistics of temperature.")
    st.write("- **Maps**: Visualize temperature geographically.")
    # Add more explanations as needed

def time_series_decomposition(df):
    st.subheader("📉 Time Series Decomposition")
    if not df.empty:
        ts = df.groupby("last_updated")[temp_col].mean()
        ts = ts.asfreq("D").ffill()
        if len(ts) > 60:
            result = seasonal_decompose(ts, model="additive", period=30)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=result.trend, name="Trend"))
            fig.add_trace(go.Scatter(x=ts.index, y=result.seasonal, name="Seasonal"))
            fig.add_trace(go.Scatter(x=ts.index, y=result.resid, name="Residual"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data.")
    else:
        st.write("No data available.")

def heatwave_detection(df):
    st.subheader("🔥 Heatwave Detection")
    if not df.empty:
        heat_threshold = df[temp_col].quantile(0.90)
        heatwaves = df[df[temp_col] > heat_threshold]
        st.metric("Heatwave Events", len(heatwaves))
        if not heatwaves.empty:
            fig = px.scatter(heatwaves, x="last_updated", y=temp_col, color="country")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No heatwaves detected.")
    else:
        st.write("No data available.")

def seasonal_rainfall_comparison(df):
    st.subheader("🌧 Seasonal Rainfall Comparison")
    if not df.empty and "precip_mm" in df.columns:
        rain = df.groupby(["country", "month"])["precip_mm"].mean().reset_index()
        fig = px.line(rain, x="month", y="precip_mm", color="country")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Rainfall data not available.")

def climate_clustering(df):
    st.subheader("🌎 Climate Zone Clustering (K-Means)")
    if not df.empty:
        features = [temp_col]
        if "humidity" in df.columns:
            features.append("humidity")
        if "precip_mm" in df.columns:
            features.append("precip_mm")

        cluster_data = df.groupby("country")[features].mean().dropna()

        if len(cluster_data) > 3:
            scaler = StandardScaler()
            X = scaler.fit_transform(cluster_data)

            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_data["cluster"] = kmeans.fit_predict(X)

            fig = px.scatter(
                cluster_data,
                x=features[0],
                y=features[1] if len(features) > 1 else features[0],
                color="cluster",
                hover_name=cluster_data.index,
                title="Climate Zone Clustering"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough countries for clustering.")
    else:
        st.write("No data available.")

def climate_risk_index(df):
    st.subheader("⚠ Climate Risk Index")
    if not df.empty:
        agg_dict = {temp_col: "mean"}
        if "precip_mm" in df.columns:
            agg_dict["precip_mm"] = "mean"

        risk = df.groupby("country").agg(agg_dict).reset_index()
        risk["risk_score"] = risk[temp_col].rank(pct=True)

        top_risk = risk.sort_values("risk_score", ascending=False).head(10)
        fig = px.bar(
            top_risk,
            x="risk_score",
            y="country",
            orientation="h",
            title="Top Climate Risk Countries"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def climate_change_indicator(df):
    st.subheader("🌡 Climate Change Indicator")
    if not df.empty:
        trend = df.groupby("year")[temp_col].mean().reset_index()
        if len(trend) > 1:
            X = trend[["year"]]
            y = trend[temp_col]

            model = LinearRegression()
            model.fit(X, y)

            slope = model.coef_[0]
            st.metric("Temperature Trend per Year", f"{slope:.3f} °C/year")

            fig = px.scatter(trend, x="year", y=temp_col, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for climate change indicator.")
    else:
        st.write("No data available.")

def data_export(df):
    st.subheader("⬇ Data Export")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="filtered_weather_data.csv")

# Global choropleth map
def global_choropleth_map(df):
    st.subheader("🌍 Global Temperature Choropleth")
    if not df.empty:
        # Need to map country names to ISO codes
        country_temp = df.groupby("country")[temp_col].mean().reset_index()
        # Assuming country names are standard, use plotly's built-in
        fig = px.choropleth(country_temp, locations="country", locationmode="country names", color=temp_col, color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

# -------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------

st.sidebar.header("🔎 Filters")

country_options = sorted(df["country"].dropna().unique())

# default to India + first other available country (if present)
default_countries = []
if "India" in country_options:
    default_countries.append("India")
if country_options:
    first_other = next((c for c in country_options if c != "India"), None)
    if first_other:
        default_countries.append(first_other)

countries = st.sidebar.multiselect(
    "Select Countries",
    country_options,
    default=default_countries
)

years = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

filtered_df = df[
    (df["country"].isin(countries)) &
    (df["year"].between(years[0], years[1]))
]

# -------------------------------------------------------
# FEATURE GROUPS
# -------------------------------------------------------

feature_groups = {
    "Overview": {
        "Key Metrics": show_key_metrics,
        "Hottest Countries": hottest_countries,
        "Top Countries Comparison": top_countries_comparison,
        "Global Insights": global_insight_summary_panel,
        "Feature Explanations": interactive_feature_explanations,
    },
    "Trends & Patterns": {
        "Long-term Trend": long_term_trend,
        "Rolling Average": rolling_avg,
        "Year-over-Year": yoy,
        "Time Series Decomposition": time_series_decomposition,
        "Seasonal Rainfall": seasonal_rainfall_comparison,
    },
    "Anomalies": {
        "Extreme Events": extreme_events,
        "Heatwave Detection": heatwave_detection,
        "DBSCAN Anomalies": dbscan_anomaly,
    },
    "Distribution": {
        "Temperature Distribution": show_distribution,
        "Seasonal Heatmap": seasonal_heatmap,
        "Monthly Distribution": monthly_dist,
        "Percentile Analysis": percentile,
        "Weather Conditions": conditions,
    },
    "Forecasting & Correlation": {
        "Temperature Forecast": forecast,
        "Correlation Heatmap": corr_vars,
        "Similarity Matrix": similarity,
        "Kruskal-Wallis Test": kruskal_test,
        "Climate Change Indicator": climate_change_indicator,
        "Climate Risk Index": climate_risk_index,
        "Climate Clustering": climate_clustering,
    },
    "Maps": {
        "Global Choropleth": global_choropleth_map,
        "Lat-Long Scatter": latitude_longitude_map,
    },
    "Other": {
        "Precipitation Analysis": precipitation_rainfall,
        "Multi-parameter": multi_param,
        "Humidity & Wind": humidity_wind,
        "Data Export": data_export,
    }
}

# -------------------------------------------------------
# MAIN INTERFACE
# -------------------------------------------------------

tabs = st.tabs(list(feature_groups.keys()))

for tab, (tab_name, features) in zip(tabs, feature_groups.items()):
    with tab:
        st.header(f"📊 {tab_name}")
        selected_features = []
        for feature_name in features.keys():
            if st.checkbox(feature_name, key=f"{tab_name}_{feature_name}", value=True):
                selected_features.append(feature_name)
        
        for feature in selected_features:
            features[feature](filtered_df)
            st.markdown("---")

st.success("✅ ClimateScope Global Weather Intelligence System dashboard is ready. Adjust the sidebar filters to explore global weather insights in real time.")