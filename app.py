"""
ClimateScope - Advanced Global Weather Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(page_title="ClimateScope Advanced", layout="wide")
st.title("🌍 ClimateScope - Global Weather Intelligence System")

# Quick guide
with st.expander("📋 Menu Guide", expanded=False):
    st.markdown("""
    **Navigation Menu:**
    - **📈 Overview**: Basic metrics, distributions, and correlations
    - **📉 Trends & Patterns**: Long-term trends, rolling averages, seasonal patterns
    - **⚠️ Anomalies**: Extreme events and anomaly detection methods
    - **📊 Distribution**: Monthly patterns and percentile analysis
    - **🔮 Forecasting & Correlation**: Predictions and variable relationships
    - **💾 Data**: Summary statistics and data export
    
    **How to use:**
    1. Select countries in filters
    2. Choose a menu tab on the left
    3. Toggle features on/off with checkboxes
    4. Or enable "Show All Features" to view everything
    """)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("processed_weather_data.csv")
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    return df

df = load_data()

# Detect temperature column automatically
temp_col = None
for col in df.columns:
    if "temp" in col.lower() and "celsius" in col.lower():
        temp_col = col
        break

if temp_col is None:
    st.error("No temperature column found!")
    st.stop()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------

st.sidebar.header("🔍 Filters")

country_list = sorted(df["country"].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    country_list,
    default=country_list[:2]
)

if not selected_countries:
    st.warning("Select at least one country")
    st.stop()

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

rolling_window = st.sidebar.slider("Rolling Window (Days)", 7, 90, 30)

# -------------------------------------------------
# FEATURE SELECTION WITH MENU TABS
# -------------------------------------------------

st.sidebar.header("📊 Analysis Menu")

# Define feature groups
feature_groups = {
    "📈 Overview": [
        "Key Metrics",
        "Hottest Countries",
        "Hottest Country Globe",
        "Latitude Longitude Map",
        "Top 10 Countries Comparison",
        "Distribution Analysis",
        "Correlation Heatmap",
        "Seasonal Heatmap"
    ],
    "📉 Trends & Patterns": [
        "Long-Term Trend",
        "Rolling Average",
        "Year-over-Year Comparison",
        "Precipitation & Rainfall Regions",
        "Multi-Parameter Analysis",
            "Humidity & Wind Rolling"
    ],
    "⚠️ Anomalies": [
        "Extreme Event Detection",
        "DBSCAN Anomaly Detection",
        "Climate Similarity Matrix",
        "Advanced Country Comparison"
    ],
    "📊 Distribution": [
        "Monthly Distribution Patterns",
        "Percentile Analysis",
        "Weather Conditions Analysis"
    ],
    "🔮 Forecasting & Correlation": [
        "Temperature Forecast",
        "Correlation with Variables"
    ],
    "💾 Data": [
        "Data Summary & Export"
    ]
}

# Create tabs in sidebar
menu_tab = st.sidebar.radio("📌 Select Tab", list(feature_groups.keys()))

selected_features = []

# Display checkbox for each feature in the selected tab
st.sidebar.subheader(f"Features in {menu_tab}")

for feature in feature_groups[menu_tab]:
    if st.sidebar.checkbox(feature, value=True):
        selected_features.append(feature)

# Add option to see all features
if st.sidebar.checkbox("✓ Show All Features"):
    selected_features = []
    for group in feature_groups.values():
        selected_features.extend(group)

# Filter Data
filtered_df = df[
    (df["country"].isin(selected_countries)) &
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

# Add season column
def get_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

filtered_df["season"] = filtered_df["month"].apply(get_season)

# -------------------------------------------------
# MAIN CONTENT HEADER
# -------------------------------------------------

st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0;">📂 Active Menu: {menu_tab}</h2>
    <p style="color: #e0e0e0; margin: 5px 0 0 0;">Showing {len(selected_features)} selected features</p>
</div>
""", unsafe_allow_html=True)

st.divider()

def show_key_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    avg_temp, max_temp = df[temp_col].mean(), df[temp_col].max()
    min_temp, var = df[temp_col].min(), df[temp_col].var()
    col1.metric("Avg Temp", f"{avg_temp:.2f}°C")
    col2.metric("Max Temp", f"{max_temp:.2f}°C")
    col3.metric("Min Temp", f"{min_temp:.2f}°C")
    col4.metric("Variance", f"{var:.2f}")


def hottest_countries(df):
    st.subheader("🔥 Hottest Countries")
    # calculate average temperature by country and sort descending
    avgs = df.groupby("country")[temp_col].mean().sort_values(ascending=False)
    if not avgs.empty:
        hottest = avgs.index[0]
        st.metric("Hottest Country", f"{hottest} ({avgs.iloc[0]:.2f}°C)")
        top10 = avgs.head(10).reset_index()
        top10.columns = ["Country", "Avg Temp"]
        st.table(top10)
    else:
        st.write("No data available to determine hottest countries.")

if "Key Metrics" in selected_features:
    show_key_metrics(filtered_df)
    st.divider()

def show_distribution(df):
    col1, col2 = st.columns(2)
    fig = px.histogram(
        df,
        x=temp_col,
        nbins=40,
        color="country",
        animation_frame="country",
        title="Temperature Distribution",
        barmode='overlay'
    )
    fig.update_traces(opacity=0.7)
    fig.update_layout(
        height=500,
        transition_duration=300,
        barmode='overlay'
    )
    col1.plotly_chart(fig, use_container_width=True)
    col2.write(df[temp_col].describe())
    col2.metric("Skew", round(skew(df[temp_col].dropna()),2))
    col2.metric("Kurt", round(kurtosis(df[temp_col].dropna()),2))

if "Distribution Analysis" in selected_features:
    show_distribution(filtered_df)
    st.divider()

if "Correlation Heatmap" in selected_features:
    st.plotly_chart(px.imshow(filtered_df.select_dtypes(include=np.number).corr(), text_auto=True),use_container_width=True)
    st.divider()


# feature-specific functions
def seasonal_heatmap(df):
    st.subheader("🌦 Seasonal Temperature Pattern")
    sa = df.groupby(["country","season"])[temp_col].mean().reset_index()
    fig = px.density_heatmap(
        sa,
        x="season",
        y="country",
        z=temp_col,
        color_continuous_scale="Viridis",
        animation_frame="country",
        title="Seasonal Temperature Heatmap"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig,use_container_width=True)

def long_term_trend(df):
    st.subheader("📉 Long-Term Trend")
    td = df.groupby("year")[temp_col].mean().reset_index()
    X,y=td["year"].values.reshape(-1,1),td[temp_col].values
    m=LinearRegression().fit(X,y)
    td["trend"]=m.predict(X)
    
    fig = px.line(
        td,
        x="year",
        y=[temp_col,"trend"],
        title="Yearly Avg Temperature with Trend",
        markers=True,
        animation_frame="year"
    )
    fig.update_layout(
        hovermode='x unified',
        transition_duration=300
    )
    st.plotly_chart(fig, use_container_width=True)

def rolling_avg(df):
    st.subheader("📊 Rolling Avg")
    rd=[]
    for c in selected_countries:
        cd=df[df["country"]==c].sort_values("last_updated").copy()
        cd["roll"]=cd[temp_col].rolling(rolling_window,min_periods=1).mean()
        rd.append(cd)
    
    fig = px.line(
        pd.concat(rd),
        x="last_updated",
        y="roll",
        color="country",
        animation_frame="month",
        animation_group="country",
        title="Rolling Average Temperature by Month"
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def extreme_events(df):
    st.subheader("⚠️ Extremes")
    hi=df[df[temp_col]>df[temp_col].quantile(0.95)]
    lo=df[df[temp_col]<df[temp_col].quantile(0.05)]
    col1,col2=st.columns(2)
    col1.metric("Heat",len(hi));col2.metric("Cold",len(lo))

def similarity(df):
    st.subheader("🌍 Similarity")
    cm=df.groupby("country")[temp_col].mean()
    st.plotly_chart(px.imshow(abs(cm.values[:,None]-cm.values),x=cm.index,y=cm.index),use_container_width=True)

def kruskal(df):
    st.subheader("🔬 Kruskal-Wallis")
    if len(selected_countries)>=2:
        groups=[df[df["country"]==c][temp_col].values for c in selected_countries]
        stat,p=stats.kruskal(*groups)
        st.metric("H",f"{stat:.3f}")
        st.metric("p",f"{p:.4f}")
        st.write("**H** is the Kruskal-Wallis H statistic indicating the degree of difference between group medians.  ")
        st.write("**p** is the associated p-value; a small p (e.g. <0.05) suggests a significant difference among countries.")

def multi_param(df):
    st.subheader("🌡 Multi-Param")
    if "humidity" in df and "precip_mm" in df:
        mm=df.groupby(["country","month"]).agg({temp_col:"mean","humidity":"mean","precip_mm":"sum"}).reset_index()
        for var,title in [(temp_col,"Temp"),("humidity","Humidity")]:
            st.plotly_chart(px.line(mm,x="month",y=var,color="country",title=title),use_container_width=True)

def yoy(df):
    st.subheader("📅 Year-over-Year Comparison")
    if df["year"].nunique()>1:
        yd=df.groupby(["year","month"])[temp_col].mean().reset_index()
        fig = px.line(
            yd,
            x="month",
            y=temp_col,
            color="year",
            animation_frame="year",
            animation_group="month",
            markers=True,
            title="Temperature Trends Year-over-Year"
        )
        fig.update_layout(
            hovermode='x unified',
            transition_duration=300
        )
        st.plotly_chart(fig, use_container_width=True)

def humidity_wind(df):
    st.subheader("💨 Humidity/Wind Rolling")
    if "humidity" in df and "wind_kph" in df:
        rd=[]
        for c in selected_countries:
            cd=df[df["country"]==c].sort_values("last_updated").copy()
            cd["hroll"]=cd["humidity"].rolling(rolling_window,min_periods=1).mean()
            cd["wroll"]=cd["wind_kph"].rolling(rolling_window,min_periods=1).mean()
            rd.append(cd)
        rpd=pd.concat(rd)
        st.plotly_chart(px.line(rpd,x="last_updated",y="hroll",color="country"),use_container_width=True)
        st.plotly_chart(px.line(rpd,x="last_updated",y="wroll",color="country"),use_container_width=True)

def dbscan_anomaly(df):
    st.subheader("🔍 DBSCAN")
    nf=[temp_col]+(["humidity"] if "humidity" in df else[])+(["wind_kph"] if "wind_kph" in df else[])
    X=df[nf].fillna(df[nf].mean()).values
    labels=DBSCAN(eps=0.5,min_samples=5).fit_predict(StandardScaler().fit_transform(X))
    dfc=df.copy();dfc['cluster']=labels
    an=dfc[dfc['cluster']==-1]
    st.metric("Anomalies",len(an))
    if len(an):st.plotly_chart(px.scatter(an,x="last_updated",y=temp_col,color="country"),use_container_width=True)

def monthly_dist(df):
    st.subheader("📊 Monthly Distribution")
    fig = px.box(
        df,
        x="month",
        y=temp_col,
        color="country",
        animation_frame="country",
        title="Monthly Temperature Distribution by Country"
    )
    fig.update_layout(height=500, transition_duration=300)
    st.plotly_chart(fig, use_container_width=True)

def percentile(df):
    st.subheader("📈 Percentiles")
    ps=[10,25,50,75,90,95,99]
    rows=[[c]+[np.percentile(df[df['country']==c][temp_col],p) for p in ps] for c in selected_countries]
    pdft=pd.DataFrame(rows,columns=["Country"]+[f"P{p}" for p in ps])
    pm=pdft.melt(id_vars="Country",var_name="Perc",value_name="Temp")
    
    fig = px.line(
        pm,
        x="Perc",
        y="Temp",
        color="Country",
        markers=True,
        animation_frame="Country",
        title="Temperature Percentiles by Country"
    )
    fig.update_layout(
        height=500,
        transition_duration=300,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def conditions(df):
    st.subheader("🌦 Conditions")
    if "condition_text" in df:
        counts = df['condition_text'].value_counts().reset_index()
        counts.columns = ['condition_text', 'count']
        st.plotly_chart(
            px.pie(counts, names='condition_text', values='count'),
            use_container_width=True
        )

def forecast(df):
    st.subheader("🔮 Forecast")
    m=LinearRegression()
    for c in selected_countries:
        cd=df[df['country']==c].groupby('month')[temp_col].mean().reset_index()
        X=cd['month'].values.reshape(-1,1);y=cd[temp_col].values
        m.fit(X,y)
        fm=np.arange(1,13+st.session_state.get('forecast',3)).reshape(-1,1)
        st.plotly_chart(px.line(pd.DataFrame({'month':fm.flatten(),'temp':m.predict(fm)}),x='month',y='temp'),use_container_width=True)

def corr_vars(df):
    st.subheader("🔗 Corr")
    vars=['humidity','wind_kph','precip_mm','cloud']
    for v in vars:
        if v in df:
            st.metric(v,df[[temp_col,v]].corr().iloc[0,1])

def data_export(df):
    st.subheader("💾 Data")
    st.download_button("Download",df.to_csv(index=False),"data.csv","text/csv")

def precipitation_rainfall(df):
    st.subheader("🌧️ Precipitation & Rainfall Regions")
    if "precip_mm" in df:
        # Identify high rainfall regions
        precip_by_country = df.groupby("country")["precip_mm"].agg(["mean", "max", "sum"]).reset_index()
        precip_by_country.columns = ["Country", "Avg Precip (mm)", "Max Precip (mm)", "Total Precip (mm)"]
        precip_by_country = precip_by_country.sort_values("Avg Precip (mm)", ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Rainfall Regions (by Average)**")
            st.table(precip_by_country.head(10))
        
        with col2:
            st.write("**Precipitation Distribution by Country**")
            st.plotly_chart(
                px.bar(precip_by_country, x="Country", y="Avg Precip (mm)", color="Avg Precip (mm)", 
                       color_continuous_scale="Blues"),
                use_container_width=True
            )
        
        # Heavy rainfall region detection
        threshold = df["precip_mm"].quantile(0.90)
        heavy_rain = df[df["precip_mm"] > threshold]
        st.write(f"**Heavy Rainfall Events (>90th percentile: {threshold:.1f}mm)**")
        heavy_by_country = heavy_rain.groupby("country").size().reset_index(name="Count")
        heavy_by_country = heavy_by_country.sort_values("Count", ascending=False)
        st.plotly_chart(
            px.bar(heavy_by_country, x="country", y="Count", title="Countries with Heavy Rainfall Events"),
            use_container_width=True
        )
    else:
        st.warning("Precipitation data not available in dataset.")

def hottest_on_globe(df):
    st.subheader("🔥 World Temperature Map - Hottest & Coldest")
    # Calculate average temperature by country
    country_temps = df.groupby("country")[temp_col].mean().reset_index()
    country_temps.columns = ["Country", "Avg Temp"]
    country_temps = country_temps.sort_values("Avg Temp", ascending=False)
    
    hottest_country = country_temps.iloc[0]["Country"]
    hottest_temp = country_temps.iloc[0]["Avg Temp"]
    coldest_country = country_temps.iloc[-1]["Country"]
    coldest_temp = country_temps.iloc[-1]["Avg Temp"]
    
    col1, col2 = st.columns(2)
    col1.metric("🔥 Hottest Country", f"{hottest_country} ({hottest_temp:.2f}°C)")
    col2.metric("Coldest Country", f"{coldest_country} ({coldest_temp:.2f}°C)")
    
    # Create a Choropleth map with diverging color scale (red for hot, blue for cold)
    country_iso_map = {
        "India": "IND", "USA": "USA", "China": "CHN", "Japan": "JPN",
        "Germany": "DEU", "France": "FRA", "UK": "GBR", "Brazil": "BRA",
        "Mexico": "MEX", "Russia": "RUS", "Canada": "CAN", "Australia": "AUS",
        "South Africa": "ZAF", "Egypt": "EGY", "Nigeria": "NGA", "Thailand": "THA",
        "Vietnam": "VNM", "Indonesia": "IDN", "Malaysia": "MYS", "Pakistan": "PAK",
        "Bangladesh": "BGD", "Philippines": "PHL", "South Korea": "KOR", "Italy": "ITA",
        "Spain": "ESP", "Netherlands": "NLD", "Belgium": "BEL", "Poland": "POL",
        "Turkey": "TUR", "Iran": "IRN", "UAE": "ARE", "Saudi Arabia": "SAU",
        "Israel": "ISR", "Singapore": "SGP", "Hong Kong": "HKG", "New Zealand": "NZL",
        "Sweden": "SWE", "Norway": "NOR", "Finland": "FIN", "Denmark": "DNK",
        "Portugal": "PRT", "Greece": "GRC", "Ukraine": "UKR", "Argentina": "ARG",
        "Chile": "CHL", "Colombia": "COL", "Peru": "PER", "Venezuela": "VEN"
    }
    
    country_temps["ISO"] = country_temps["Country"].map(country_iso_map)
    country_temps_valid = country_temps.dropna(subset=["ISO"])
    
    if not country_temps_valid.empty:
        # Use RdBu_r scale: Red for hot, Blue for cold
        fig = go.Figure(data=go.Choropleth(
            locations=country_temps_valid["ISO"],
            z=country_temps_valid["Avg Temp"],
            text=country_temps_valid["Country"],
            colorscale="RdBu_r",
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            showscale=True,
            colorbar=dict(title="Temp (°C)", ticks="outside", lenmode="fraction", len=0.5),
            hovertemplate="<b>%{text}</b><br>Avg Temp: %{z:.2f}°C<extra></extra>"
        ))
        fig.update_layout(
            title="Global Temperature Distribution: Hottest (Red) to Coldest (Blue)",
            geo=dict(
                projection_type="natural earth",
                showland=True,
                landcolor='rgb(243, 243, 243)',
                bgcolor='rgba(240, 248, 255, 0.5)'
            ),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Unable to show map - country data not mappable.")
        st.write("**Temperature by Country (Hottest to Coldest):**")
        st.table(country_temps.sort_values("Avg Temp", ascending=False))

def latitude_longitude_map(df):
    st.subheader("🗺️ Latitude & Longitude Temperature Distribution")
    
    if "latitude" not in df or "longitude" not in df:
        st.warning("Latitude/Longitude data not available.")
        return
    
    # Create scatter plot with lat/lon coordinates
    plot_df = df[["latitude", "longitude", temp_col, "country"]].copy()
    plot_df = plot_df.dropna(subset=["latitude", "longitude", temp_col])
    
    # Create the scatter map
    fig = px.scatter_geo(
        plot_df,
        lat="latitude",
        lon="longitude",
        color=temp_col,
        hover_name="country",
        hover_data={temp_col: ":.2f", "latitude": ":.3f", "longitude": ":.3f"},
        color_continuous_scale="RdYlBu_r",
        title="Temperature Distribution by Location (Latitude & Longitude)",
        size_max=8,
        projection="natural earth"
    )
    
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            bgcolor='rgba(240, 248, 255, 0.5)',
            coastlinecolor='rgb(150, 150, 150)'
        ),
        height=700,
        coloraxis_colorbar=dict(title="Temp (°C)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Total Locations", len(plot_df))
    col2.metric("🔥 Max Temp", f"{plot_df[temp_col].max():.2f}°C")
    col3.metric("❄ Min Temp", f"{plot_df[temp_col].min():.2f}°C")

def top_countries_comparison(df):
    st.subheader("📊 Top 10 Countries Comparison")
    
    # Calculate metrics by country
    country_stats = df.groupby("country").agg({
        temp_col: "mean",
        "precip_mm": "sum" if "precip_mm" in df else "mean",
        "humidity": "mean" if "humidity" in df else "mean"
    }).reset_index()
    
    country_stats.columns = ["Country", "Avg_Temp", "Total_Precip", "Avg_Humidity"]
    
    # Create tabs for each comparison
    tab1, tab2, tab3 = st.tabs(["🔥 Top 10 Hottest", "❄ Top 10 Coldest", "🌧 Top 10 Rainy"])
    
    with tab1:
        top_hot = country_stats.nlargest(10, "Avg_Temp").reset_index(drop=True)
        top_hot["Rank"] = range(1, len(top_hot) + 1)
        fig_hot = px.bar(
            top_hot,
            x="Country",
            y="Avg_Temp",
            color="Avg_Temp",
            color_continuous_scale="Reds",
            title="Top 10 Hottest Countries",
            labels={"Avg_Temp": "Average Temperature (°C)"},
            text="Avg_Temp",
            animation_frame="Rank"
        )
        fig_hot.update_traces(texttemplate='%{text:.1f}°C', textposition='outside')
        fig_hot.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
        fig_hot.update_xaxes(categoryorder="total ascending")
        st.plotly_chart(fig_hot, use_container_width=True)
        
        # Display table
        st.write("### Hottest Countries Details")
        top_hot_display = top_hot.copy()
        top_hot_display["Avg_Temp"] = top_hot_display["Avg_Temp"].round(2)
        top_hot_display["Rank"] = range(1, len(top_hot_display) + 1)
        st.table(top_hot_display[["Rank", "Country", "Avg_Temp"]])
    
    with tab2:
        top_cold = country_stats.nsmallest(10, "Avg_Temp").reset_index(drop=True)
        top_cold["Rank"] = range(1, len(top_cold) + 1)
        fig_cold = px.bar(
            top_cold,
            x="Country",
            y="Avg_Temp",
            color="Avg_Temp",
            color_continuous_scale="Blues",
            title="Top 10 Coldest Countries",
            labels={"Avg_Temp": "Average Temperature (°C)"},
            text="Avg_Temp",
            animation_frame="Rank"
        )
        fig_cold.update_traces(texttemplate='%{text:.1f}°C', textposition='outside')
        fig_cold.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
        fig_cold.update_xaxes(categoryorder="total ascending")
        st.plotly_chart(fig_cold, use_container_width=True)
        
        # Display table
        st.write("### Coldest Countries Details")
        top_cold_display = top_cold.copy()
        top_cold_display["Avg_Temp"] = top_cold_display["Avg_Temp"].round(2)
        top_cold_display["Rank"] = range(1, len(top_cold_display) + 1)
        st.table(top_cold_display[["Rank", "Country", "Avg_Temp"]])
    
    with tab3:
        top_rainy = country_stats.nlargest(10, "Total_Precip").reset_index(drop=True)
        top_rainy["Rank"] = range(1, len(top_rainy) + 1)
        fig_rainy = px.bar(
            top_rainy,
            x="Country",
            y="Total_Precip",
            color="Total_Precip",
            color_continuous_scale="Blues",
            title="Top 10 Rainiest Countries (Total Precipitation)",
            labels={"Total_Precip": "Total Precipitation (mm)"},
            text="Total_Precip",
            animation_frame="Rank"
        )
        fig_rainy.update_traces(texttemplate='%{text:.0f}mm', textposition='outside')
        fig_rainy.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
        fig_rainy.update_xaxes(categoryorder="total ascending")
        st.plotly_chart(fig_rainy, use_container_width=True)
        
        # Display table
        st.write("### Rainiest Countries Details")
        top_rainy_display = top_rainy.copy()
        top_rainy_display["Total_Precip"] = top_rainy_display["Total_Precip"].round(2)
        top_rainy_display["Rank"] = range(1, len(top_rainy_display) + 1)
        st.table(top_rainy_display[["Rank", "Country", "Total_Precip"]])

# register functions
feature_funcs={
    "Seasonal Heatmap": seasonal_heatmap,
    "Long-Term Trend": long_term_trend,
    "Rolling Average": rolling_avg,
    "Hottest Countries": hottest_countries,
    "Hottest Country Globe": hottest_on_globe,
    "Latitude Longitude Map": latitude_longitude_map,
    "Top 10 Countries Comparison": top_countries_comparison,
    "Precipitation & Rainfall Regions": precipitation_rainfall,
    "Extreme Event Detection": extreme_events,
    "Climate Similarity Matrix": similarity,
    "Advanced Country Comparison": kruskal,
    "Multi-Parameter Analysis": multi_param,
    "Year-over-Year Comparison": yoy,
    "Humidity & Wind Rolling": humidity_wind,
    "DBSCAN Anomaly Detection": dbscan_anomaly,
    "Monthly Distribution Patterns": monthly_dist,
    "Percentile Analysis": percentile,
    "Weather Conditions Analysis": conditions,
    "Temperature Forecast": forecast,
    "Correlation with Variables": corr_vars,
    "Data Summary & Export": data_export
}

# execute selected
for feat in selected_features:
    feature_funcs.get(feat,lambda df:None)(filtered_df)
    st.divider()

st.success("✅ Advanced ClimateScope Intelligence Dashboard Ready")