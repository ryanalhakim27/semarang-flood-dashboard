# ==========================================================
# 🌊 Semarang Flood–Rainfall Dashboard + LULC Map Viewer
# ==========================================================


import rasterio
import numpy as np
import tempfile
import os
import json
import streamlit as st
import pandas as pd
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import plotly.express as px
from geopy.distance import geodesic
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO


# Set wide layout
st.set_page_config(page_title="Semarang Flood Monitor", layout="wide")

st.markdown(
    """
    <div style='
        text-align:center; 
        font-family: -apple-system, BlinkMacSystemFont, "San Francisco", Helvetica, Arial, sans-serif; 
        max-width:90%; 
        margin:auto;
    '>
        <h1 style='
            font-size: clamp(24px, 4vw, 48px); 
            font-weight:bold;
            margin-bottom:10px;
        '>
            Semarang Flood Monitor
        </h1>
        <p style='
            font-size: clamp(12px, 2vw, 20px); 
            color: #f0f0f0;
        '>
            Interactive dashboard of flood incidents, rainfall-climate data, and runoff & flood risk analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load logo
logo = Image.open("logo.png")

# Convert logo to base64
buffered = BytesIO()
logo.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Logo (centered, responsive)
st.sidebar.markdown(
    f"""
    <p style="text-align:center;">
        <img src="data:image/png;base64,{img_str}" style="max-width:80%; height:auto; margin-bottom:20px;"/>
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar branding (responsive fonts)
st.sidebar.markdown("""
<p style="text-align:center; font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', Helvetica, Arial, sans-serif; margin:0;">
    <span style="display:block; font-size:clamp(18px, 2.5vw, 24px); font-weight:bold; margin-bottom:4px;">TitikTahan</span>
    <span style="display:block; font-size:clamp(14px, 2vw, 20px); font-weight:bold; margin-bottom:6px;">Resilience Points</span>
    <span style="display:block; font-size:clamp(12px, 1.5vw, 16px); margin-bottom:4px;">Mapping slopes, sediments & survival</span>
    <span style="display:block; font-size:clamp(12px, 1.5vw, 16px); font-weight:bold; margin-bottom:12px;">Land • Water • City • People</span>
</p>
""", unsafe_allow_html=True)

# Instagram link (responsive)
st.sidebar.markdown(
    '<p style="text-align:center; font-size:clamp(12px, 1.5vw, 16px); margin-top:8px;"><a href="https://www.instagram.com/titiktahan/" target="_blank" style="text-decoration:none;">📸 Follow us on Instagram</a></p>',
    unsafe_allow_html=True
)


# ==========================================================
# Load Data
# ==========================================================
@st.cache_data
def load_data():
    flood_path = "data/flood_data_with_latlon.csv"
    rain_path = "data/rainfall_data_filtered.csv"


    for path in [flood_path, rain_path]:
        if not os.path.exists(path):
            st.error(f"❌ File not found: {path}")
            st.stop()

    flood_df = pd.read_csv(flood_path)
    rain_df = pd.read_csv(rain_path)

    flood_df['date'] = pd.to_datetime(flood_df['date'], format="%m/%d/%y", errors='coerce')
    rain_df['date'] = pd.to_datetime(rain_df['date'], format="%m/%d/%y", errors='coerce')

    flood_df['lat'] = pd.to_numeric(flood_df['lat'], errors='coerce')
    flood_df['lon'] = pd.to_numeric(flood_df['lon'], errors='coerce')
    flood_df = flood_df.dropna(subset=['lat', 'lon'])

    rain_df['Lat_DD'] = pd.to_numeric(rain_df['Lat_DD'], errors='coerce')
    rain_df['Lon_DD'] = pd.to_numeric(rain_df['Lon_DD'], errors='coerce')
    rain_df = rain_df.dropna(subset=['Lat_DD', 'Lon_DD'])

    rain_df = rain_df.dropna(subset=['date'])
    flood_df = flood_df.dropna(subset=['date'])


    return flood_df, rain_df

flood_df, rain_df = load_data()

# ==========================================================
# Prepare Map Data
# ==========================================================
station_map_df = rain_df[['Lat_DD','Lon_DD','location_name']].drop_duplicates().copy()
station_map_df['hover_text'] = station_map_df['location_name']
station_map_df['type'] = 'Station'
station_map_df = station_map_df.rename(columns={'Lat_DD':'lat','Lon_DD':'lon'})

flood_map_df = flood_df[['lat','lon','date','Name_of_area']].copy()

def find_nearest_station(lat, lon, station_df):
    min_dist = float('inf')
    nearest_name = None
    nearest_coords = None
    for _, s in station_df.iterrows():
        dist = geodesic((lat, lon), (s['lat'], s['lon'])).meters
        if dist < min_dist:
            min_dist = dist
            nearest_name = s['hover_text']
            nearest_coords = (s['lat'], s['lon'])
    return nearest_name, nearest_coords

flood_map_df[['nearest_station', 'nearest_coords']] = flood_map_df.apply(
    lambda x: find_nearest_station(x['lat'], x['lon'], station_map_df), axis=1, result_type="expand"
)
flood_map_df['hover_text'] = (
    flood_map_df['date'].dt.strftime('%Y-%m-%d') + 
    " | " + flood_map_df['Name_of_area'] + 
    " | Nearest Station: " + flood_map_df['nearest_station']
)


st.markdown(
    """
    <div id="flood_overview" style="
        margin-top:50px; 
        font-size: clamp(20px, 4vw, 32px);
        font-weight: bold;
        line-height: 1.2;
    ">
        Flood Incidents & Rainfall Overview
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Station Selection & Slider with Flood Date Reference
# -----------------------------
station_options = ["All Stations"] + list(station_map_df['hover_text'])
selected_station = st.selectbox("Select Station", station_options)

st.markdown(
    """
    <div style="
        max-width:90%; 
        margin:10px auto; 
        text-align:center; 
        font-family:-apple-system, BlinkMacSystemFont, 'San Francisco', Helvetica, Arial, sans-serif;
    ">
        <p style="
            font-weight:bold; 
            font-size:clamp(14px, 2vw, 18px); 
            margin-bottom:4px;
        ">
            Reference: Flood Incident Dates
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Show only the unique flood dates without index
unique_flood_dates = flood_map_df['date'].dt.date.drop_duplicates().sort_values().reset_index(drop=True)
st.table(unique_flood_dates.to_frame(name="Flood Date"))

# -----------------------------
# Slider
# -----------------------------
min_date = flood_map_df['date'].min().date()
max_date = flood_map_df['date'].max().date()

st.markdown(
    '<div style="max-width:600px; margin:10px auto 30px auto;">', 
    unsafe_allow_html=True
)
selected_date = st.slider(
    "Select Flood Date",
    min_value=min_date,
    max_value=max_date,
    value=min_date,
    format="YYYY-MM-DD"
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Filter flood data
# -----------------------------
flood_selected_df = flood_map_df[flood_map_df['date'].dt.date == selected_date]

# Checkbox: show all flood data
if selected_station == "All Stations":
    show_all_floods = st.checkbox("Show all flood data", value=False)
    if show_all_floods:
        flood_to_plot = flood_map_df
    else:
        flood_to_plot = flood_selected_df
else:
    flood_to_plot = flood_selected_df

st.subheader("🗺️ Flood and Climate Station Point Map")
# -----------------------------
# Map Section
# -----------------------------
if not flood_to_plot.empty:
    flood_coords = flood_to_plot[['lat','lon']].mean().to_list()
    nearest_coords = flood_to_plot.iloc[0]['nearest_coords']
    if nearest_coords:
        center_lat = (flood_coords[0] + nearest_coords[0]) / 2
        center_lon = (flood_coords[1] + nearest_coords[1]) / 2
    else:
        center_lat, center_lon = flood_coords
else:
    center_lat = station_map_df['lat'].mean()
    center_lon = station_map_df['lon'].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

# Add stations
for _, row in station_map_df.iterrows():
    icon_color = 'green' if row['hover_text'] == selected_station else 'blue'
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['hover_text'],
        tooltip=row['hover_text'],
        icon=folium.Icon(color=icon_color, icon='info-sign')
    ).add_to(m)

# Add flood points
for _, row in flood_to_plot.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['hover_text'],
        tooltip=row['hover_text'],
        icon=folium.Icon(color='red', icon='tint')
    ).add_to(m)

# Adjust map container width for responsiveness
st_folium(m, width=None, height=500)


st.subheader("📈 Rainfall Trends")

# Filter data by selected station and make a copy
rain_plot_df = (
    rain_df.copy() if selected_station == "All Stations" 
    else rain_df[rain_df['location_name'] == selected_station].copy()
)

# Drop invalid rows
rain_plot_df = rain_plot_df.dropna(subset=['date', 'rainfall(mm)'])

# Plot raw daily rainfall as line chart
fig = px.line(
    rain_plot_df,
    x='date',
    y='rainfall(mm)',
    color='location_name',
    title="Daily Rainfall (mm)",
    markers=True
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Rainfall (mm)",
    height=1000,
    width=1200,
    showlegend=True
)
fig.update_xaxes(tickangle=45)
fig.update_layout(xaxis_rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)



# ----------------------------------------------
# 🌊 KALI BABON WATERSHED ANALYSIS
# ----------------------------------------------
st.markdown(
    """
    <div id="kalibabon-section" style="
        max-width:90%;
        margin:60px auto 20px auto; 
        line-height:1.8; 
        text-align:justify; 
        color:white;
    ">
        <h1 style="
            color:white;
            font-size:clamp(24px, 4vw, 36px);
            font-weight:700;
            margin-bottom:20px;
            text-align:center;
        ">
            Kali Babon Watershed Runoff Analysis
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style="
        max-width:90%;
        margin:20px auto; 
        line-height:1.8; 
        text-align:justify; 
        color:white;
    ">
        <h2 style="
            color:white; 
            font-size:clamp(20px, 2.3vw, 28px); 
            margin-bottom:15px;
        ">
            Selection of Area
        </h2>
        <p style="
            font-size:clamp(14px, 1.6vw, 18px); 
            color:white;
        ">
            The upper portion of the Kali Babon Watershed, located immediately upstream of 
            <b>Perumahan Dinar Indah</b> (−7.038325, 110.436617), was selected as the primary focus of this 
            hydrological analysis due to its critical influence on downstream flood dynamics and its 
            documented history of recurrent flooding. This upstream segment functions as the main 
            contributing area to runoff accumulation that directly affects Perumahan Dinar Indah, 
            one of the most flood-prone residential zones within the watershed. Between 2020 and 2023, 
            multiple flood events were recorded in this area, with the most severe occurring on 
            <b>6 January 2023</b>, when the <b>Kali Babon River embankment experienced a levee breach</b>, 
            leading to extensive inundation. On that date, daily rainfall across the upper watershed 
            ranged from <b>21.52 to 64.20 mm</b>, which falls within the <b>high to very high rainfall intensity</b> 
            categories based on standard hydrological classifications. Concurrently, twelve additional 
            locations downstream were also affected by flooding, illustrating the widespread hydrological 
            response to this extreme precipitation event. The selection of this area enables a focused 
            examination of upstream rainfall–runoff processes, watershed physiography, and their collective 
            role in amplifying flood hazards at critical downstream points such as Perumahan Dinar Indah.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div id="streamnetwork" style="
        max-width:90%; 
        margin:50px auto; 
        color:white;
    ">
        <h1 style="
            font-size:clamp(24px, 2.5vw, 32px); 
            color:white; 
            margin-bottom:20px;
        ">
            Stream Network Configuration of the Upper Kali Babon Watershed
        </h1>
        <div style="
            font-size:clamp(14px, 1.6vw, 18px); 
            line-height:1.6; 
            text-align:justify;
        ">
            The stream network delineation within the upper Kali Babon Watershed was conducted from the downstream reach—situated immediately before Perumahan Dinar Indah (−7.038325, 110.436617)—toward the upstream contributing areas. 
            This methodical approach provides a detailed understanding of how the upstream tributaries collectively influence the hydrological regime and flood behavior in the lower basin. 
            Two principal upstream branches, identified as the <strong>western</strong> and <strong>eastern segments</strong>, converge near <strong>−7.07677, 110.46459</strong>, forming a major hydrological junction that regulates runoff conveyance toward the downstream floodplain. 
            This confluence acts as a critical transition zone where multiple headwater tributaries integrate, shaping the overall discharge pattern and determining flow concentration toward Perumahan Dinar Indah and surrounding low-lying areas.
            <br><br>
            The <strong>western segment</strong> originates from two upper tributaries—designated as <em>West-1</em> and <em>East-1</em>—which merge at approximately <strong>−7.08217, 110.45413</strong>. 
            The <em>West-1 tributary</em> drains from the <strong>northwestern upland zone of Banyumanik</strong>, located at around <strong>300 meters above sea level</strong>. 
            This section is characterized by relatively steep slopes and elongated flow paths that enhance the velocity of surface runoff and contribute to elevated peak discharge during high-intensity rainfall events. 
            <br><br>
            The <em>East-1 tributary</em>, in contrast, is a composite headwater system formed by three upstream channels that converge at <strong>−7.09335, 110.45182</strong>. 
            These tributaries originate respectively from <strong>Susukan, West Ungaran</strong> (−7.10488, 110.42572), <strong>Curug Gending Asmara</strong> (−7.12331, 110.43080), and <strong>Desa Kalongan, Ungaran</strong> (−7.14859, 110.44057). 
            Each sub-catchment exhibits distinct topographic characteristics, influencing runoff concentration and flow response. 
            The confluence of these three tributaries forms the East-1 channel, which then joins with West-1 to establish the main western branch of the upper watershed. 
            This combined network regulates the routing of upstream flow and sediment transport toward the major confluence at <strong>−7.07677, 110.46459</strong>.
            <br><br>
            The <strong>eastern segment</strong> of the system originates from the <strong>Kalikayen</strong> sub-catchment, located near <strong>−7.09220, 110.46928</strong>. 
            This headwater area occupies an elevated terrain where orographic uplift enhances convective rainfall. 
            During the extreme precipitation event on <strong>6 January 2023</strong>, the Kalikayen sub-catchment recorded a <strong>daily rainfall intensity of 64.20 mm</strong>—the highest value observed across the watershed. 
            This high-intensity rainfall produced substantial runoff that propagated rapidly through the eastern tributary, merging with the western branch at the confluence and contributing to the severe downstream flooding at Perumahan Dinar Indah.
            <br><br>
            Overall, the delineated stream network reveals the complex hydrological connectivity of the upper Kali Babon Watershed, where multiple tributaries converge and interact with the terrain structure to form the main drainage corridor. Understanding this upstream configuration is essential for accurate runoff estimation, flow accumulation modeling, and identification of potential flood convergence zones along the downstream reach. You can observe these stream channels in the subsequent maps, particularly within the LULC, Slope, and Channel layers.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# LULC–Slope–River Overlay Map
# -----------------------------
st.subheader("LULC–Slope–River Overlay Map")

# -----------------------------
# Raster File Paths
# -----------------------------
@st.cache_data
def load_geojson(path):
    with open(path) as f:
        return json.load(f)
slope_tif = "data/slope_map.tif"
lulc_files = {
    "2020-09-15": "data/lulc_2020_styled.tif",
    "2021-09-30": "data/lulc_2021_styled.tif",
    "2022-09-15": "data/lulc_2022_styled.tif",
    "2023-09-20": "data/lulc_2023_styled.tif",
    "2025-08-15": "data/lulc_2025_styled.tif",
}
river_geojson = load_geojson("data/babon_channel.geojson")

# -----------------------------
# LULC Year Selection
# -----------------------------
available_years = sorted({int(d.split("-")[0]) for d in lulc_files.keys()})
selected_year = st.select_slider(
    "🗓️ Select LULC Year",
    options=available_years,
    value=available_years[-1]
)
selected_date = next(d for d in lulc_files.keys() if int(d.split("-")[0]) == selected_year)
lulc_tif = lulc_files[selected_date]
st.markdown(f"<p style='text-align:center;'>Selected LULC date: <b>{selected_date}</b></p>", unsafe_allow_html=True)

# -----------------------------
# Layer Checkboxes
# -----------------------------
show_slope = st.checkbox("Show Slope Map", value=True)
show_lulc = st.checkbox("Show LULC Map", value=True)
show_river = st.checkbox("Show River Network", value=True)

# -----------------------------
# Initialize Folium Map
# -----------------------------
m = folium.Map(location=[-7.1, 110.45], zoom_start=12, tiles='Cartodb Positron')

# -----------------------------
# Raster-to-PNG conversion with caching
# -----------------------------
@st.cache_data(show_spinner=False)
def geotiff_to_temp_png(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        # Multi-band RGBA
        if src.count >= 4:
            arr = src.read([1,2,3,4])
            arr = np.transpose(arr, (1,2,0))
            img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        elif src.count == 3:
            arr = src.read([1,2,3])
            arr = np.transpose(arr, (1,2,0))
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
        else:
            arr = src.read(1)
            arr = np.where(arr == src.nodata, 0, arr)
            arr = ((arr - arr.min()) / (arr.max()-arr.min()) * 255).astype(np.uint8)
            img = Image.fromarray(arr).convert("RGBA")
            # make zero values transparent
            datas = img.getdata()
            new_data = [(0,0,0,0) if item[0]==0 else item for item in datas]
            img.putdata(new_data)

        # Save to temporary PNG
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp_file.name)
        return tmp_file.name, bounds

# -----------------------------
# Add Selected Layers
# -----------------------------
temp_files = []  # keep track of temp files to clean up
if show_slope:
    png_file, bounds = geotiff_to_temp_png(slope_tif)
    temp_files.append(png_file)
    ImageOverlay(
        name="Slope Map",
        image=png_file,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

if show_lulc:
    png_file, bounds = geotiff_to_temp_png(lulc_tif)
    temp_files.append(png_file)
    ImageOverlay(
        name=f"LULC {selected_date}",
        image=png_file,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=2
    ).add_to(m)

if show_river:
    folium.GeoJson(
        river_geojson,
        name="River Network",
        style_function=lambda feature: {
            "color": "blue",
            "weight": 2,
            "opacity": 0.8
        }
    ).add_to(m)

# -----------------------------
# Display Folium Map
# -----------------------------
st_folium(m,width=None, height=600)

# -----------------------------
# Optional: Clean up temp PNGs (comment out if debugging)
# -----------------------------
#for f in temp_files:
#    os.remove(f)

# -----------------------------
# Description Tables (Slope & LULC)
# -----------------------------
# Slope Table
slope_data = {
    "Class": [1,2,3,4,5,6],
    "Slope Range (°)": ["0–2", ">2–5", ">5–8", ">8–15", ">15–25", ">25–33"],
    "Description": ["Flat","Gently Undulating","Gentle Slope","Moderate Slope","Steep","Very Steep"],
    "Color": ["white","honeydew","lightgreen","mediumseagreen","seagreen","darkgreen"]
}
df_slope = pd.DataFrame(slope_data)
df_slope["Color"] = df_slope["Color"].apply(
    lambda c: f'<div style="width:25px;height:25px;background-color:{c};border-radius:4px;border:1px solid #ccc;"></div>'
)

# LULC Tables
df_lanina = pd.DataFrame({
    "No": list(range(1,8)),
    "Class Name": ["Moist Evergreen Forest","Seasonally Moist Forest","Grassland","Agriculture","Bare Soil","Urban Fabric","Road / Pavement"],
    "Description": ["Dense, year-round green canopy","Forest showing slight canopy lightening","Herbaceous cover, green and active in wet season",
                    "Cultivated area (actively cropped)","Exposed mineral surface, no vegetation","Built-up settlement area","Impervious linear surface"],
    "Color": ["darkgreen","seagreen","lightgreen","yellow","tan","firebrick","gray"]
})
df_elnino = pd.DataFrame({
    "No": list(range(1,8)),
    "Class Name": ["Moist Evergreen Forest","Seasonally Dry Forest","Grassland (Dry)","Agriculture","Bare Soil","Urban Fabric","Road / Pavement"],
    "Description": ["Still green, minimal stress","Forest turning brownish or lighter canopy","Yellowish herbaceous cover, partly senescent",
                    "Actively irrigated or resilient cropland","Completely exposed earth","Built-up area","Impervious linear features"],
    "Color": ["darkgreen","saddlebrown","beige","yellow","tan","firebrick","gray"]
})
df_lanina["Color"] = df_lanina["Color"].apply(
    lambda c: f'<div style="width:25px;height:25px;background-color:{c};border-radius:4px;border:1px solid #ccc;"></div>'
)
df_elnino["Color"] = df_elnino["Color"].apply(
    lambda c: f'<div style="width:25px;height:25px;background-color:{c};border-radius:4px;border:1px solid #ccc;"></div>'
)

# LULC phase mapping
lulc_phase = {
    "2020-09-15": "Normal",
    "2021-09-30": "Normal",
    "2022-09-15": "Normal",
    "2023-09-20": "El Niño",
    "2025-08-15": "Normal",
}

# Table selection
table_choice = st.selectbox("Select Table Description", ["Slope", "LULC"], index=0)

# Display table
if table_choice == "Slope":
    st.subheader("🌾 Slope Classification (0–33°)")
    st.write(df_slope.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    phase = lulc_phase[selected_date]
    st.subheader(f"🌎 LULC Description – {phase} Phase ({selected_date})")
    st.write(df_lanina.to_html(escape=False, index=False) if phase=="Normal" else df_elnino.to_html(escape=False, index=False), unsafe_allow_html=True)
st.markdown(
    """
    <div id="lulcstatus" style="margin:5vw auto; max-width:90%; line-height:1.6; text-align:justify; color:white;">
        <h1 style="font-size:clamp(22px, 2.2vw, 32px); margin-bottom:1rem;">
            Land Cover Dynamics and Indications of Elevated Runoff in the Upper Kali Babon Watershed
        </h1>
        <div style="font-size:clamp(14px, 1.5vw, 18px); line-height:1.8;">
            The upper Kali Babon Watershed exhibits a diverse land cover composition consisting of forest, grassland, agricultural land, bare soil, and urban built-up areas. 
            This heterogeneous landscape reflects both natural physiographic variation and human land use, shaping how the watershed responds hydrologically under different climatic conditions.
            <br><br>
            During the <strong>La Niña years (2020–2022)</strong>, above-average rainfall maintained dense vegetation cover across forest, grassland, and agricultural zones. 
            Soil moisture remained high, resulting in vigorous plant growth and minimal exposure of bare soil. 
            In contrast, the <strong>El Niño year (2023)</strong> brought significantly reduced rainfall, causing a decline in vegetation greenness. 
            Portions of forest became seasonally dry, while grassland and agricultural areas exhibited browning and partial exposure of bare soil. 
            This transformation indicates that although La Niña delivered abundant rainfall, much of the water was rapidly converted to surface runoff rather than stored in the soil, a pattern reinforced by steep slopes and limited infiltration capacity.
            <br><br>
            In the <strong>western segment</strong>, the <em>West-1 tributary</em> originates from the <strong>Banyumanik upland</strong>, where urban built-up areas occupy relatively flat highlands. 
            These impervious surfaces promote rapid overland flow and reduce infiltration potential. 
            Downstream, the terrain transitions into steeper slopes dominated by forest and grassland, which direct lateral flow toward the main channel. 
            The adjacent <em>East-1 tributary</em>, which drains the same western system, lies on steep slopes where runoff intensification was most evident during La Niña, indicating that heavy rainfall over shallow or compacted soils primarily enhances surface flow.
            <br><br>
            The <strong>eastern segment</strong>, represented by the <strong>Kalikayen sub-catchment</strong>, is mainly covered by grassland and agriculture distributed along moderate to steep slopes. 
            During wet years, these areas remain vegetatively active, maintaining strong surface connectivity to the main stream. 
            However, under El Niño-driven dryness, soil compaction and reduced vegetation cover lower infiltration capacity, increasing susceptibility to surface runoff once rainfall resumes.
            <br><br>
            Taken together, the arrangement of <strong>urbanized uplands in West-1</strong>, <strong>steep forest–grassland slopes in East-1</strong>, and <strong>agro-grassland areas in Kalikayen</strong> explains the watershed’s generally high runoff potential. 
            These land cover transitions and climatic responses demonstrate a limited ability of the landscape to retain rainfall inputs, leading to saturation-driven flooding during wet years and infiltration-limited runoff after dry periods.
            <br><br>
            You can further explore these spatial dynamics through the <strong>interactive map</strong>. 
            Try checking or unchecking the map layers to compare <strong>slope</strong>, <strong>land-use/land-cover (LULC)</strong>, and <strong>river channels</strong>. 
            Use the <strong>year slider</strong> to observe how vegetation and urban areas evolved between 2020 and 2023, and refer to the <strong>dropdown menu</strong> to view classification tables or layer descriptions relevant to each map.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Runoff Potential Index & Rainfall / Runoff Map
# -----------------------------
st.subheader("Runoff Potential Index & Rainfall / Runoff Map")

# -----------------------------
# Raster File Paths
# -----------------------------
raster_files = {
    "RPI": "data/RPI_2025.tif",
    "Rainfall": "data/rainfall_babon.tif",
    "Q": "data/Q_map.tif"
}

# -----------------------------
# Layer Checkboxes
# -----------------------------
show_rpi = st.checkbox("Show RPI Map", value=True)
show_rainfall = st.checkbox("Show Rainfall Map", value=True)
show_q = st.checkbox("Show Q Map", value=True)

# -----------------------------
# Initialize Folium Map
# -----------------------------
m_rpi = folium.Map(location=[-7.1, 110.45], zoom_start=12, tiles='Cartodb Positron')

# -----------------------------
# Cached raster → PNG converter
# -----------------------------
@st.cache_data(show_spinner=False)
def geotiff_to_temp_png(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        if src.count >= 4:
            arr = src.read([1,2,3,4])
            arr = np.transpose(arr, (1,2,0))
            img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        elif src.count == 3:
            arr = src.read([1,2,3])
            arr = np.transpose(arr, (1,2,0))
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
        else:
            arr = src.read(1)
            arr = np.where(arr == src.nodata, 0, arr)
            arr = ((arr - arr.min()) / (arr.max()-arr.min()) * 255).astype(np.uint8)
            img = Image.fromarray(arr).convert("RGBA")
            datas = img.getdata()
            new_data = [(0,0,0,0) if item[0]==0 else item for item in datas]
            img.putdata(new_data)
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp_file.name)
        return tmp_file.name, bounds

# -----------------------------
# Add selected layers
# -----------------------------
temp_files = []
if show_rpi:
    png_file, bounds = geotiff_to_temp_png(raster_files["RPI"])
    temp_files.append(png_file)
    ImageOverlay(
        name="RPI (Rendered)",
        image=png_file,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=1.0,
        interactive=True
    ).add_to(m_rpi)

if show_rainfall:
    png_file, bounds = geotiff_to_temp_png(raster_files["Rainfall"])
    temp_files.append(png_file)
    ImageOverlay(
        name="Rainfall (Interpolated)",
        image=png_file,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.95,
        interactive=True
    ).add_to(m_rpi)

if show_q:
    png_file, bounds = geotiff_to_temp_png(raster_files["Q"])
    temp_files.append(png_file)
    ImageOverlay(
        name="Q (Runoff)",
        image=png_file,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.95,
        interactive=True
    ).add_to(m_rpi)

# -----------------------------
# Display Folium map
# -----------------------------
st_folium(m_rpi, width=None, height=600)

# -----------------------------
# Dropdown to select legend type
# -----------------------------
legend_option = st.selectbox(
    "Select Legend / Description",
    ["Rainfall", "Discharge (Q)", "Runoff Potential Index (RPI)"]
)

# -----------------------------
# Rainfall Gradient Legend
# -----------------------------
if legend_option == "Rainfall":
    st.subheader("Rainfall Gradient Legend")
    min_val = 21.5194
    max_val = 64.2044
    gradient_html = f"""
    <div style="display:flex; align-items:center;">
      <span style="margin-right:8px;">{min_val:.2f}</span>
      <div style="
          height: 25px;
          flex-grow: 1;
          background: linear-gradient(to right, lightblue, darkblue);
          border: 1px solid #ccc;
          border-radius: 4px;">
      </div>
      <span style="margin-left:8px;">{max_val:.2f}</span>
    </div>
    """
    st.markdown(gradient_html, unsafe_allow_html=True)

# -----------------------------
# Q Data Description Table
# -----------------------------
elif legend_option == "Discharge (Q)":
    st.subheader("Q Data Description Table")
    df_q = pd.DataFrame({
        "Range (mm)": ["2.98 – 9.34", "9.34 – 12.26", "12.26 – 17.03", "17.03 – 40.60", "> 40.60"],
        "Description": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "Color": ["#f7fbff", "#d7e6f5", "#c8ddf0", "#a3cce3", "#08306b"]
    })
    def color_cells(val):
        return f'background-color: {val}; color: {"white" if val == "#08306b" else "black"}; text-align:center;'
    styled_df = df_q.style.applymap(color_cells, subset=["Color"])
    st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# -----------------------------
# RPI Classification Table
# -----------------------------
elif legend_option == "Runoff Potential Index (RPI)":
    st.subheader("🧭 Runoff Potential Index (RPI) Classification")
    df_rpi = pd.DataFrame({
        "Class": ["1", "2", "3", "4", "5"],
        "RPI Range": ["0.12–0.25", "0.25–0.38", "0.38–0.51", "0.51–0.64", "0.64–0.72"],
        "Color": ["aliceblue", "lightskyblue", "deepskyblue", "royalblue", "navy"],
        "Runoff Potential": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "Description": [
            "Dominated by dense vegetation or permeable soils. High infiltration capacity; low imperviousness and minimal surface runoff.",
            "Mostly pervious cover such as cropland and grassland with gentle slopes. Moderate infiltration; limited direct runoff.",
            "Mixed surfaces or transitional land uses. Balanced infiltration and surface runoff; moderately impervious areas.",
            "Predominantly compacted or semi-impervious surfaces (urban fringe, infrastructure). Reduced infiltration; higher direct runoff response.",
            "Highly impervious or steep terrain (urban core, paved surfaces). Very low infiltration and rapid surface runoff generation.",
        ],
    })
    df_rpi["Color"] = df_rpi["Color"].apply(lambda c: f'<div style="width:25px; height:25px; background-color:{c}; border-radius:4px; border:1px solid #ccc;"></div>')
    st.write(df_rpi.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.caption("Source: Dhakal, N. (2019). *Development of Guidance for Runoff Coefficient Selection and Modified Rational Unit Hydrograph Method for Hydrologic Design.*")
st.markdown(
    """
    <div id="rpi_q" style="margin:5vw auto; max-width:90%; line-height:1.6; text-align:justify; color:white;">
        <h1 style="font-size:clamp(22px, 2.2vw, 32px); text-align:left; margin-bottom:20px;">
            Runoff Potential Index (RPI), Rainfall, and Discharge (Q) Analysis
        </h1>
        <div style="font-size:clamp(14px, 1.5vw, 18px); line-height:1.8;">
            <strong>Runoff Potential Index (RPI)</strong><br>
            The RPI was developed to estimate spatial variations in runoff potential across the upper Kali Babon Watershed. 
            The analysis integrates three key factors: land cover (LULC 2025), slope, and flow accumulation. 
            The runoff coefficient (C) for each LULC class was derived based on <strong>Dhakal (2019)</strong>, slope values were normalized from the 8 × 8 m DEMNAS data, and flow accumulation was obtained from the r.watershed plugin. 
            A weighted overlay was applied with <strong>w_LULC = 0.5</strong>, <strong>w_slope = 0.3</strong>, and <strong>w_accumulation = 0.2</strong> to generate the RPI map.
            <br><br>
            The resulting RPI distribution indicates that the <strong>West-1 Segment</strong> ranges from moderate to very high, the <strong>East-1 Segment</strong> is predominantly low to moderate, and the <strong>East Segment</strong> exhibits very low to low runoff potential.
            <br><br>
            <strong>Rainfall and Discharge (Q)</strong><br>
            Rainfall data from all climate stations were collected for the extreme flood event on <strong>6 January 2023</strong>, when daily precipitation ranged from 21.52 mm to 64.20 mm. 
            These data were spatially interpolated to create a continuous rainfall surface, revealing that the highest rainfall occurred in the <strong>East Segment</strong>, while lower intensities were observed toward the southwest.
            <br><br>
            By combining the interpolated rainfall map with the RPI, a <strong>discharge (Q) map</strong> was generated to estimate spatial runoff magnitudes and identify flood-prone zones. 
            Zonal statistics for half-basin segments quantified flow accumulation and peak discharge, demonstrating how the interaction of land cover, slope, and upstream contributing areas governs downstream hydrological response.
            <br><br>
            The analysis identifies the <strong>West-1 Segment</strong> and the <strong>East Segment</strong> as the dominant contributors to downstream flooding. 
            The West-1 Segment is mostly urbanized on a flat, high-elevation plateau, with only small areas of steep slopes covered by forest and grassland that locally enhance runoff. 
            The East Segment, in contrast, contributes significant flow due to the extreme rainfall (>60 mm/day) it received during the event. 
            This combination of high-intensity rainfall in the East segment and localized runoff from steeper portions of West-1 explains the major flooding observed at Perumahan Dinar Indah. 
            The East-1 Segment shows lower discharge due to moderate slopes and land cover that promotes infiltration.
            <br><br>
            <em style="color:#c2e5ff;">
            Users can interact with the map above by selecting or deselecting the <strong>RPI</strong>, <strong>Rainfall</strong>, and <strong>Discharge (Q)</strong> layers using the checklist. 
            Dropdown menus allow switching between descriptive classifications and hydrological interpretations for each layer, providing a dynamic and interactive way to explore watershed behavior and flood vulnerability.
            </em>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# ----------------------------
# Opacity Slider
# ----------------------------
st.subheader("Adjust Half-Basin Opacity")
basin_opacity = st.slider("Half-Basin Opacity", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# ----------------------------
# Load Styled GeoJSON
# ----------------------------
geojson_path = "data/Runoff_statistic_styled.geojson"
try:
    with open(geojson_path) as f:
        basins_geojson = json.load(f)
except FileNotFoundError:
    st.error(f"❌ Styled GeoJSON not found: {geojson_path}")
    st.stop()

# ----------------------------
# Apply Dynamic Opacity
# ----------------------------
for feature in basins_geojson['features']:
    if 'style' in feature['properties']:
        feature['properties']['style']['fillOpacity'] = basin_opacity

# ----------------------------
# Create Folium Map
# ----------------------------
st.subheader("Half-Basin Map (Mean Q)")
m = folium.Map(location=[-7.1, 110.45], zoom_start=12, tiles='OpenStreetMap')

# Define tooltip with all statistics
tooltip = folium.GeoJsonTooltip(
    fields=['mean_Q', 'med_Q', 'max_Q', 'min_Q', 'sum_Q', 'std_Q'],
    aliases=['Mean Q', 'Median Q', 'Max Q', 'Min Q', 'Sum Q', 'Std Dev Q'],
    localize=True,
    sticky=False,
    labels=True,
    style="""
        background-color: white;
        border: 1px solid black;
        border-radius: 3px;
        padding: 5px;
    """
)

# Add GeoJSON layer with style and full tooltip
folium.GeoJson(
    basins_geojson,
    name="Half-Basin Mean Q",
    style_function=lambda feature: feature['properties']['style'],
    tooltip=tooltip  # ← use the full tooltip object here
).add_to(m)

# Render map
st_folium(m,width=None, height=600)

st.markdown(
    """
    <div id="q_analysis" style="margin:5vw auto; max-width:90%; line-height:1.8; color:white;">
        <h1 style="
            font-size:clamp(22px, 2.2vw, 32px); 
            text-align:center; 
            margin-bottom:15px;
        ">
            Runoff (Q) Analysis by Half-Basin
        </h1>
        <div style="
            font-size:clamp(14px, 1.5vw, 18px); 
            line-height:1.8; 
            text-align:justify;
        ">
            The runoff (Q) analysis was performed by aggregating flow estimates for each <strong>half-basin segment</strong>, providing a detailed view of sub-catchment contributions to downstream flooding. The <strong>mean Q values</strong> per half-basin highlight spatial differences in runoff response:
            <br><br>
            <strong>West-1 Segment:</strong> Mean Q ranges from 12 to 17. Mostly urbanized on a flat plateau, with small areas of steeper slopes covered by forest and grassland that locally enhance runoff.
            <br>
            <strong>East Segment:</strong> Mean Q ranges from 11 to 15. Receives the highest rainfall (>60 mm/day) but runoff is moderated by gentle slopes and mixed agricultural/grassland cover.
            <br>
            <strong>East-1 Segment:</strong> Mean Q is approximately 10–11, reflecting lower runoff due to moderate slopes and forested patches promoting infiltration.
            <br><br>
            These results illustrate that <strong>flood risk is influenced by rainfall intensity, physiography, and land cover</strong>. 
            <br><br>
            On the interactive map:
            <ul>
                <li>Hover over each half-basin to view detailed statistics and notes for that segment.</li>
                <li>Use the opacity slider to adjust the half-basin transparency, making it easier to observe underlying geographic features and segment names.</li>
            </ul>
            This approach allows users to dynamically explore which areas dominate runoff production and how different segments interact to influence downstream flooding, supporting data-informed management decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div id="methodology" style="margin-top:50px; font-size:40px; font-weight:bold; text-align:left; color:white;">
        Methodology
    </div>

    <div style="font-size:16px; color:white; text-align:justify; margin-top:15px;">
        This study integrates <strong>data collection, processing, hydrological analysis, and mapping</strong>
        to assess flood hazards and runoff dynamics within the Kali Babon Watershed. Flood incident data 
        (2020–2023) were obtained from <em>BPBD Jawa Tengah</em>, while rainfall data (2021–2023) were 
        sourced from PUSDATARU Jawa Tengah and BBWS Pemali Juana, then extracted from PDF format using 
        a dedicated Python script.
    </div>

    <div style="font-size:16px; color:white; text-align:justify; margin-top:15px;">
        <strong>DEM data</strong> from <em>DEMNAS</em> with 8×8 m resolution were used to derive 
        <strong>elevation, slope, and watershed boundaries</strong>. The DEM was first hydrologically 
        corrected using <strong>r.fill</strong> to remove sinks, then <strong>r.watershed</strong> 
        was applied to calculate flow accumulation.
    </div>

    <div style="font-size:16px; color:white; text-align:justify; margin-top:15px;">
        <strong>LULC data</strong> were obtained from <em>Sentinel-2 Copernicus imagery</em> and 
        classified using the <strong>Semi-Automatic Classification Plugin (SCP) in QGIS</strong> 
        with a Random Forest classifier of 400 trees and balanced class weights to ensure accurate 
        land cover mapping.
    </div>

    <div style="font-size:16px; color:white; text-align:justify; margin-top:15px;">
        The <strong>Runoff Potential Index (RPI)</strong> was computed by integrating LULC-derived 
        runoff coefficients (C, according to Dhakal, 2019), normalized slope, and flow accumulation, 
        using weights of 0.5 (LULC), 0.3 (slope), and 0.2 (accumulation). RPI values were standardized 
        and classified into minimum–maximum ranges to evaluate runoff potential for each segment.
    </div>

    <div style="font-size:16px; color:white; text-align:justify; margin-top:15px;">
        <strong>Runoff (Q)</strong> was estimated using the RPI and interpolated rainfall data, 
        followed by a half-basin analysis to summarize mean Q per sub-catchment, providing insight 
        into the contribution of upstream segments to downstream flood risk.
    </div>

    <div style="font-size:16px; color:white; text-align:justify; margin-top:15px;">
        All maps—including RPI, rainfall intensity, slope, and runoff—were standardized according 
        to hydrological theory and conventional classification ranges (e.g., very low to very high), 
        enabling consistent interpretation of flood hazards and runoff dynamics.
    </div>

    <div style="font-size:16px; color:white; text-align:justify; margin-top:15px;">
        This integrated workflow provides a comprehensive framework combining <strong>observational 
        data, satellite imagery classification, hydrologically corrected DEM analysis, spatial processing, 
        hydrological modeling, and standardized mapping</strong> to support robust understanding of 
        flood behavior and runoff characteristics across the Kali Babon Watershed.
    </div>
    """,
    unsafe_allow_html=True
)





# Sidebar navigation
st.sidebar.markdown(
    """
    <div style="color:white; font-size:1rem; font-family:sans-serif;">
        <h3 style="margin-bottom:15px; border-bottom:1px solid white; padding-bottom:5px;">Navigation</h3>
        <ul style="list-style-type:none; padding-left:0; line-height:2;">
            <li>
                <a href="#flood_overview" style="
                    color:white; 
                    text-decoration:none; 
                    display:block; 
                    padding:5px 0; 
                    transition:0.3s;
                " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='white'">
                    Flood Incident, Rainfall Chart and Climate Station Location
                </a>
            </li>
            <li>
                <a href="#kalibabon-section" style="
                    color:white; 
                    text-decoration:none; 
                    display:block; 
                    padding:5px 0; 
                    transition:0.3s;
                " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='white'">
                    Kali Babon Watershed Analysis
                </a>
            </li>
            <li>
                <a href="#streamnetwork" style="
                    color:white; 
                    text-decoration:none; 
                    display:block; 
                    padding:5px 0; 
                    transition:0.3s;
                " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='white'">
                    Stream Network Configuration
                </a>
            </li>
            <li>
                <a href="#lulcstatus" style="
                    color:white; 
                    text-decoration:none; 
                    display:block; 
                    padding:5px 0; 
                    transition:0.3s;
                " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='white'">
                    LULC Status and Slope
                </a>
            </li>
            <li>
                <a href="#rpi_q" style="
                    color:white; 
                    text-decoration:none; 
                    display:block; 
                    padding:5px 0; 
                    transition:0.3s;
                " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='white'">
                    RPI, Rainfall, and Runoff (Q)
                </a>
            </li>
            <li>
                <a href="#q_analysis" style="
                    color:white; 
                    text-decoration:none; 
                    display:block; 
                    padding:5px 0; 
                    transition:0.3s;
                " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='white'">
                    Half-Basin Runoff (Q) Statistics
                </a>
            </li>
            <li>
                <a href="#methodology" style="
                    color:white; 
                    text-decoration:none; 
                    display:block; 
                    padding:5px 0; 
                    font-weight:bold;
                    transition:0.3s;
                " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='white'">
                    Methodology
                </a>
            </li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)



# Separator
st.sidebar.markdown('<hr style="margin-top:50px; margin-bottom:50px;">', unsafe_allow_html=True)
st.sidebar.markdown("## 📊 Dashboard Metrics", unsafe_allow_html=True)
st.sidebar.metric("Flood Incidents", len(flood_map_df))
st.sidebar.metric("Stations", len(station_map_df)-1)