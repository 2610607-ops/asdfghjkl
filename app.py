import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import numpy as np

st.set_page_config(page_title="한반도 실시간 비행기 추적", layout="wide")

st.title("✈️ 한반도 상공 실시간 비행기 이상 탐지 웹앱")
st.write("OpenSky API 데이터에 Z-score 통계 기법을 적용하여 급강하 중인 비행기를 자동으로 감지합니다.")

# -----------------------------------------------------------
# 1. 사이드바 UI 설정 (슬라이더 추가)
# -----------------------------------------------------------
st.sidebar.header("⚙️ 컨트롤 타워")
refresh_button = st.sidebar.button("🔄 실시간 데이터 새로고침")

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 이상 탐지(Anomaly Detection) 설정")

# 사용자가 직접 Z-score 기준값을 조절할 수 있는 슬라이더를 만듭니다.
# 기본값은 통계학적 기준인 -3.0으로 설정합니다.
z_threshold = st.sidebar.slider(
    "급강하 감지 Z-score 기준값",
    min_value=-5.0,
    max_value=5.0,
    value=-3.0,
    step=0.1
)

# -----------------------------------------------------------
# 2. 데이터 수집 (OpenSky API)
# -----------------------------------------------------------
def get_flight_data():
    url = "https://opensky-network.org/api/states/all&quot;
    params = {"lamin": 33.0, "lamax": 39.0, "lomin": 124.0, "lomax": 132.0}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data is not None and data.get("states") is not None:
            return data["states"]
        return []
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return []

raw_data = get_flight_data()

# -----------------------------------------------------------
# 3. 데이터 전처리 및 Z-score 계산 (Pandas)
# -----------------------------------------------------------
if len(raw_data) > 0:
    columns = [
        'icao24', 'callsign', 'origin_country', 'time_position', 'last_contact',
        'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity',
        'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'
    ]
    df = pd.DataFrame(raw_data, columns=columns)
   
    # [수정] 수직 속도(vertical_rate)를 데이터 분석 대상에 포함시킵니다.
    df = df[['callsign', 'longitude', 'latitude', 'baro_altitude', 'velocity', 'vertical_rate']]
   
    # 위치 정보와 수직 속도가 없는 데이터는 지워줍니다.
    df = df.dropna(subset=['longitude', 'latitude', 'vertical_rate'])
    df['callsign'] = df['callsign'].astype(str).str.strip().replace('', '알 수 없음')

    # --- [핵심 기능] Z-score 계산 ---
    # 현재 한반도 상공 모든 비행기의 수직 속도 평균(mean)과 표준편차(std)를 구합니다.
    mean_vr = df['vertical_rate'].mean()
    std_vr = df['vertical_rate'].std()
   
    # 만약 비행기가 너무 적어서 표준편차가 0이 되는 경우를 대비한 안전 장치입니다.
    if std_vr > 0:
        df['z_score'] = (df['vertical_rate'] - mean_vr) / std_vr
    else:
        df['z_score'] = 0.0

    # 사용자가 설정한 슬라이더 기준값(z_threshold) 이하이면 '위험(급강하)', 아니면 '정상'으로 분류합니다.
    df['status'] = df['z_score'].apply(lambda z: '위험(급강하)' if z <= z_threshold else '정상')

    # --- [시각화 꿀팁] 상태에 따른 색상 부여 ---
    # 정상 비행기는 노란색[255, 200, 0], 위험 비행기는 빨간색[255, 0, 0]으로 지정합니다.
    def assign_color(status):
        if status == '위험(급강하)':
            return [255, 0, 0, 255] # 빨간색 (R, G, B, A)
        return [255, 200, 0, 180]    # 노란색
       
    df['color'] = df['status'].apply(assign_color)

    # 대시보드 요약 정보 표시
    diving_count = len(df[df['status'] == '위험(급강하)'])
    st.sidebar.success(f"현재 추적 비행기: {len(df)}대")
    if diving_count > 0:
        st.sidebar.error(f"⚠️ 급강하 감지: {diving_count}대!!")
    else:
        st.sidebar.info("✅ 현재 특이 이상 징후 없음")

    # -----------------------------------------------------------
    # 4. Pydeck 3D 지도 시각화
    # -----------------------------------------------------------
    view_state = pdk.ViewState(latitude=36.0, longitude=128.0, zoom=6, pitch=45)

    # [수정] get_fill_color에 고정된 값이 아닌, 위에서 우리가 만든 'color' 컬럼을 연동합니다.
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_radius=6000,
        get_fill_color="color",
        pickable=True
    )

    # 툴팁에 Z-score와 현재 상태, 수직속도 정보를 추가하여 사용자가 확인할 수 있게 합니다.
    tooltip = {
        "html": """
        <b>콜사인:</b> {callsign} <br/>
        <b>상태:</b> {status} <br/>
        <b>수직 속도:</b> {vertical_rate} m/s <br/>
        <b>Z-score:</b> {z_score} <br/>
        <b>현재 고도:</b> {baro_altitude} m
        """,
        "style": {"backgroundColor": "black", "color": "white"}
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="dark"
    )

    st.pydeck_chart(r)
   
    # -----------------------------------------------------------
    # 5. 데이터 테이블 확인
    # -----------------------------------------------------------
    st.subheader("📊 실시간 항공 통계 및 데이터")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="평균 수직 속도", value=f"{mean_vr:.2f} m/s")
    with col2:
        st.metric(label="수직 속도 표준편차", value=f"{std_vr:.2f}")
       
    st.dataframe(df[['callsign', 'status', 'z_score', 'vertical_rate', 'baro_altitude', 'velocity']])
else:
    st.warning("현재 한반도 상공에서 감지된 비행기 데이터가 없습니다. (잠시 후 다시 시도해보세요)")
