import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import numpy as np

st.set_page_config(page_title="한반도 실시간 비행기 추적", layout="wide")

st.title("✈️ 한반도 상공 실시간 비행기 이상 탐지 웹앱")
st.write("OpenSky API 데이터에 Z-score 통계 기법을 적용하여 급강하 중인 비행기를 자동으로 감지합니다.")

# -----------------------------------------------------------
# 0. OAuth2 인증 정보 설정 (★여기를 수정하세요★)
# -----------------------------------------------------------
TOKEN_URL = "https://opensky-network.org/oauth/token"
# 본인의 실제 Client ID와 Secret 값을 아래에 정확히 넣어주세요.
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"] # 내부에 따옴표가 있는 문자열일 수 있으므로 겉은 작은따옴표(')로 감싸는 것이 안전합니다.

# -----------------------------------------------------------
# 1. 사이드바 UI 설정
# -----------------------------------------------------------
st.sidebar.header("⚙️ 컨트롤 타워")
refresh_button = st.sidebar.button("🔄 실시간 데이터 새로고침")

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 이상 탐지 설정")
z_threshold = st.sidebar.slider(
    "급강하 감지 Z-score 기준값",
    min_value=-5.0,
    max_value=5.0,
    value=-3.0,
    step=0.1
)

# -----------------------------------------------------------
# 2. 토큰 발급 및 데이터 수집 (OAuth2 적용)
# -----------------------------------------------------------

# [핵심] 토큰 캐싱: 50분(3000초) 동안은 기존 토큰을 재사용하여 통신 부하를 줄입니다.
@st.cache_data(ttl=3000)
def get_oauth2_token(client_id, client_secret):
    try:
        data = {'grant_type': 'client_credentials'}
        auth = (client_id, client_secret)
        
        # 토큰 서버로 요청을 보냅니다.
        response = requests.post(TOKEN_URL, data=data, auth=auth, timeout=15)
        response.raise_for_status()
        
        return response.json().get('access_token')
    except Exception as e:
        st.error(f"토큰 발급 실패 (인증 정보를 확인하세요): {e}")
        return None

# [핵심] 발급받은 토큰을 활용하여 데이터 요청
def get_flight_data(token):
    if not token:
        return []
        
    url = "https://opensky-network.org/api/states/all"
    params = {"lamin": 33.0, "lamax": 39.0, "lomin": 124.0, "lomax": 132.0}
    
    # Bearer 방식으로 헤더에 토큰을 심어서 보냅니다. (이것이 봇 차단을 우회하는 열쇠입니다)
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status() # 401(권한 없음) 등의 에러가 나면 여기서 걸러집니다.
        
        data = response.json()
        if data is not None and data.get("states") is not None:
            return data["states"]
        return []
    except Exception as e:
        st.error(f"비행기 데이터 조회 중 오류가 발생했습니다: {e}")
        return []

# 토큰을 먼저 받아온 뒤, 그 토큰으로 비행기 데이터를 조회합니다.
access_token = get_oauth2_token(CLIENT_ID, CLIENT_SECRET)
raw_data = get_flight_data(access_token)

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
    
    df = df[['callsign', 'longitude', 'latitude', 'baro_altitude', 'velocity', 'vertical_rate']]
    df = df.dropna(subset=['longitude', 'latitude', 'vertical_rate'])
    df['callsign'] = df['callsign'].astype(str).str.strip().replace('', '알 수 없음')

    mean_vr = df['vertical_rate'].mean()
    std_vr = df['vertical_rate'].std()
    
    if std_vr > 0:
        df['z_score'] = (df['vertical_rate'] - mean_vr) / std_vr
    else:
        df['z_score'] = 0.0

    df['status'] = df['z_score'].apply(lambda z: '위험(급강하)' if z <= z_threshold else '정상')

    def assign_color(status):
        if status == '위험(급강하)':
            return [255, 0, 0, 255]
        return [255, 200, 0, 180]
        
    df['color'] = df['status'].apply(assign_color)

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

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_radius=6000,
        get_fill_color="color",
        pickable=True
    )

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
    st.warning("현재 한반도 상공에서 감지된 비행기 데이터가 없거나 서버와 연결할 수 없습니다.")
