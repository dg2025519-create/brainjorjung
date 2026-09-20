import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# 페이지 기본 설정
# -------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 데이터 탐색")
st.markdown("---")

# -------------------------------
# 데이터 불러오기
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# -------------------------------
# 1. 나이와 평균 혈당 분포 히스토그램
# -------------------------------
st.header("1️⃣ 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.markdown("---")

# -------------------------------
# 2. 뇌졸중 여부에 따른 나이/혈당 상자그림 + 평균값 표
# -------------------------------
st.header("2️⃣ 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

df_box = df.copy()
df_box["뇌졸중 여부"] = df_box["stroke"].map({0: "겪지 않음", 1: "겪음"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df_box,
        x="뇌졸중 여부",
        y="age",
        title="뇌졸중 여부별 나이 상자그림",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df_box,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        title="뇌졸중 여부별 평균 혈당 상자그림",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

mean_table = df_box.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]].mean().round(2)
mean_table.columns = ["나이 평균", "평균 혈당 평균"]
st.subheader("📊 그룹별 평균값")
st.dataframe(mean_table, use_container_width=True)

st.markdown("---")

# -------------------------------
# 3. 고혈압/심장병 여부에 따른 뇌졸중 비율 막대그래프
# -------------------------------
st.header("3️⃣ 고혈압·심장병 여부에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map({0: "없음", 1: "있음"})
    hyper_ratio["stroke"] = (hyper_ratio["stroke"] * 100).round(2)

    fig_hyper = px.bar(
        hyper_ratio,
        x="hypertension",
        y="stroke",
        title="고혈압 여부별 뇌졸중 비율(%)",
        labels={"hypertension": "고혈압 여부", "stroke": "뇌졸중 비율(%)"},
        text="stroke"
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "없음", 1: "있음"})
    heart_ratio["stroke"] = (heart_ratio["stroke"] * 100).round(2)

    fig_heart = px.bar(
        heart_ratio,
        x="heart_disease",
        y="stroke",
        title="심장병 여부별 뇌졸중 비율(%)",
        labels={"heart_disease": "심장병 여부", "stroke": "뇌졸중 비율(%)"},
        text="stroke"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.markdown("---")

# -------------------------------
# 4. bmi 결측치와 뇌졸중 비율 비교
# -------------------------------
st.header("4️⃣ 체질량지수(bmi) 결측치와 뇌졸중 비율")

bmi_missing_count = df["bmi"].isnull().sum()
bmi_missing_stroke_ratio = df[df["bmi"].isnull()]["stroke"].mean() * 100
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare = pd.DataFrame({
    "구분": ["bmi 결측치인 사람", "전체 사람"],
    "인원 수": [bmi_missing_count, len(df)],
    "뇌졸중 비율(%)": [round(bmi_missing_stroke_ratio, 2), round(overall_stroke_ratio, 2)]
})

st.dataframe(bmi_compare, use_container_width=True)

st.markdown("---")

# -------------------------------
# 5. 흡연 상태별 사람 수
# -------------------------------
st.header("5️⃣ 흡연 상태(smoking_status)별 사람 수")

smoking_count = df["smoking_status"].value_counts().reset_index()
smoking_count.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count, use_container_width=True)
