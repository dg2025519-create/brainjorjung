import streamlit as st
import pandas as pd

# -------------------------------
# 페이지 기본 설정 (브라우저 탭 제목 + 아이콘)
# -------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# 앱 제목
# -------------------------------
st.title("🧠 뇌졸중 예측 실습실")
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
# 데이터 소개
# -------------------------------
st.header("📌 데이터 소개")
st.write(
    """
    이 데이터는 환자의 다양한 건강 정보와 생활 습관을 바탕으로
    **뇌졸중(stroke) 발생 여부**를 담고 있는 데이터입니다.
    나이, 고혈압, 심장병, 결혼 여부, 직업 종류, 거주 형태,
    평균 혈당 수치, 체질량지수(BMI), 흡연 상태 등의 정보를 포함하고 있습니다.
    """
)

# -------------------------------
# 큰 숫자 카드 4개
# -------------------------------
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = round(stroke_count / total_people * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중(stroke=1) 인원", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio} %")

st.markdown("---")

# -------------------------------
# 열 설명 표
# -------------------------------
st.header("📋 열(컬럼) 설명")

col_info = []
for col in df.columns:
    dtype = df[col].dtype

    if dtype == "object":
        unique_vals = df[col].dropna().unique()
        value_kind = ", ".join(map(str, unique_vals))
    else:
        n_unique = df[col].nunique()
        if n_unique <= 10:
            unique_vals = sorted(df[col].dropna().unique())
            value_kind = ", ".join(map(str, unique_vals))
        else:
            value_kind = f"숫자形 (최소 {df[col].min()} ~ 최대 {df[col].max()})"

    missing_count = df[col].isnull().sum()

    col_info.append({
        "열 이름": col,
        "우리말 뜻": "",
        "값의 종류": value_kind,
        "빈 값 개수": missing_count
    })

col_info_df = pd.DataFrame(col_info)

st.data_editor(
    col_info_df,
    use_container_width=True,
    num_rows="fixed",
    key="col_info_editor"
)

st.caption("👉 '우리말 뜻' 칸은 교재를 참고하여 직접 채워 넣어 보세요.")

st.markdown("---")

# -------------------------------
# 데이터 첫 5줄 보기
# -------------------------------
st.header("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# -------------------------------
# 데이터 출처
# -------------------------------
st.header("📚 데이터 출처")
source_text = st.text_area(
    "아래에 교재에 적힌 데이터 출처를 입력하세요.",
    placeholder="예) 출처: ...",
    height=100
)

if source_text:
    st.success("입력한 출처:")
    st.write(source_text)
