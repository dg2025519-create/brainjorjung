import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 페이지 기본 설정
# -------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 분류 모델 만들기")
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
# 속성 이름 매핑
# -------------------------------
FEATURE_KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
KOR_TO_ENG = {v: k for k, v in FEATURE_KOR.items()}
ALL_FEATURES_ENG = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
ALL_FEATURES_KOR = [FEATURE_KOR[f] for f in ALL_FEATURES_ENG]

# -------------------------------
# 1. 속성 선택
# -------------------------------
st.header("1️⃣ 입력 속성 고르기")

default_kor = [FEATURE_KOR[f] for f in ALL_FEATURES_ENG if f != "bmi"]

selected_kor = st.multiselect(
    "모델에 사용할 속성을 고르세요 (처음에는 체질량지수를 뺀 넷이 선택되어 있습니다).",
    options=ALL_FEATURES_KOR,
    default=default_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 2개 이상 골라야 모델을 만들 수 있습니다.")
    st.stop()

selected_eng = [KOR_TO_ENG[k] for k in selected_kor]

st.markdown("---")

# -------------------------------
# 2. 훈련/테스트 나누기 (열 명씩 묶어 앞 세 명 고정)
# -------------------------------
st.header("2️⃣ 훈련용·테스트용 나누기")

df_sorted = df.sort_values("id").reset_index(drop=True)
df_sorted["group_no"] = df_sorted.index // 10
df_sorted["pos_in_group"] = df_sorted.index % 10

test_df = df_sorted[df_sorted["pos_in_group"] < 3].copy()
train_df = df_sorted[df_sorted["pos_in_group"] >= 3].copy()

st.write(f"- 훈련용 인원: **{len(train_df):,} 명**")
st.write(f"- 테스트용 인원: **{len(test_df):,} 명**")

# -------------------------------
# 3. bmi 결측치 채우기 (선택된 경우에만)
# -------------------------------
if "bmi" in selected_eng:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.write(f"- 체질량지수(bmi) 결측치는 훈련용 중앙값인 **{bmi_median:.2f}** 로 채웠습니다.")

X_train = train_df[selected_eng]
y_train = train_df["stroke"]
X_test = test_df[selected_eng]
y_test = test_df["stroke"]

st.markdown("---")

# -------------------------------
# 4. 모델 학습
# -------------------------------
st.header("3️⃣ 모델 학습 결과")

# 로지스틱 회귀
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# 더미 모델 (많은 쪽으로만 답하는 모델)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# 정확도 계산
def get_accuracies(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc
