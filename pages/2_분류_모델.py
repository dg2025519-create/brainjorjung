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
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model, X_train, y_train, X_test, y_test)
tree_train_acc, tree_test_acc = get_accuracies(tree_model, X_train, y_train, X_test, y_test)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model, X_train, y_train, X_test, y_test)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{log_test_acc:.3f}"
    )
    st.caption(f"훈련 정확도: {log_train_acc:.3f}　|　테스트 정확도: {log_test_acc:.3f}")

with col2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{tree_test_acc:.3f}"
    )
    st.caption(f"훈련 정확도: {tree_train_acc:.3f}　|　테스트 정확도: {tree_test_acc:.3f}")

with col3:
    st.metric(
        label="입력을 하나도 보지 않고 많은 쪽으로 답하는 모델",
        value=f"{dummy_test_acc:.3f}"
    )
    st.caption(f"훈련 정확도: {dummy_train_acc:.3f}　|　테스트 정확도: {dummy_test_acc:.3f}")

st.markdown("---")

# -------------------------------
# 5. 산점도 + 결정 경계
# -------------------------------
st.header("4️⃣ 산점도와 로지스틱 회귀 경계선")

axis_kor = st.multiselect(
    "가로축과 세로축으로 쓸 속성 2개를 고르세요.",
    options=selected_kor,
    default=selected_kor[:2],
    max_selections=2
)

if len(axis_kor) != 2:
    st.warning("⚠️ 가로축과 세로축으로 쓸 속성을 정확히 2개 골라야 그림을 그릴 수 있습니다.")
else:
    x_kor, y_kor = axis_kor[0], axis_kor[1]
    x_eng, y_eng = KOR_TO_ENG[x_kor], KOR_TO_ENG[y_kor]

    other_features = [f for f in selected_eng if f not in [x_eng, y_eng]]
    fixed_values = {}
    for f in other_features:
        fixed_values[f] = X_test[f].median()

    if fixed_values:
        fixed_text = ", ".join([f"{FEATURE_KOR[f]} = {v:.2f}" for f, v in fixed_values.items()])
        st.write(f"📌 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
    else:
        st.write("📌 선택한 속성이 두 축뿐이라 고정할 다른 속성이 없습니다.")

    # 산점도 그리기
    fig = go.Figure()

    for label, name, color in [(0, "뇌졸중 아님", "blue"), (1, "뇌졸중", "red")]:
        subset = test_df[test_df["stroke"] == label]
        fig.add_trace(go.Scatter(
            x=subset[x_eng],
            y=subset[y_eng],
            mode="markers",
            name=name,
            marker=dict(color=color, size=7, opacity=0.6)
        ))

    x_min, x_max = X_test[x_eng].min(), X_test[x_eng].max()
    y_min, y_max = X_test[y_eng].min(), X_test[y_eng].max()

    # 로지스틱 회귀 결정 경계선 계산
    coef = log_model.coef_[0]
    intercept = log_model.intercept_[0]

    feature_index = {f: i for i, f in enumerate(selected_eng)}
    x_idx = feature_index[x_eng]
    y_idx = feature_index[y_eng]

    # 고정된 값들의 기여도 계산
    fixed_contribution = 0.0
    for f, v in fixed_values.items():
        fixed_contribution += coef[feature_index[f]] * v

    coef_x = coef[x_idx]
    coef_y = coef[y_idx]

    line_drawn = False
    line_out_of_range = False

    if abs(coef_y) > 1e-10:
        x_range = np.linspace(x_min, x_max, 200)
        # coef_x * x + coef_y * y + fixed_contribution + intercept = 0
        y_line = -(coef_x * x_range + fixed_contribution + intercept) / coef_y

        # 그림 범위 안에 있는지 확인
        in_range_mask = (y_line >= y_min) & (y_line <= y_max)

        if in_range_mask.any():
            fig.add_trace(go.Scatter(
                x=x_range[in_range_mask],
                y=y_line[in_range_mask],
                mode="lines",
                name="로지스틱 회귀 경계선 (0.5)",
                line=dict(color="black", width=2, dash="dash")
            ))
            line_drawn = True
        else:
            line_out_of_range = True
    else:
        line_out_of_range = True

    # 의사결정트리 배경 칠하기 (다른 속성은 고정값 사용)
    grid_size = 100
    xx = np.linspace(x_min, x_max, grid_size)
    yy = np.linspace(y_min, y_max, grid_size)
    xx_grid, yy_grid = np.meshgrid(xx, yy)

    grid_df = pd.DataFrame({x_eng: xx_grid.ravel(), y_eng: yy_grid.ravel()})
    for f, v in fixed_values.items():
        grid_df[f] = v
    grid_df = grid_df[selected_eng]

    tree_pred_grid = tree_model.predict(grid_df)
    tree_pred_grid = tree_pred_grid.reshape(xx_grid.shape)

    fig.add_trace(go.Contour(
        x=xx,
        y=yy,
        z=tree_pred_grid,
        showscale=False,
        opacity=0.2,
        colorscale=[[0, "blue"], [1, "red"]],
        contours=dict(coloring="fill"),
        name="의사결정트리 영역",
        hoverinfo="skip"
    ))

    fig.update_layout(
        title="테스트 데이터 산점도와 결정 경계",
        xaxis_title=x_kor,
        yaxis_title=y_kor,
        legend_title="실제 뇌졸중 여부"
    )

    st.plotly_chart(fig, use_container_width=True)

    if line_out_of_range:
        st.write("📌 로지스틱 회귀의 0.5 경계선은 이 그림의 범위 밖에 있습니다.")
    elif line_drawn:
        st.write("📌 위 그림의 검은 점선이 로지스틱 회귀의 0.5 결정 경계선입니다.")

st.markdown("---")

# -------------------------------
# 6. 의사결정트리 가지 그림
# -------------------------------
st.header("5️⃣ 의사결정트리 가지 그림")

def build_dot(tree, feature_names_eng, feature_names_kor_map):
    tree_ = tree.tree_
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    def recurse(node_id):
        n_samples = tree_.n_node_samples[node_id]
        value = tree_.value[node_id][0]
        n_stroke = int(value[1]) if len(value) > 1 else 0
        ratio = n_stroke / n_samples if n_samples > 0 else 0

        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]

        if is_leaf:
            pred_class = np.argmax(value)
            if pred_class == 1:
                color = "salmon"
                answer = "뇌졸중"
            else:
                color = "lightblue"
                answer = "아님"
            label = f"인원: {n_samples}\\n뇌졸중: {n_stroke}\\n비율: {ratio:.2f}\\n답: {answer}"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
        else:
            feat_idx = tree_.feature[node_id]
            feat_eng = feature_names_eng[feat_idx]
            feat_kor = feature_names_kor_map[feat_eng]
            threshold = tree_.threshold[node_id]
            label = f"{feat_kor} <= {threshold:.2f} ?\\n인원: {n_samples}\\n뇌졸중: {n_stroke}\\n비율: {ratio:.2f}"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="white"];')

            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]

            dot_lines.append(f'{node_id} -> {left_id} [label="예"];')
            dot_lines.append(f'{node_id} -> {right_id} [label="아니요"];')

            recurse(left_id)
            recurse(right_id)

    recurse(0)
    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_str = build_dot(tree_model, selected_eng, FEATURE_KOR)
st.graphviz_chart(dot_str)

st.markdown("---")

# -------------------------------
# 7. 트리 요약
# -------------------------------
st.header("6️⃣ 의사결정트리 요약")

tree_ = tree_model.tree_
is_leaf_arr = tree_.children_left == tree_.children_right
n_leaf = is_leaf_arr.sum()

n_leaf_not_stroke = 0
for node_id in range(tree_.node_count):
    if is_leaf_arr[node_id]:
        value = tree_.value[node_id][0]
        pred_class = np.argmax(value)
        if pred_class == 0:
            n_leaf_not_stroke += 1

used_features_idx = set(tree_.feature[tree_.feature >= 0])
used_features_eng = [selected_eng[i] for i in used_features_idx]
used_features_kor = [FEATURE_KOR[f] for f in used_features_eng]

st.write(f"- 답을 내는 마디(리프 노드)는 모두 **{n_leaf} 칸**입니다.")
st.write(f"- 그중 **{n_leaf_not_stroke} 칸**이 '아님'이라고 답합니다.")
if used_features_kor:
    st.write(f"- 이 나무가 실제로 물은 속성은: **{', '.join(used_features_kor)}** 입니다.")
else:
    st.write("- 이 나무는 어떤 속성도 실제로 묻지 않았습니다.")
