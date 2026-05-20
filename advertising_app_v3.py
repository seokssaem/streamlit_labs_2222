from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ── 한글 폰트 설정 ─────────────────────────────────────────
# Matplotlib은 기본 폰트가 한글을 지원하지 않는 경우가 많습니다.
# Windows 환경에서는 Malgun Gothic을 우선 사용하고, 음수 기호가 깨지지 않도록 unicode_minus를 끕니다.
matplotlib.rcParams["axes.unicode_minus"] = False
try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"


# ── 페이지 기본 설정 ────────────────────────────────────────
st.set_page_config(
    page_title="광고 플랫폼별 판매량 예측",
    page_icon="📊",
)

st.title("📊 광고 플랫폼별 판매량 예측")
st.caption("Advertising Dataset · Linear Regression (Simple & Multiple)")
st.divider()


# ══════════════════════════════════════════════════════════════
# 0. 데이터 로드
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """Kaggle Advertising 데이터셋을 읽어옵니다.

    Streamlit은 실행 위치에 따라 상대 경로가 달라질 수 있으므로,
    현재 파이썬 파일이 있는 폴더를 기준으로 input/advertising.csv를 찾습니다.
    파일이 없을 때는 앱 구조를 확인할 수 있도록 샘플 데이터를 생성합니다.
    """
    csv_path = Path(__file__).resolve().parent / "input" / "advertising.csv"

    try:
        df_loaded = pd.read_csv(csv_path)
        data_source = f"Kaggle CSV: {csv_path}"
    except Exception:
        # 실습 환경에서 CSV가 없을 때 앱이 멈추지 않도록 Advertising 데이터와 비슷한 형태의 샘플을 만듭니다.
        np.random.seed(42)
        n = 200
        tv = np.random.uniform(0.7, 296.4, n)
        radio = np.random.uniform(0.0, 49.6, n)
        newspaper = np.random.uniform(0.3, 114.0, n)
        sales = 0.047 * tv + 0.189 * radio + 0.003 * newspaper + np.random.normal(7, 1.5, n)
        df_loaded = pd.DataFrame({"TV": tv, "Radio": radio, "Newspaper": newspaper, "Sales": sales})
        data_source = "샘플 데이터(광고 CSV 파일을 찾지 못함)"

    # Kaggle 원본에 불필요한 인덱스 컬럼이 포함된 경우를 대비해 제거합니다.
    df_loaded = df_loaded.loc[:, ~df_loaded.columns.str.contains(r"^Unnamed")]
    return df_loaded, data_source


df, source_text = load_data()

# 분석에 필요한 컬럼을 고정해 두면, 이후 코드에서 오타나 컬럼 누락을 빨리 발견할 수 있습니다.
FEATURE_COLUMNS = ["TV", "Radio", "Newspaper"]
TARGET_COLUMN = "Sales"
required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"CSV 파일에 필요한 컬럼이 없습니다: {', '.join(missing_columns)}")
    st.stop()

# scikit-learn에 들어갈 값은 숫자형이어야 하므로 필요한 컬럼만 숫자로 변환합니다.
# 변환할 수 없는 값은 NaN이 되고, 아래에서 결측치 처리 상태를 확인할 수 있습니다.
df = df[required_columns].apply(pd.to_numeric, errors="coerce")


# ══════════════════════════════════════════════════════════════
# 1. 데이터 탐색 탭
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 데이터 탐색", "🔥 상관관계 분석", "📈 단순 선형 회귀", "📊 다중 선형 회귀"]
)

with tab1:
    st.subheader("데이터셋 미리보기")
    st.caption(source_text)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("총 데이터 수", f"{len(df):,}개")
    col_b.metric("속성 수", f"{df.shape[1]}개")
    col_c.metric("결측치", f"{df.isnull().sum().sum()}개")

    st.markdown("**랜덤 샘플 5개**")
    st.dataframe(df.sample(min(5, len(df)), random_state=42), width="stretch")

    st.markdown("**기초 통계량**")
    st.dataframe(df.describe().T.style.format("{:.2f}"), width="stretch")

    # 각 컬럼의 값 분포를 한눈에 확인합니다.
    # TV, Radio, Newspaper는 광고비이고 Sales는 판매량이므로, 분포 차이가 모델 해석에 영향을 줍니다.
    st.markdown("**속성별 분포**")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for ax, col, color in zip(axes, df.columns, colors):
        ax.hist(df[col].dropna(), bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("값")
        ax.set_ylabel("빈도")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 2. 상관관계 분석 탭
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("속성별 상관관계")

    # 상관계수는 -1~1 범위이며, Sales와 어떤 광고 매체가 같이 움직이는지 확인하는 용도입니다.
    # 상관관계가 높다고 해서 반드시 인과관계가 있다는 뜻은 아닙니다.
    corr = df.corr(numeric_only=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**상관계수 행렬**")
        st.dataframe(corr.style.background_gradient(cmap="RdPu").format("{:.3f}"), width="stretch")

    with col2:
        st.markdown("**Sales 기준 내림차순 정렬**")
        corr_sort = corr[[TARGET_COLUMN]].sort_values(TARGET_COLUMN, ascending=False)
        st.dataframe(corr_sort.style.background_gradient(cmap="YlGn").format("{:.3f}"), width="stretch")

    st.markdown("**히트맵**")
    cmap_choice = st.selectbox("컬러맵 선택", ["RdPu", "YlGn", "Blues", "coolwarm", "viridis"], index=0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap_choice, ax=ax2, linewidths=0.5)
    ax2.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # 광고비와 판매량의 관계를 산점도로 보고, 단순 추세선을 추가합니다.
    st.markdown("**광고 플랫폼별 Sales 산점도 (추세선 포함)**")
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    palette = ["#4C72B0", "#55A868", "#C44E52"]
    for ax, platform, color in zip(axes3, FEATURE_COLUMNS, palette):
        ax.scatter(df[platform], df[TARGET_COLUMN], alpha=0.4, color=color, s=20)
        slope, intercept = np.polyfit(df[platform], df[TARGET_COLUMN], 1)
        x_line = np.linspace(df[platform].min(), df[platform].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5)
        ax.set_xlabel(f"{platform} 광고비 ($)")
        ax.set_ylabel(TARGET_COLUMN)
        ax.set_title(f"{platform} vs Sales  (r={df[platform].corr(df[TARGET_COLUMN]):.2f})")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ══════════════════════════════════════════════════════════════
# 3. 단순 선형 회귀 탭
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("단순 선형 회귀 (TV → Sales)")
    st.markdown("> $y = wx + b$ · 독립변수: **TV 광고비**, 종속변수: **Sales**")

    st.markdown("#### ⚙️ 모델 파라미터")
    col_p1, col_p2 = st.columns(2)
    test_size1 = col_p1.slider("테스트 비율", 0.1, 0.5, 0.3, 0.05, key="ts1")
    random_state1 = col_p2.number_input("Random State", 0, 999, 10, key="rs1")

    # 단순 선형 회귀는 TV 광고비 하나만 사용합니다.
    # LinearRegression은 변수 스케일링이 필수는 아니므로 원래 광고비 단위 그대로 학습합니다.
    # 이렇게 해야 화면에 표시되는 회귀식의 계수가 실제 입력 단위($)와 일치합니다.
    X1 = df[["TV"]]
    y1 = df[TARGET_COLUMN]
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X1, y1, test_size=test_size1, random_state=int(random_state1)
    )

    model1 = LinearRegression()
    model1.fit(X_tr1, y_tr1)
    y_pred1 = model1.predict(X_te1)

    mse1 = mean_squared_error(y_te1, y_pred1)
    r2_1 = r2_score(y_te1, y_pred1)
    w1 = model1.coef_[0]
    b1 = model1.intercept_

    st.markdown("#### 📐 성능 지표")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MSE", f"{mse1:.2f}", help="평균제곱오차입니다. 0에 가까울수록 좋습니다.")
    m2.metric("R² Score", f"{r2_1:.3f}", help="설명력입니다. 1에 가까울수록 좋습니다.")
    m3.metric("기울기 (w)", f"{w1:.4f}")
    m4.metric("절편 (b)", f"{b1:.2f}")

    st.info(f"📌 회귀식: **Sales = {w1:.4f} × TV + {b1:.2f}**")

    # 테스트 데이터의 실제값과 예측값, 잔차를 비교합니다.
    # 잔차가 0 주변에서 특별한 패턴 없이 흩어지면 선형 모델이 비교적 잘 맞는 편입니다.
    c_idx = list(range(1, len(y_te1) + 1))
    error1 = y_te1.values - y_pred1

    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    axes4[0].plot(c_idx, y_te1.values, color="red", label="실젯값", linewidth=1.2)
    axes4[0].plot(c_idx, y_pred1, color="blue", label="예측값", linewidth=1.2, linestyle="--")
    axes4[0].set_title("실젯값 vs 예측값")
    axes4[0].set_xlabel("index")
    axes4[0].set_ylabel("Sales")
    axes4[0].legend()

    axes4[1].plot(c_idx, error1, color="green", linewidth=1.2)
    axes4[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes4[1].set_title("잔차(Residual)")
    axes4[1].set_xlabel("index")
    axes4[1].set_ylabel("error")

    # 회귀선은 x축 값을 정렬해서 그려야 선이 지그재그로 보이지 않습니다.
    x_line = np.linspace(X1["TV"].min(), X1["TV"].max(), 100)
    y_line = model1.predict(pd.DataFrame({"TV": x_line}))
    axes4[2].scatter(X1["TV"], y1, color="red", s=15, alpha=0.5, label="전체 데이터")
    axes4[2].plot(x_line, y_line, color="blue", linewidth=2, label="회귀선")
    axes4[2].set_title("단순 선형 회귀선")
    axes4[2].set_xlabel("TV 광고비 ($)")
    axes4[2].set_ylabel("Sales")
    axes4[2].legend()

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")
    st.markdown("#### 🔮 TV 광고비 입력 → 판매량 예측")
    tv_input = st.slider(
        "TV 광고비 ($)",
        float(df["TV"].min()),
        float(df["TV"].max()),
        150.0,
        0.1,
        key="tv_pred",
    )
    predicted_sales = model1.predict(pd.DataFrame({"TV": [tv_input]}))[0]
    st.success(f"TV 광고비 **${tv_input:,.1f}** 투입 시 예상 판매량: **{predicted_sales:.2f}** (단위: $1,000)")


# ══════════════════════════════════════════════════════════════
# 4. 다중 선형 회귀 탭
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("다중 선형 회귀 (TV + Radio + Newspaper → Sales)")
    st.markdown("> $y = w_1x_1 + w_2x_2 + w_3x_3 + b$ · 독립변수: **3개**, 종속변수: **Sales**")

    st.markdown("#### ⚙️ 모델 파라미터")
    col_p3, col_p4 = st.columns(2)
    test_size2 = col_p3.slider("테스트 비율", 0.1, 0.5, 0.3, 0.05, key="ts2")
    random_state2 = col_p4.number_input("Random State", 0, 999, 10, key="rs2")

    # 다중 선형 회귀는 TV, Radio, Newspaper를 동시에 사용합니다.
    # 여러 변수를 함께 넣으면 각 계수는 "다른 변수들이 같을 때" 해당 광고비가 1 증가할 때의 변화량으로 해석합니다.
    X2 = df[FEATURE_COLUMNS]
    y2 = df[TARGET_COLUMN]
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X2, y2, test_size=test_size2, random_state=int(random_state2)
    )

    model2 = LinearRegression()
    model2.fit(X_tr2, y_tr2)
    y_pred2 = model2.predict(X_te2)

    mse2 = mean_squared_error(y_te2, y_pred2)
    r2_2 = r2_score(y_te2, y_pred2)
    coefs = dict(zip(FEATURE_COLUMNS, model2.coef_))
    b2 = model2.intercept_

    st.markdown("#### 📐 성능 지표")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("MSE", f"{mse2:.2f}")
    mc2.metric("R² Score", f"{r2_2:.3f}")
    mc3.metric("w(TV)", f"{coefs['TV']:.4f}")
    mc4.metric("w(Radio)", f"{coefs['Radio']:.4f}")
    mc5.metric("w(Newspaper)", f"{coefs['Newspaper']:.4f}")

    formula = (
        f"Sales = {coefs['TV']:.4f} × TV + {coefs['Radio']:.4f} × Radio "
        f"+ {coefs['Newspaper']:.4f} × Newspaper + {b2:.2f}"
    )
    st.info(f"📌 회귀식: **{formula}**")

    st.markdown("#### 📊 단순 vs 다중 선형 회귀 성능 비교")
    comp_df = pd.DataFrame(
        {
            "모델": ["단순 선형 회귀 (TV만)", "다중 선형 회귀 (TV+Radio+Newspaper)"],
            "MSE": [mse1, mse2],
            "R² Score": [r2_1, r2_2],
        }
    )
    st.dataframe(
        comp_df.style.highlight_min(subset=["MSE"], color="#d4edda")
        .highlight_max(subset=["R² Score"], color="#d4edda")
        .format({"MSE": "{:.2f}", "R² Score": "{:.3f}"}),
        width="stretch",
    )

    c_idx2 = list(range(1, len(y_te2) + 1))
    error2 = y_te2.values - y_pred2

    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4))
    axes5[0].plot(c_idx2, y_te2.values, color="red", label="실젯값", linewidth=1.2)
    axes5[0].plot(c_idx2, y_pred2, color="blue", label="예측값", linewidth=1.2, linestyle="--")
    axes5[0].set_title("실젯값 vs 예측값 (다중 선형 회귀)")
    axes5[0].set_xlabel("index")
    axes5[0].set_ylabel("Sales")
    axes5[0].legend()

    axes5[1].plot(c_idx2, error2, color="green", linewidth=1.2)
    axes5[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes5[1].set_title("잔차(Residual) (다중 선형 회귀)")
    axes5[1].set_xlabel("index")
    axes5[1].set_ylabel("error")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # 계수 막대그래프는 광고비 1단위 증가가 Sales에 미치는 추정 효과를 비교합니다.
    # 변수 단위가 모두 광고비($)라서 원 단위 계수를 그대로 보여주는 것이 해석하기 쉽습니다.
    st.markdown("#### 🏆 광고 플랫폼별 계수(영향력) 비교")
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    bars = ax6.barh(list(coefs.keys()), list(coefs.values()), color=["#4C72B0", "#55A868", "#C44E52"])
    ax6.set_xlabel("계수(coefficient)")
    ax6.set_title("광고 플랫폼별 판매 영향력")
    for bar, val in zip(bars, coefs.values()):
        label_x = val + 0.003 if val >= 0 else val - 0.003
        horizontal_align = "left" if val >= 0 else "right"
        ax6.text(label_x, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", ha=horizontal_align, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    st.markdown("---")
    st.markdown("#### 🔮 광고비 입력 → 판매량 예측")
    col_i1, col_i2, col_i3 = st.columns(3)
    tv_in = col_i1.number_input("TV 광고비 ($)", 0.0, 300.0, 150.0, 1.0)
    radio_in = col_i2.number_input("Radio 광고비 ($)", 0.0, 50.0, 25.0, 0.5)
    news_in = col_i3.number_input("Newspaper 광고비 ($)", 0.0, 120.0, 30.0, 1.0)

    input_df = pd.DataFrame([[tv_in, radio_in, news_in]], columns=FEATURE_COLUMNS)
    pred_sales2 = model2.predict(input_df)[0]
    st.success(
        f"TV **${tv_in}** · Radio **${radio_in}** · Newspaper **${news_in}** 투입 시 "
        f"예상 판매량: **{pred_sales2:.2f}** (단위: $1,000)"
    )


# ── 푸터 ────────────────────────────────────────────────────
st.divider()
st.caption("📦 Stack: pandas · scikit-learn · matplotlib · seaborn · Streamlit")
