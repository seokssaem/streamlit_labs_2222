import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ── 한글 폰트 설정 ─────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 페이지 기본 설정 ────────────────────────────────────────
st.set_page_config(
    page_title="광고 플랫폼별 판매량 예측",
    page_icon="📊",
    layout="wide",
)

st.title("📊 광고 플랫폼별 판매량 예측")
st.caption("Advertising Dataset · Linear Regression (Simple & Multiple)")
st.divider()

# ══════════════════════════════════════════════════════════════
# 0. 데이터 로드
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    # 캐글 데이터셋을 직접 사용하거나, 파일 경로를 맞춰주세요
    # https://www.kaggle.com/datasets/ashydv/advertising-dataset
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # 네트워크 안 될 때 샘플 데이터 생성
        np.random.seed(42)
        n = 200
        tv   = np.random.uniform(0.7, 296.4, n)
        radio= np.random.uniform(0.0, 49.6, n)
        news = np.random.uniform(0.3, 114.0, n)
        sales= 0.047*tv + 0.189*radio + 0.003*news + np.random.normal(7, 1.5, n)
        df   = pd.DataFrame({"TV": tv, "Radio": radio, "Newspaper": news, "Sales": sales})
    return df

df = load_data()

# ══════════════════════════════════════════════════════════════
# 1. 데이터 탐색 탭
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 데이터 탐색", "🔥 상관관계 분석", "📈 단순 선형 회귀", "📊 다중 선형 회귀"]
)

with tab1:
    st.subheader("데이터셋 미리보기")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("총 데이터 수", f"{len(df):,}개")
    col_b.metric("속성 수", f"{df.shape[1]}개")
    col_c.metric("결측치", f"{df.isnull().sum().sum()}개")

    st.markdown("**랜덤 샘플 5개**")
    st.dataframe(df.sample(5, random_state=42), use_container_width=True)

    st.markdown("**기초 통계량**")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    # 분포 히스토그램
    st.markdown("**속성별 분포**")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for ax, col, c in zip(axes, df.columns, colors):
        ax.hist(df[col], bins=20, color=c, alpha=0.8, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("값")
        ax.set_ylabel("빈도")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# 2. 상관관계 분석 탭
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("속성별 상관관계")

    corr = df.corr(numeric_only=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**상관계수 행렬**")
        st.dataframe(corr.style.background_gradient(cmap="RdPu").format("{:.3f}"),
                     use_container_width=True)

    with col2:
        st.markdown("**Sales 기준 내림차순 정렬**")
        corr_sort = corr[["Sales"]].sort_values("Sales", ascending=False)
        st.dataframe(corr_sort.style.background_gradient(cmap="YlGn").format("{:.3f}"),
                     use_container_width=True)

    # 히트맵
    st.markdown("**히트맵**")
    cmap_choice = st.selectbox("컬러맵 선택", ["RdPu", "YlGn", "Blues", "coolwarm", "viridis"], index=0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap_choice, ax=ax2, linewidths=0.5)
    ax2.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # pairplot 섹션
    st.markdown("**광고 플랫폼별 Sales 산점도 (추세선 포함)**")
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    platforms = ["TV", "Radio", "Newspaper"]
    pal = ["#4C72B0", "#55A868", "#C44E52"]
    for ax, plat, c in zip(axes3, platforms, pal):
        ax.scatter(df[plat], df["Sales"], alpha=0.4, color=c, s=20)
        m, b = np.polyfit(df[plat], df["Sales"], 1)
        x_line = np.linspace(df[plat].min(), df[plat].max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5)
        ax.set_xlabel(f"{plat} 광고비 ($)")
        ax.set_ylabel("Sales")
        ax.set_title(f"{plat} vs Sales  (r={df[plat].corr(df['Sales']):.2f})")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ══════════════════════════════════════════════════════════════
# 3. 단순 선형 회귀 탭
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("단순 선형 회귀 (TV → Sales)")
    st.markdown("> $y = wx + b$ · 독립변수: **TV 광고비**, 종속변수: **Sales**")

    # 사이드바 파라미터
    st.markdown("#### ⚙️ 모델 파라미터")
    col_p1, col_p2 = st.columns(2)
    test_size1 = col_p1.slider("테스트 비율", 0.1, 0.5, 0.3, 0.05, key="ts1")
    random_state1 = col_p2.number_input("Random State", 0, 999, 10, key="rs1")

    # 전처리 & 학습
    X1 = df[["TV"]]
    y1 = df["Sales"]
    scaler1 = StandardScaler()
    X_scaled1 = scaler1.fit_transform(X1)
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X_scaled1, y1, test_size=test_size1, random_state=int(random_state1)
    )
    model1 = LinearRegression()
    model1.fit(X_tr1, y_tr1)
    y_pred1 = model1.predict(X_te1)

    mse1 = mean_squared_error(y_te1, y_pred1)
    r2_1 = r2_score(y_te1, y_pred1)
    w1   = model1.coef_[0]
    b1   = model1.intercept_

    # 성능 지표
    st.markdown("#### 📐 성능 지표")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MSE", f"{mse1:.2f}", help="0에 가까울수록 좋음")
    m2.metric("R² Score", f"{r2_1:.3f}", help="1에 가까울수록 좋음")
    m3.metric("기울기 (w)", f"{w1:.2f}")
    m4.metric("절편 (b)", f"{b1:.2f}")

    st.info(f"📌 회귀식: **y = {w1:.2f}x + {b1:.2f}**")

    # 그래프 3종
    c_idx = list(range(1, len(y_te1) + 1))
    error1 = y_te1.values - y_pred1

    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    # ① 실젯값 vs 예측값
    axes4[0].plot(c_idx, y_te1.values, color="red",  label="실젯값", linewidth=1.2)
    axes4[0].plot(c_idx, y_pred1,      color="blue", label="예측값", linewidth=1.2, linestyle="--")
    axes4[0].set_title("실젯값 vs 예측값")
    axes4[0].set_xlabel("index"); axes4[0].set_ylabel("Sales")
    axes4[0].legend()

    # ② 오차(잔차)
    axes4[1].plot(c_idx, error1, color="green", linewidth=1.2)
    axes4[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes4[1].set_title("잔차(Residual)")
    axes4[1].set_xlabel("index"); axes4[1].set_ylabel("error")

    # ③ 회귀선
    axes4[2].scatter(X_scaled1, y1, color="red", s=15, alpha=0.5, label="전체 데이터")
    axes4[2].plot(X_te1, y_pred1, color="blue", linewidth=2, label="회귀선")
    axes4[2].set_title("단순 선형 회귀선")
    axes4[2].set_xlabel("TV 광고비 (표준화)"); axes4[2].set_ylabel("Sales")
    axes4[2].legend()

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # 판매량 예측기
    st.markdown("---")
    st.markdown("#### 🔮 TV 광고비 입력 → 판매량 예측")
    tv_input = st.slider("TV 광고비 ($)", float(df["TV"].min()), float(df["TV"].max()),
                         150.0, 0.1, key="tv_pred")
    tv_scaled = scaler1.transform([[tv_input]])
    predicted_sales = model1.predict(tv_scaled)[0]
    st.success(f"TV 광고비 **${tv_input:,.1f}** 투입 시 예상 판매량: **{predicted_sales:.2f}** (단위: $1,000)")

# ══════════════════════════════════════════════════════════════
# 4. 다중 선형 회귀 탭
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("다중 선형 회귀 (TV + Radio + Newspaper → Sales)")
    st.markdown("> $y = w_1x_1 + w_2x_2 + w_3x_3 + b$ · 독립변수: **3개**, 종속변수: **Sales**")

    st.markdown("#### ⚙️ 모델 파라미터")
    col_p3, col_p4 = st.columns(2)
    test_size2   = col_p3.slider("테스트 비율", 0.1, 0.5, 0.3, 0.05, key="ts2")
    random_state2 = col_p4.number_input("Random State", 0, 999, 10, key="rs2")

    X2 = df.drop(columns=["Sales"])
    y2 = df["Sales"]
    scaler2 = StandardScaler()
    X_scaled2 = scaler2.fit_transform(X2)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_scaled2, y2, test_size=test_size2, random_state=int(random_state2)
    )
    model2 = LinearRegression()
    model2.fit(X_tr2, y_tr2)
    y_pred2 = model2.predict(X_te2)

    mse2 = mean_squared_error(y_te2, y_pred2)
    r2_2 = r2_score(y_te2, y_pred2)
    coefs = dict(zip(["TV", "Radio", "Newspaper"], model2.coef_))
    b2   = model2.intercept_

    # 성능 지표
    st.markdown("#### 📐 성능 지표")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("MSE", f"{mse2:.2f}")
    mc2.metric("R² Score", f"{r2_2:.3f}")
    mc3.metric("w(TV)", f"{coefs['TV']:.2f}")
    mc4.metric("w(Radio)", f"{coefs['Radio']:.2f}")
    mc5.metric("w(Newspaper)", f"{coefs['Newspaper']:.2f}")

    formula = (f"y = {coefs['TV']:.2f}·TV + {coefs['Radio']:.2f}·Radio "
               f"+ {coefs['Newspaper']:.2f}·Newspaper + {b2:.2f}")
    st.info(f"📌 회귀식: **{formula}**")

    # 단순 vs 다중 비교 카드
    st.markdown("#### 📊 단순 vs 다중 선형 회귀 성능 비교")
    comp_df = pd.DataFrame({
        "모델": ["단순 선형 회귀 (TV만)", "다중 선형 회귀 (TV+Radio+Newspaper)"],
        "MSE": [mse1, mse2],
        "R² Score": [r2_1, r2_2],
    })
    st.dataframe(comp_df.style.highlight_min(subset=["MSE"], color="#d4edda")
                              .highlight_max(subset=["R² Score"], color="#d4edda")
                              .format({"MSE": "{:.2f}", "R² Score": "{:.3f}"}),
                 use_container_width=True)

    # 그래프 2종
    c_idx2 = list(range(1, len(y_te2) + 1))
    error2 = y_te2.values - y_pred2

    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4))
    axes5[0].plot(c_idx2, y_te2.values, color="red",  label="실젯값", linewidth=1.2)
    axes5[0].plot(c_idx2, y_pred2,      color="blue", label="예측값", linewidth=1.2, linestyle="--")
    axes5[0].set_title("실젯값 vs 예측값 (다중 선형 회귀)")
    axes5[0].set_xlabel("index"); axes5[0].set_ylabel("Sales"); axes5[0].legend()

    axes5[1].plot(c_idx2, error2, color="green", linewidth=1.2)
    axes5[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes5[1].set_title("잔차(Residual) (다중 선형 회귀)")
    axes5[1].set_xlabel("index"); axes5[1].set_ylabel("error")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    # 계수 중요도 바 차트
    st.markdown("#### 🏆 광고 플랫폼별 계수(영향력) 비교")
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    bars = ax6.barh(list(coefs.keys()), list(coefs.values()),
                    color=["#4C72B0", "#55A868", "#C44E52"])
    ax6.set_xlabel("계수(coefficient)")
    ax6.set_title("광고 플랫폼별 판매 영향력")
    for bar, val in zip(bars, coefs.values()):
        ax6.text(val + 0.03, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    # 다중 회귀 예측기
    st.markdown("---")
    st.markdown("#### 🔮 광고비 입력 → 판매량 예측")
    col_i1, col_i2, col_i3 = st.columns(3)
    tv_in    = col_i1.number_input("TV 광고비 ($)", 0.0, 300.0, 150.0, 1.0)
    radio_in = col_i2.number_input("Radio 광고비 ($)", 0.0, 50.0, 25.0, 0.5)
    news_in  = col_i3.number_input("Newspaper 광고비 ($)", 0.0, 120.0, 30.0, 1.0)

    inp_scaled = scaler2.transform([[tv_in, radio_in, news_in]])
    pred_sales2 = model2.predict(inp_scaled)[0]
    st.success(
        f"TV **${tv_in}** · Radio **${radio_in}** · Newspaper **${news_in}** 투입 시 "
        f"예상 판매량: **{pred_sales2:.2f}** (단위: $1,000)"
    )

# ── 푸터 ────────────────────────────────────────────────────
st.divider()
st.caption("📦 Stack: pandas · scikit-learn · matplotlib · seaborn · Streamlit")
