# ─────────────────────────────────────────────────────────────
# [1] 라이브러리 import
#
# pathlib.Path : 파일 경로를 OS에 상관없이 안전하게 다루는 표준 라이브러리
#   - 윈도우: C:\Users\...\input\advertising.csv
#   - Mac/Linux: /home/.../input/advertising.csv
#   두 환경 모두 Path 객체 하나로 처리할 수 있습니다.
#
# import 순서 관례: 표준 라이브러리 → 서드파티 라이브러리 (알파벳 순)
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# [2] 한글 폰트 설정
#
# matplotlib의 기본 폰트(DejaVu Sans)는 한글을 지원하지 않아
# 그래프 제목·축 레이블이 □□□로 깨집니다.
#
# rcParams : matplotlib 전역 설정 딕셔너리
#   - axes.unicode_minus = False : 음수 기호(-)가 □로 깨지는 현상 방지
#   - font.family           : 그래프 전체에 적용할 폰트
#
# try/except 이유: 폰트 설정은 실패해도 앱이 멈추면 안 되기 때문입니다.
#   - Windows  → "Malgun Gothic" (맑은 고딕) 사용
#   - Mac/Linux → 설정 실패 시 기본 "DejaVu Sans" 유지
# ─────────────────────────────────────────────────────────────
matplotlib.rcParams["axes.unicode_minus"] = False
try:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"


# ─────────────────────────────────────────────────────────────
# [3] 페이지 기본 설정
#
# st.set_page_config()는 반드시 모든 st 코드보다 먼저 호출해야 합니다.
# 두 번째로 호출하면 StreamlitAPIException 오류가 발생합니다.
#
# page_title : 브라우저 탭에 표시되는 제목
# page_icon  : 탭 아이콘 (이모지 또는 이미지 경로)
# layout     : "centered"(기본·좁음) / "wide"(전체 너비)
#              넓은 화면이 필요할 때만 "wide"로 변경
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="광고 플랫폼별 판매량 예측",
    page_icon="📊",
)

st.title("📊 광고 플랫폼별 판매량 예측")
st.caption("Advertising Dataset · Linear Regression (Simple & Multiple)")
st.divider()   # 시각적 구분선 (얇은 수평선)


# ─────────────────────────────────────────────────────────────
# [4] 데이터 로드
#
# @st.cache_data 데코레이터
#   Streamlit은 사용자가 슬라이더를 움직이거나 위젯을 바꿀 때마다
#   전체 스크립트를 위에서 아래로 다시 실행합니다.
#   @st.cache_data가 붙은 함수는 같은 인자로 호출될 때 결과를 캐시에서
#   꺼내 반환하므로 CSV를 반복해서 읽지 않아 속도가 크게 향상됩니다.
#
# Path(__file__) 사용 이유
#   streamlit run 명령은 어느 폴더에서든 실행할 수 있습니다.
#   상대 경로 "input/advertising.csv"는 실행 위치에 따라 달라지지만,
#   Path(__file__).resolve().parent 는 항상 이 .py 파일이 있는 폴더를
#   가리키므로 위치에 무관하게 안전하게 파일을 찾습니다.
#
#   예시:
#     이 파일 위치: C:\ai_class\st_labs\advertising_app_v3.py
#     csv_path   : C:\ai_class\st_labs\input\advertising.csv
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Kaggle Advertising 데이터셋을 읽어옵니다."""
    csv_path = Path(__file__).resolve().parent / "input" / "advertising.csv"

    try:
        df_loaded = pd.read_csv(csv_path)
        data_source = f"Kaggle CSV: {csv_path}"
    except Exception:
        # CSV가 없을 때 앱이 멈추지 않도록 비슷한 분포의 샘플 데이터를 생성합니다.
        # 실제 수업에서는 input/advertising.csv가 있는지 먼저 확인하세요.
        np.random.seed(42)   # 시드 고정: 실행마다 동일한 샘플 데이터 생성
        n = 200
        tv        = np.random.uniform(0.7,   296.4, n)
        radio     = np.random.uniform(0.0,    49.6, n)
        newspaper = np.random.uniform(0.3,   114.0, n)
        sales     = 0.047 * tv + 0.189 * radio + 0.003 * newspaper \
                    + np.random.normal(7, 1.5, n)
        df_loaded   = pd.DataFrame({"TV": tv, "Radio": radio,
                                    "Newspaper": newspaper, "Sales": sales})
        data_source = "샘플 데이터(광고 CSV 파일을 찾지 못함)"

    # Kaggle CSV를 내려받으면 맨 앞에 "Unnamed: 0" 같은 인덱스 컬럼이
    # 붙어 있는 경우가 있습니다. 정규표현식 r"^Unnamed"로 시작하는
    # 컬럼을 자동으로 제거합니다.
    df_loaded = df_loaded.loc[:, ~df_loaded.columns.str.contains(r"^Unnamed")]
    return df_loaded, data_source


df, source_text = load_data()


# ─────────────────────────────────────────────────────────────
# [5] 컬럼 상수 정의 & 데이터 검증
#
# FEATURE_COLUMNS, TARGET_COLUMN을 상수로 분리하는 이유:
#   - 컬럼명을 코드 여러 곳에 문자열로 직접 쓰면 오타를 발견하기 어렵습니다.
#   - 상수로 정의하면 한 곳만 수정해도 전체에 반영됩니다.
#   - 관례적으로 상수는 대문자 + 밑줄로 표기합니다. (SCREAMING_SNAKE_CASE)
#
# 컬럼 검증 (missing_columns 체크)
#   CSV 파일이 예상과 다른 구조일 때 sklearn이 "컬럼을 찾을 수 없다"는
#   불친절한 에러를 내기 전에, 먼저 명확한 안내 메시지를 보여줍니다.
#   st.stop() : 이 줄 이후의 코드를 실행하지 않고 앱을 중단합니다.
#
# pd.to_numeric(errors="coerce")
#   CSV에 숫자처럼 보이지만 실제로는 문자열인 값이 섞여 있을 수 있습니다.
#   errors="coerce"는 변환할 수 없는 값을 NaN으로 바꿔 오류 없이 진행합니다.
#   탭1 결측치 카운트에서 이 NaN을 확인할 수 있습니다.
# ─────────────────────────────────────────────────────────────
FEATURE_COLUMNS  = ["TV", "Radio", "Newspaper"]   # 독립변수(입력) 컬럼명
TARGET_COLUMN    = "Sales"                         # 종속변수(예측 대상) 컬럼명
required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"CSV 파일에 필요한 컬럼이 없습니다: {', '.join(missing_columns)}")
    st.stop()

# 필요한 컬럼만 추출하고 숫자형으로 변환
df = df[required_columns].apply(pd.to_numeric, errors="coerce")


# ══════════════════════════════════════════════════════════════
# [6] 탭 레이아웃
#
# st.tabs(리스트) : 탭 이름 리스트를 받아 각 탭의 컨텍스트 객체를 반환합니다.
# with tab: 블록 안에 작성한 코드가 해당 탭에만 렌더링됩니다.
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 데이터 탐색", "🔥 상관관계 분석", "📈 단순 선형 회귀", "📊 다중 선형 회귀"]
)


# ══════════════════════════════════════════════════════════════
# TAB 1 — 데이터 탐색
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("데이터셋 미리보기")

    # source_text: 실제 CSV인지 샘플 데이터인지 출처를 표시합니다.
    st.caption(source_text)

    # st.columns(3) : 화면을 3등분해 나란히 배치합니다.
    # st.metric(레이블, 값) : 큼직한 숫자 카드를 표시합니다.
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("총 데이터 수", f"{len(df):,}개")
    col_b.metric("속성 수", f"{df.shape[1]}개")
    col_c.metric("결측치", f"{df.isnull().sum().sum()}개")

    # df.sample(n) : 무작위로 n행을 추출합니다.
    # min(5, len(df)) : 데이터가 5행 미만일 때도 오류 없이 동작합니다.
    # random_state=42 : 추출 패턴을 고정해 실행마다 같은 행이 나옵니다.
    # width="stretch" : 테이블을 컨테이너 너비에 맞게 늘립니다.
    st.markdown("**랜덤 샘플 5개**")
    st.dataframe(df.sample(min(5, len(df)), random_state=42), width="stretch")

    # df.describe() : count, mean, std, min, 25%, 50%, 75%, max 통계량
    # .T             : 행·열 전치(Transpose) → 속성이 행으로 보여 읽기 편합니다.
    # .style.format  : 소수점 2자리로 표시
    st.markdown("**기초 통계량**")
    st.dataframe(df.describe().T.style.format("{:.2f}"), width="stretch")

    # 히스토그램: 각 속성의 값 분포를 시각화합니다.
    # plt.subplots(1, 4) : 1행 4열 격자 → 속성 4개를 나란히 배치
    # zip(axes, df.columns, colors) : 세 리스트를 동시에 순회
    # df[col].dropna() : NaN을 제외하고 히스토그램을 그립니다.
    # plt.close(fig)   : 특정 Figure 객체를 메모리에서 해제합니다.
    #   plt.close() 를 빠뜨리면 Figure가 누적되어 메모리 경고가 발생합니다.
    st.markdown("**속성별 분포**")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for ax, col, color in zip(axes, df.columns, colors):
        ax.hist(df[col].dropna(), bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("값")
        ax.set_ylabel("빈도")
    plt.tight_layout()   # 서브플롯 간격 자동 조정
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# TAB 2 — 상관관계 분석
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("속성별 상관관계")

    # df.corr() : 피어슨 상관계수 행렬 계산
    #   값의 범위: -1(강한 음의 상관) ~ 0(무관) ~ +1(강한 양의 상관)
    #   주의: 상관관계가 높다고 인과관계가 있다는 의미는 아닙니다.
    #         (TV 광고비↑ → 판매량↑일 수 있지만, 제3의 요인 가능성도 있음)
    # numeric_only=True : 숫자형 컬럼만 포함 (문자형 컬럼 경고 방지)
    corr = df.corr(numeric_only=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**상관계수 행렬**")
        # background_gradient : 값 크기에 따라 셀 배경색을 그라디언트로 표시
        # format("{:.3f}")     : 소수점 3자리 표시
        st.dataframe(
            corr.style.background_gradient(cmap="RdPu").format("{:.3f}"),
            width="stretch"
        )

    with col2:
        st.markdown("**Sales 기준 내림차순 정렬**")
        # corr[[TARGET_COLUMN]] : Sales 열만 선택 (대괄호 두 겹 → DataFrame 유지)
        # sort_values(..., ascending=False) : 내림차순 정렬
        # → 판매량과 상관관계가 가장 높은 속성이 맨 위에 옵니다.
        corr_sort = corr[[TARGET_COLUMN]].sort_values(TARGET_COLUMN, ascending=False)
        st.dataframe(
            corr_sort.style.background_gradient(cmap="YlGn").format("{:.3f}"),
            width="stretch"
        )

    # 히트맵: 상관계수 행렬을 색상으로 표현합니다.
    # st.selectbox : 드롭다운 위젯 → 수업 중 컬러맵을 실시간으로 바꿔볼 수 있습니다.
    # sns.heatmap 주요 파라미터:
    #   annot=True      : 각 셀에 수치 표시
    #   fmt=".2f"       : 소수점 2자리
    #   linewidths=0.5  : 셀 경계선 두께
    st.markdown("**히트맵**")
    cmap_choice = st.selectbox(
        "컬러맵 선택",
        ["RdPu", "YlGn", "Blues", "coolwarm", "viridis"],
        index=0
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap_choice,
                ax=ax2, linewidths=0.5)
    ax2.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # 산점도 + 추세선: 광고비와 판매량의 관계를 시각적으로 확인합니다.
    # FEATURE_COLUMNS를 순회하므로 TV, Radio, Newspaper 순으로 3개 그래프가 나옵니다.
    #
    # np.polyfit(x, y, 1) : 1차(직선) 최소제곱 피팅
    #   → 기울기(slope)와 절편(intercept)을 반환합니다.
    #   → 이 선이 나중에 LinearRegression이 학습할 회귀선과 유사합니다.
    #
    # np.linspace(min, max, 100) : min~max 사이를 100등분한 x값 배열
    #   → 이 배열로 선을 그려야 지그재그 없이 매끄러운 직선이 나옵니다.
    #
    # df[platform].corr(df[TARGET_COLUMN]) : 두 컬럼의 피어슨 상관계수(r)
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
# TAB 3 — 단순 선형 회귀   y = w·x + b
#
# ▶ 표준화를 사용하지 않는 이유 (v3의 핵심 설계 결정)
#   LinearRegression은 표준화가 필수가 아닙니다.
#   표준화 없이 원본 단위($)로 학습하면:
#     w = "TV 광고비를 $1 늘릴 때 Sales가 얼마나 오르는가"
#   학생이 회귀식에 직접 숫자를 대입해 검산할 수 있습니다.
#
#   예: Sales = 0.0475 × TV + 7.03
#       TV = $150 입력 → 0.0475 × 150 + 7.03 = 14.15 ← 예측기 결과와 일치
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("단순 선형 회귀 (TV → Sales)")
    st.markdown("> $y = wx + b$ · 독립변수: **TV 광고비**, 종속변수: **Sales**")

    # 모델 파라미터 위젯
    # test_size   : 전체 데이터 중 테스트에 사용할 비율 (기본 30%)
    # random_state: 훈련/테스트 분할 패턴을 고정하는 시드 값
    #   → 같은 random_state면 실행마다 동일하게 분할됩니다.
    #   → 바꿔보면 성능 지표가 달라지는 것을 확인할 수 있습니다.
    # key="ts1"   : 같은 위젯 타입이 여러 개일 때 고유 식별자를 지정합니다.
    #   → 지정하지 않으면 탭3과 탭4의 슬라이더가 충돌합니다.
    st.markdown("#### ⚙️ 모델 파라미터")
    col_p1, col_p2 = st.columns(2)
    test_size1    = col_p1.slider("테스트 비율", 0.1, 0.5, 0.3, 0.05, key="ts1")
    random_state1 = col_p2.number_input("Random State", 0, 999, 10, key="rs1")

    # ── 독립변수 / 종속변수 분리 ──────────────────────────────
    # X1 : 독립변수. df[["TV"]] — 대괄호 두 겹으로 DataFrame 형태 유지
    #   sklearn은 2D 배열(행렬)을 입력으로 요구합니다.
    #   df["TV"] 처럼 대괄호 한 겹이면 1D Series가 되어 경고가 발생합니다.
    # y1 : 종속변수. df["Sales"] — 1D Series 형태가 맞습니다.
    X1 = df[["TV"]]
    y1 = df[TARGET_COLUMN]

    # ── 훈련/테스트 분할 ──────────────────────────────────────
    # train_test_split(X, y, test_size, random_state)
    #   test_size=0.3  → 200개 중 60개는 테스트, 140개는 훈련용
    #   random_state   → 분할 패턴 고정 (재현성 보장)
    #
    # 반환값 4개: 훈련X, 테스트X, 훈련y, 테스트y
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X1, y1, test_size=test_size1, random_state=int(random_state1)
    )

    # ── 모델 생성 및 학습 ──────────────────────────────────────
    # LinearRegression() : 최소제곱법(OLS)으로 w, b를 계산합니다.
    #   → 예측값과 실젯값의 차이(잔차)의 제곱합을 최소화하는 w, b를 찾습니다.
    # fit(X_train, y_train) : 훈련 데이터로 학습
    # predict(X_test)       : 테스트 데이터로 예측값 생성
    model1 = LinearRegression()
    model1.fit(X_tr1, y_tr1)
    y_pred1 = model1.predict(X_te1)

    # ── 평가 지표 계산 ─────────────────────────────────────────
    # MSE (Mean Squared Error, 평균제곱오차)
    #   = 평균( (실젯값 - 예측값)² )
    #   → 0에 가까울수록 예측이 정확합니다.
    #
    # R² (결정계수, R-squared)
    #   = 모델이 데이터의 분산을 얼마나 설명하는가 (0~1)
    #   → 1에 가까울수록 설명력이 높습니다.
    #   → 0이면 평균으로 예측한 것과 다를 바 없다는 의미입니다.
    #
    # model.coef_[0]   : 학습된 기울기 w (리스트에서 첫 번째 값)
    # model.intercept_ : 학습된 절편 b
    mse1 = mean_squared_error(y_te1, y_pred1)
    r2_1 = r2_score(y_te1, y_pred1)
    w1   = model1.coef_[0]
    b1   = model1.intercept_

    # 성능 지표 카드
    # help= : 마우스를 올리면 나타나는 툴팁 설명
    st.markdown("#### 📐 성능 지표")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MSE",        f"{mse1:.2f}", help="평균제곱오차입니다. 0에 가까울수록 좋습니다.")
    m2.metric("R² Score",   f"{r2_1:.3f}", help="설명력입니다. 1에 가까울수록 좋습니다.")
    m3.metric("기울기 (w)", f"{w1:.4f}")
    m4.metric("절편 (b)",   f"{b1:.2f}")

    # 회귀식을 실제 수치로 표시합니다.
    # 원본 단위이므로 "TV 광고비 $1 증가 → Sales {w1:.4f} 증가" 로 해석합니다.
    st.info(f"📌 회귀식: **Sales = {w1:.4f} × TV + {b1:.2f}**")

    # ── 그래프 3종 ────────────────────────────────────────────
    # c_idx  : x축에 쓸 인덱스 번호 리스트 [1, 2, 3, ..., n]
    # error1 : 잔차 = 실젯값 - 예측값
    #   잔차가 0 근처에서 특별한 패턴 없이 분포하면 모델이 잘 맞는 편입니다.
    c_idx  = list(range(1, len(y_te1) + 1))
    error1 = y_te1.values - y_pred1

    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    # ① 실젯값 vs 예측값
    #   빨간선(실젯값)과 파란선(예측값)이 얼마나 겹치는지 시각 확인
    axes4[0].plot(c_idx, y_te1.values, color="red",  label="실젯값", linewidth=1.2)
    axes4[0].plot(c_idx, y_pred1,      color="blue", label="예측값",
                  linewidth=1.2, linestyle="--")
    axes4[0].set_title("실젯값 vs 예측값")
    axes4[0].set_xlabel("index")
    axes4[0].set_ylabel("Sales")
    axes4[0].legend()

    # ② 잔차 그래프
    #   axhline(0) : y=0 기준선. 잔차가 이 선 위아래로 고르게 퍼져야 합니다.
    axes4[1].plot(c_idx, error1, color="green", linewidth=1.2)
    axes4[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes4[1].set_title("잔차(Residual)")
    axes4[1].set_xlabel("index")
    axes4[1].set_ylabel("error")

    # ③ 회귀선 그래프
    #   np.linspace로 x 범위 전체를 100등분한 뒤 predict()로 y를 구합니다.
    #   → 테스트 데이터(X_te1)를 그대로 쓰면 x값이 섞인 순서라
    #     선이 지그재그로 보일 수 있습니다.
    #   pd.DataFrame({"TV": x_line}) 으로 감싸는 이유:
    #     model1이 컬럼명이 있는 DataFrame으로 fit됐으므로
    #     같은 형태로 넣어야 feature name 경고가 없습니다.
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

    # ── 판매량 예측기 ─────────────────────────────────────────
    # 슬라이더로 TV 광고비를 입력하면 바로 예측값이 나옵니다.
    # pd.DataFrame({"TV": [tv_input]}) : 1행짜리 DataFrame
    #   → model1이 학습할 때와 동일한 컬럼명 구조로 맞춰야 경고가 없습니다.
    # [0] : predict()는 배열을 반환하므로 첫 번째 값만 꺼냅니다.
    #
    # ✅ 검산 방법 (학생 실습 포인트):
    #   회귀식에 직접 대입해보세요.
    #   예: TV = $150 → Sales = {w1:.4f} × 150 + {b1:.2f} = ?
    #   슬라이더 예측 결과와 같으면 회귀식을 제대로 이해한 것입니다.
    st.markdown("---")
    st.markdown("#### 🔮 TV 광고비 입력 → 판매량 예측")
    tv_input = st.slider(
        "TV 광고비 ($)",
        float(df["TV"].min()),   # 슬라이더 최솟값 = 데이터 최솟값
        float(df["TV"].max()),   # 슬라이더 최댓값 = 데이터 최댓값
        150.0,                   # 기본값
        0.1,                     # 이동 단위
        key="tv_pred",
    )
    predicted_sales = model1.predict(pd.DataFrame({"TV": [tv_input]}))[0]
    st.success(
        f"TV 광고비 **${tv_input:,.1f}** 투입 시 "
        f"예상 판매량: **{predicted_sales:.2f}** (단위: $1,000)"
    )


# ══════════════════════════════════════════════════════════════
# TAB 4 — 다중 선형 회귀   y = w1*x1 + w2*x2 + w3*x3 + b
#
# 단순 선형 회귀(TV 1개)와 다중 선형 회귀(3개)를 비교해
# 독립변수를 늘렸을 때 성능이 어떻게 달라지는지 확인합니다.
#
# 다중 회귀 계수 해석 (원본 단위 기준):
#   w_TV        = TV 광고비 $1 증가 시, 다른 광고비가 동일할 때 Sales 변화량
#   w_Radio     = Radio 광고비 $1 증가 시, 다른 광고비가 동일할 때 Sales 변화량
#   w_Newspaper = Newspaper 광고비 $1 증가 시, 다른 광고비가 동일할 때 Sales 변화량
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("다중 선형 회귀 (TV + Radio + Newspaper → Sales)")
    st.markdown("> $y = w_1x_1 + w_2x_2 + w_3x_3 + b$ · 독립변수: **3개**, 종속변수: **Sales**")

    st.markdown("#### ⚙️ 모델 파라미터")
    col_p3, col_p4 = st.columns(2)
    test_size2    = col_p3.slider("테스트 비율", 0.1, 0.5, 0.3, 0.05, key="ts2")
    random_state2 = col_p4.number_input("Random State", 0, 999, 10, key="rs2")

    # ── 독립변수 / 종속변수 분리 ──────────────────────────────
    # df[FEATURE_COLUMNS] : TV, Radio, Newspaper 3개 컬럼 선택
    #   상수 FEATURE_COLUMNS를 사용하므로 오타 없이 안전합니다.
    X2 = df[FEATURE_COLUMNS]
    y2 = df[TARGET_COLUMN]

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X2, y2, test_size=test_size2, random_state=int(random_state2)
    )

    # 단순 선형 회귀와 동일한 구조입니다.
    # sklearn은 독립변수 개수에 상관없이 같은 API를 사용합니다.
    model2 = LinearRegression()
    model2.fit(X_tr2, y_tr2)
    y_pred2 = model2.predict(X_te2)

    mse2  = mean_squared_error(y_te2, y_pred2)
    r2_2  = r2_score(y_te2, y_pred2)

    # model2.coef_ : [w_TV, w_Radio, w_Newspaper] 순서로 반환
    # zip(FEATURE_COLUMNS, model2.coef_) 으로 이름과 값을 묶어 딕셔너리로 만듭니다.
    coefs = dict(zip(FEATURE_COLUMNS, model2.coef_))
    b2    = model2.intercept_

    st.markdown("#### 📐 성능 지표")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("MSE",           f"{mse2:.2f}")
    mc2.metric("R² Score",      f"{r2_2:.3f}")
    mc3.metric("w(TV)",         f"{coefs['TV']:.4f}")
    mc4.metric("w(Radio)",      f"{coefs['Radio']:.4f}")
    mc5.metric("w(Newspaper)",  f"{coefs['Newspaper']:.4f}")

    formula = (
        f"Sales = {coefs['TV']:.4f} × TV"
        f" + {coefs['Radio']:.4f} × Radio"
        f" + {coefs['Newspaper']:.4f} × Newspaper"
        f" + {b2:.2f}"
    )
    st.info(f"📌 회귀식: **{formula}**")

    # ── 단순 vs 다중 성능 비교표 ───────────────────────────────
    # tab3에서 계산한 mse1, r2_1을 참조합니다. (전역 변수)
    # highlight_min : MSE가 더 낮은 행을 녹색으로 강조
    # highlight_max : R²가 더 높은 행을 녹색으로 강조
    # → 다중 회귀가 두 지표 모두에서 더 좋음을 한눈에 확인할 수 있습니다.
    st.markdown("#### 📊 단순 vs 다중 선형 회귀 성능 비교")
    comp_df = pd.DataFrame({
        "모델":      ["단순 선형 회귀 (TV만)", "다중 선형 회귀 (TV+Radio+Newspaper)"],
        "MSE":       [mse1, mse2],
        "R² Score":  [r2_1, r2_2],
    })
    st.dataframe(
        comp_df.style
            .highlight_min(subset=["MSE"],      color="#d4edda")
            .highlight_max(subset=["R² Score"], color="#d4edda")
            .format({"MSE": "{:.2f}", "R² Score": "{:.3f}"}),
        width="stretch",
    )

    # ── 실젯값 vs 예측값 / 잔차 그래프 ───────────────────────
    c_idx2 = list(range(1, len(y_te2) + 1))
    error2 = y_te2.values - y_pred2

    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4))
    axes5[0].plot(c_idx2, y_te2.values, color="red",  label="실젯값", linewidth=1.2)
    axes5[0].plot(c_idx2, y_pred2,      color="blue", label="예측값",
                  linewidth=1.2, linestyle="--")
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

    # ── 계수(영향력) 가로 막대 차트 ───────────────────────────
    # 가로 막대(barh)로 플랫폼별 계수 크기를 직관적으로 비교합니다.
    # 계수가 클수록 해당 광고비 $1이 Sales에 더 큰 영향을 줍니다.
    #
    # 레이블 위치 계산 (음수 계수 대비):
    #   val >= 0 → 막대 오른쪽(val + 0.003)에 왼쪽 정렬
    #   val <  0 → 막대 왼쪽(val - 0.003)에 오른쪽 정렬
    #   이 데이터에서는 음수 계수가 없지만, 방어 코드로 처리해둡니다.
    st.markdown("#### 🏆 광고 플랫폼별 계수(영향력) 비교")
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    bars = ax6.barh(
        list(coefs.keys()),
        list(coefs.values()),
        color=["#4C72B0", "#55A868", "#C44E52"]
    )
    ax6.set_xlabel("계수(coefficient)")
    ax6.set_title("광고 플랫폼별 판매 영향력")
    for bar, val in zip(bars, coefs.values()):
        label_x         = val + 0.003 if val >= 0 else val - 0.003
        horizontal_align = "left"    if val >= 0 else "right"
        ax6.text(label_x, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", ha=horizontal_align, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    # ── 판매량 예측기 (숫자 입력) ──────────────────────────────
    # 세 광고비를 입력하면 다중 회귀 모델로 Sales를 예측합니다.
    # pd.DataFrame([[...]], columns=FEATURE_COLUMNS) :
    #   1행짜리 DataFrame을 만들어 model2.predict()에 전달합니다.
    #   columns를 지정하면 학습 시와 컬럼 순서·이름이 일치하여 경고가 없습니다.
    #
    # ✅ 검산 방법 (학생 실습 포인트):
    #   회귀식에 직접 대입해보세요.
    #   예: TV=150, Radio=25, Newspaper=30 →
    #       Sales = {coefs['TV']:.4f}×150 + {coefs['Radio']:.4f}×25
    #             + {coefs['Newspaper']:.4f}×30 + {b2:.2f} = ?
    st.markdown("---")
    st.markdown("#### 🔮 광고비 입력 → 판매량 예측")
    col_i1, col_i2, col_i3 = st.columns(3)
    tv_in    = col_i1.number_input("TV 광고비 ($)",        0.0, 300.0, 150.0, 1.0)
    radio_in = col_i2.number_input("Radio 광고비 ($)",     0.0,  50.0,  25.0, 0.5)
    news_in  = col_i3.number_input("Newspaper 광고비 ($)", 0.0, 120.0,  30.0, 1.0)

    input_df    = pd.DataFrame([[tv_in, radio_in, news_in]], columns=FEATURE_COLUMNS)
    pred_sales2 = model2.predict(input_df)[0]
    st.success(
        f"TV **${tv_in}** · Radio **${radio_in}** · Newspaper **${news_in}** 투입 시 "
        f"예상 판매량: **{pred_sales2:.2f}** (단위: $1,000)"
    )


# ── 푸터 ────────────────────────────────────────────────────
st.divider()
st.caption("📦 Stack: pandas · scikit-learn · matplotlib · seaborn · Streamlit")