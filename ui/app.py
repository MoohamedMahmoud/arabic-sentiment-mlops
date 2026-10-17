from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8000/predict")
API_ROOT = API_URL.rsplit("/", 1)[0]
HEALTH_URL = f"{API_ROOT}/health"
METADATA_URL = f"{API_ROOT}/metadata"

LABELS_AR = {
    "POSITIVE": "إيجابي",
    "NEGATIVE": "سلبي",
    "NEUTRAL": "محايد",
}
LABEL_CLASSES = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
}
SAMPLES = {
    "إيجابي": "الخدمة ممتازة والتجربة كانت رائعة، سأكرر الشراء بالتأكيد.",
    "سلبي": "المنتج سيئ جداً ووصل متأخراً ولن أشتريه مرة أخرى.",
    "محايد": "استخدمتها شهر كامل. الطعم أنقى من الفرنش برس بفرق واضح، بس تحتاج مطحنة زينة.",
}

st.set_page_config(
    page_title="محلّل المشاعر العربية",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 82% 0%, #e7f4f2 0, #f5f8fc 38%, #f5f8fc 100%);
    }
    .block-container { max-width: 1240px; padding-top: 1.7rem; padding-bottom: 3rem; }
    /* Keep the physical column order deterministic: first column on the left. */
    div[data-testid="stHorizontalBlock"] { direction: ltr; gap: 1.35rem; }
    div[data-testid="column"] { direction: rtl; text-align: right; min-width: 0; }
    div[data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
    .hero {
        direction: rtl; text-align: right;
        background: linear-gradient(120deg, #102a43 0%, #174c62 65%, #1b756e 100%);
        color: #fff; padding: 1.8rem 2rem; border-radius: 24px;
        margin-bottom: 1.25rem; box-shadow: 0 16px 40px rgba(16,42,67,.16);
    }
    .hero h1 { color: #fff; font-size: 2.15rem; margin: 0 0 .55rem 0; }
    .hero p { color: #dbeafe; font-size: 1.05rem; margin: 0; line-height: 1.9; }
    .surface {
        direction: rtl; text-align: right; background: rgba(255,255,255,.9);
        border: 1px solid #e4eaf0; border-radius: 18px; padding: 1.25rem;
        box-shadow: 0 8px 24px rgba(16,42,67,.05);
    }
    .muted { color: #60758a; font-size: .9rem; line-height: 1.8; }
    .result-label { font-size: 2rem; font-weight: 800; margin: .15rem 0 .5rem 0; }
    .positive { color: #087f5b; }
    .negative { color: #c2413a; }
    .neutral { color: #9a6700; }
    div[data-testid="stTextArea"] textarea {
        direction: rtl; text-align: right; font-size: 1.08rem; line-height: 1.9;
    }
    div[data-testid="stTextArea"] label {
        direction: rtl; text-align: right; width: 100%; justify-content: flex-start;
    }
    div[data-testid="stButton"] button { direction: rtl; font-weight: 600; }
    div[data-testid="stMetricLabel"] { direction: rtl; text-align: right; }
    div[data-testid="stAlert"] { direction: rtl; text-align: right; }
    div[data-testid="stSidebar"] { direction: rtl; }
    div[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] { text-align: right; }
    pre, code, div[data-testid="stJson"] { direction: ltr; text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "review_text" not in st.session_state:
    st.session_state.review_text = SAMPLES["إيجابي"]
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

with st.sidebar:
    st.markdown("### 💬 محلّل المشاعر العربية")
    st.caption("AraBERT · BentoML · MLOps")
    st.markdown("---")
    st.markdown("**حالة خدمة التوقّع**")
    health_ok = False
    try:
        health_response = requests.get(HEALTH_URL, timeout=3)
        health_body = health_response.json()
        health_ok = health_response.ok and bool(health_body.get("upstream_ok", False))
    except (requests.RequestException, ValueError):
        health_body = {}
    if health_ok:
        st.success("الخدمة متاحة وتستجيب")
    else:
        st.error("الخدمة غير متاحة؛ تحقّق من حاوية BentoML")
    st.markdown("---")
    st.markdown("**عن النتيجة**")
    st.write("يعرض التصنيف المتوقع ودرجة ثقة النموذج وزمن الاستجابة عبر واجهة الخدمة.")
    st.caption("درجة الثقة ليست ضماناً لصحة التوقّع، وليست بالضرورة احتمالاً مُعايراً.")
    with st.expander("تفاصيل الاتصال بالخدمة"):
        st.code(API_URL, language=None)
        if health_body:
            st.json(health_body)
    with st.expander("بيانات النموذج"):
        try:
            metadata_response = requests.get(METADATA_URL, timeout=3)
            metadata_response.raise_for_status()
            st.json(metadata_response.json())
        except (requests.RequestException, ValueError) as exc:
            st.caption(f"تعذّر جلب بيانات النموذج: {exc}")

st.markdown(
    """
    <section class="hero">
      <h1>محلّل المشاعر العربية</h1>
      <p>اكتشف انطباع العملاء من مراجعاتهم العربية. أدخل نص المراجعة في الجهة اليمنى، وستظهر نتيجة التحليل في الجهة اليسرى.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# First Streamlit column is physically on the left; second is physically on the right.
result_col, review_col = st.columns([1.0, 1.25], gap="large")

with review_col:
    st.markdown("### ✍️ نص المراجعة")
    st.caption("أدخل تعليق العميل أو مراجعة المنتج/الخدمة باللغة العربية.")

    sample_cols = st.columns(3)
    for sample_col, (sample_label, sample_text) in zip(
        sample_cols, SAMPLES.items(), strict=True
    ):
        if sample_col.button(f"مثال {sample_label}", use_container_width=True):
            st.session_state.review_text = sample_text
            st.session_state.last_error = None

    review = st.text_area(
        "اكتب مراجعة العميل هنا",
        key="review_text",
        height=220,
        placeholder="مثال: المنتج ممتاز والتوصيل كان سريعاً...",
        help="تجنّب إدخال الأسماء أو أرقام الهواتف أو أي معلومات شخصية.",
    )
    predict_clicked = st.button(
        "حلّل المراجعة", type="primary", use_container_width=True
    )

    if predict_clicked:
        st.session_state.last_error = None
        if not review or not review.strip():
            st.session_state.last_error = (
                "اكتب نص المراجعة أولاً ثم اضغط على زر التحليل."
            )
            st.session_state.last_result = None
        else:
            try:
                started = time.perf_counter()
                response = requests.post(
                    API_URL,
                    json={"texts": [review.strip()]},
                    timeout=180,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                if response.status_code >= 400:
                    st.session_state.last_error = (
                        f"تعذّر تنفيذ التوقّع (HTTP {response.status_code}). "
                        f"{response.text[:400]}"
                    )
                    st.session_state.last_result = None
                else:
                    body: Any = response.json()
                    predictions: Any = body
                    if isinstance(body, dict):
                        predictions = body.get("predictions", body.get("value", body))
                    if isinstance(predictions, dict) and isinstance(
                        predictions.get("value"), list
                    ):
                        predictions = predictions["value"]
                    if isinstance(predictions, dict):
                        predictions = [predictions]
                    if not isinstance(predictions, list) or not predictions:
                        st.session_state.last_error = (
                            "وصل رد غير متوقع من الخدمة؛ راجع سجل الحاوية."
                        )
                        st.session_state.last_result = None
                    elif not isinstance(predictions[0], dict):
                        st.session_state.last_error = "صيغة نتيجة النموذج غير متوقعة."
                        st.session_state.last_result = None
                    else:
                        item = predictions[0]
                        label = str(item.get("label", "UNKNOWN")).upper()
                        confidence = float(item.get("confidence", 0.0))
                        st.session_state.last_result = {
                            "label": label,
                            "confidence": min(max(confidence, 0.0), 1.0),
                            "elapsed_ms": elapsed_ms,
                            "review_text": review.strip(),
                            "details": item,
                        }
            except requests.Timeout:
                st.session_state.last_error = (
                    "انتهت مهلة الطلب؛ تحقّق من حمل الخدمة ثم أعد المحاولة."
                )
                st.session_state.last_result = None
            except requests.RequestException as exc:
                st.session_state.last_error = f"تعذّر الاتصال بخدمة التوقّع: {exc}"
                st.session_state.last_result = None
            except (TypeError, ValueError, KeyError) as exc:
                st.session_state.last_error = f"تعذّرت قراءة استجابة النموذج: {exc}"
                st.session_state.last_result = None

    st.caption("التصنيفات المتاحة: إيجابي، سلبي، ومحايد.")

with result_col:
    st.markdown("### 🎯 نتيجة التحليل")
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    result = st.session_state.last_result
    if result:
        label = result["label"]
        label_ar = LABELS_AR.get(label, "غير معروف")
        label_class = LABEL_CLASSES.get(label, "")
        st.markdown(
            f'<div class="surface"><p class="muted">التصنيف المتوقع</p>'
            f'<p class="result-label {label_class}">{label_ar}</p>'
            f'<p class="muted">رمز التصنيف: {label}</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**نص المراجعة التي تم تحليلها**")
        st.write(result["review_text"])
        st.metric("درجة الثقة", f"{result['confidence']:.1%}")
        st.progress(result["confidence"])
        metric_left, metric_right = st.columns(2)
        metric_left.metric("زمن الاستجابة", f"{result['elapsed_ms']:.0f} مللي ثانية")
        metric_right.metric("عدد المراجعات", "١")
        st.caption(
            "الزمن المعروض هو زمن الطلب من الواجهة إلى واجهة الخدمة، وليس زمن استدلال النموذج وحده."
        )
        with st.expander("تفاصيل الاستجابة التقنية"):
            st.json(result["details"])
        st.caption(
            "تنبيه: قد يخطئ النموذج، خصوصاً مع السخرية أو اللهجات أو المراجعات المختلطة."
        )
    else:
        st.markdown(
            """
            <div class="surface">
              <h4>ستظهر النتيجة هنا</h4>
              <p>١. أدخل مراجعة عربية في الجهة اليمنى.</p>
              <p>٢. اضغط على <b>حلّل المراجعة</b>.</p>
              <p>٣. راجع التصنيف ودرجة الثقة وزمن الاستجابة.</p>
              <hr/>
              <p class="muted">يُصنّف النموذج المراجعة إلى: إيجابي أو سلبي أو محايد.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    "<p class='muted' style='direction:rtl;text-align:right'>"
    "مشروع تعليمي لتطبيق ممارسات MLOps على تحليل المشاعر العربية. "
    "تُستخدم المراجعة المُدخلة لإنتاج التوقّع فقط ضمن هذه الواجهة."
    "</p>",
    unsafe_allow_html=True,
)
