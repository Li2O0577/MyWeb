import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pages._api import is_backend_connected
from pages._prepare import render_sidebar


st.set_page_config(page_title="首页", layout="wide", initial_sidebar_state="collapsed")
render_sidebar("main.py")


MODEL_LABELS = {
    "regression": ("回归预测", "pages/4_regression.py"),
    "classification": ("分类决策", "pages/5_classification.py"),
    "diy_mlp": ("自定义 MLP", "pages/6_diy_mlp.py"),
    "decision_tree": ("决策树", "pages/7_decision_tree.py"),
    "clustering": ("聚类分析", "pages/8_clustering.py"),
}

WORKFLOW = [
    ("数据加载", "上传 CSV/Excel，检查缺失值和异常值。", "pages/1_data_load.py"),
    ("数据可视化", "快速查看分布、相关性和类别结构。", "pages/2_data_visualization.py"),
    ("数据处理", "清洗、编码、标准化和构造新特征。", "pages/3_data_processing.py"),
    ("建模训练", "选择回归、分类、MLP、决策树或聚类。", "pages/4_regression.py"),
    ("大模型分析", "让 Agent 做摘要、解释和图表生成。", "pages/9_llm_analysis.py"),
]

MODEL_CARDS = [
    ("回归预测", "连续值预测", "线性/神经网络回归，查看 R²、MAE、RMSE。", "pages/4_regression.py", "regression"),
    ("分类决策", "类别预测", "分类训练、类别风险提示和单条/批量预测。", "pages/5_classification.py", "classification"),
    ("自定义 MLP", "自定义网络", "配置隐藏层、任务类型和训练参数。", "pages/6_diy_mlp.py", "diy"),
    ("决策树", "可解释模型", "查看树规则、混淆矩阵和回归指标。", "pages/7_decision_tree.py", "tree"),
    ("聚类分析", "无监督学习", "K-means、DBSCAN 和二维聚类可视化。", "pages/8_clustering.py", "cluster"),
    ("大模型分析", "Agent 分析", "调用平台工具完成数据问答和图表生成。", "pages/9_llm_analysis.py", "llm"),
]


def _e(value):
    return html.escape(str(value))


def _install_home_styles():
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        .home-hero {
            border-bottom: 1px solid #d0d7de;
            padding: 10px 0 16px 0;
            margin-bottom: 14px;
        }
        .home-title {
            font-size: 44px;
            font-weight: 720;
            line-height: 1.1;
            margin: 0 0 6px 0;
        }
        .home-subtitle {
            color: #57606a;
            font-size: 17px;
            max-width: 820px;
        }
        .home-status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 18px 0;
        }
        .home-status-card,
        .home-panel,
        .home-step,
        .home-model-card {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            background: #ffffff;
        }
        .home-status-card {
            padding: 10px 12px;
            min-height: 78px;
            border-left-width: 5px;
        }
        .home-status-data {
            border-left-color: #2563eb;
            background: #eff6ff;
        }
        .home-status-backend-ok {
            border-left-color: #059669;
            background: #ecfdf5;
        }
        .home-status-backend-down {
            border-left-color: #dc2626;
            background: #fef2f2;
        }
        .home-status-sync-ok {
            border-left-color: #0d9488;
            background: #f0fdfa;
        }
        .home-status-sync-wait {
            border-left-color: #d97706;
            background: #fffbeb;
        }
        .home-status-model {
            border-left-color: #7c3aed;
            background: #f5f3ff;
        }
        .home-status-label {
            color: #6b7280;
            font-size: 13px;
            margin-bottom: 3px;
        }
        .home-status-value {
            font-size: 24px;
            font-weight: 680;
            overflow-wrap: anywhere;
        }
        .home-status-note {
            color: #6b7280;
            font-size: 13px;
            margin-top: 2px;
        }
        .home-panel {
            padding: 14px 16px;
            min-height: 190px;
            border-top-width: 4px;
        }
        .home-panel-next {
            border-top-color: #2563eb;
        }
        .home-panel-data {
            border-top-color: #0d9488;
        }
        .home-panel-model {
            border-top-color: #7c3aed;
        }
        .home-panel-title {
            font-size: 20px;
            font-weight: 680;
            margin-bottom: 4px;
        }
        .home-panel-desc {
            color: #57606a;
            font-size: 15px;
            margin-bottom: 12px;
        }
        .home-kv {
            display: grid;
            grid-template-columns: 92px minmax(0, 1fr);
            gap: 7px 10px;
            font-size: 15px;
        }
        .home-kv-label {
            color: #6b7280;
        }
        .home-kv-value {
            color: #111827;
            overflow-wrap: anywhere;
        }
        .home-step {
            padding: 11px 12px;
            min-height: 118px;
            background: #f8fafc;
        }
        .home-step-index {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #2563eb;
            color: #ffffff;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .home-step-1 .home-step-index { background: #2563eb; }
        .home-step-2 .home-step-index { background: #0d9488; }
        .home-step-3 .home-step-index { background: #d97706; }
        .home-step-4 .home-step-index { background: #7c3aed; }
        .home-step-5 .home-step-index { background: #047857; }
        .home-step-title,
        .home-model-title {
            font-size: 17px;
            font-weight: 680;
            margin-bottom: 4px;
        }
        .home-step-desc,
        .home-model-desc {
            color: #57606a;
            font-size: 15px;
            line-height: 1.42;
        }
        .home-model-card {
            padding: 12px;
            min-height: 126px;
            margin-bottom: 8px;
            border-top-width: 4px;
            border-top-color: #047857;
            background: #fcfdfd;
        }
        .home-model-card.regression { border-top-color: #2563eb; }
        .home-model-card.classification { border-top-color: #7c3aed; }
        .home-model-card.diy { border-top-color: #d97706; }
        .home-model-card.tree { border-top-color: #0d9488; }
        .home-model-card.cluster { border-top-color: #0891b2; }
        .home-model-card.llm { border-top-color: #047857; }
        .home-model-tag {
            color: #047857;
            font-size: 13px;
            font-weight: 650;
            margin-bottom: 4px;
        }
        .home-section-title {
            font-size: 22px;
            font-weight: 700;
            margin: 22px 0 4px 0;
        }
        .home-section-desc {
            color: #57606a;
            font-size: 15px;
            margin-bottom: 10px;
        }
        .home-empty {
            color: #6b7280;
            font-size: 13px;
            padding: 8px 0;
        }
        @media (max-width: 980px) {
            .home-status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .home-title { font-size: 34px; }
        }
        @media (max-width: 620px) {
            .home-status-grid { grid-template-columns: 1fr; }
            .home-kv { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _model_registry():
    path = Path("backend/models/registry.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _data_profile():
    df = st.session_state.get("main_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "has_data": False,
            "name": "未加载数据",
            "shape": "0 行 · 0 列",
            "numeric": "0",
            "categorical": "0",
            "missing": "0",
            "rows": 0,
            "columns": 0,
        }

    meta = st.session_state.get("session_meta", {}) or {}
    name = meta.get("source_name") or st.session_state.get("_source_file") or "当前数据"
    numeric_count = len(df.select_dtypes(include="number").columns)
    categorical_count = len(df.columns) - numeric_count
    missing_count = int(df.isna().sum().sum())
    return {
        "has_data": True,
        "name": str(name),
        "shape": f"{len(df)} 行 · {len(df.columns)} 列",
        "numeric": str(numeric_count),
        "categorical": str(categorical_count),
        "missing": str(missing_count),
        "rows": len(df),
        "columns": len(df.columns),
    }


def _model_summary(registry):
    total = 0
    active = []
    for model_type, (label, _) in MODEL_LABELS.items():
        item = registry.get(model_type, {}) or {}
        versions = item.get("versions", {}) or {}
        total += len(versions)
        if item.get("active"):
            active.append(label)
    return total, active


def _latest_model(registry):
    latest = None
    latest_label = ""
    for model_type, (label, _) in MODEL_LABELS.items():
        versions = (registry.get(model_type, {}) or {}).get("versions", {}) or {}
        for version_id, meta in versions.items():
            item = {"version_id": version_id, **meta}
            created = item.get("created_at", "")
            if latest is None or created > latest.get("created_at", ""):
                latest = item
                latest_label = label
    return latest_label, latest


def _status_cards(data, backend_ok, model_total, active_models):
    sync_label = "已同步" if st.session_state.get("session_id") and backend_ok else "待同步"
    if not data["has_data"]:
        sync_label = "无数据"
    backend_class = "home-status-backend-ok" if backend_ok else "home-status-backend-down"
    sync_class = "home-status-sync-ok" if sync_label == "已同步" else "home-status-sync-wait"
    cards = [
        ("当前数据集", data["name"], data["shape"] if data["has_data"] else "先上传数据", "home-status-data"),
        ("后端状态", "已连接" if backend_ok else "未连接", "训练/预测依赖 Flask 后端", backend_class),
        ("同步状态", sync_label, f"session {str(st.session_state.get('session_id', ''))[:12]}" if st.session_state.get("session_id") else "暂无 session", sync_class),
        ("模型版本", f"{model_total} 个", "已激活：" + "、".join(active_models[:3]) if active_models else "暂无激活模型", "home-status-model"),
    ]
    cells = []
    for label, value, note, class_name in cards:
        cells.append(
            f'<div class="home-status-card {class_name}">'
            f'<div class="home-status-label">{_e(label)}</div>'
            f'<div class="home-status-value">{_e(value)}</div>'
            f'<div class="home-status-note">{_e(note)}</div>'
            "</div>"
        )
    st.markdown(f'<div class="home-status-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def _next_action(data, backend_ok, model_total):
    if not data["has_data"]:
        return {
            "title": "先上传一个数据集",
            "detail": "没有当前数据时，可视化、处理、训练和 Agent 分析都无法开始。",
            "button": "去数据加载",
            "path": "pages/1_data_load.py",
        }
    if int(data["missing"]) > 0:
        return {
            "title": "建议先处理缺失值",
            "detail": f"当前数据有 {data['missing']} 个缺失值，训练前处理会减少失败和偏差。",
            "button": "去数据处理",
            "path": "pages/3_data_processing.py",
        }
    if not backend_ok:
        return {
            "title": "启动后端后再训练",
            "detail": "数据已在前端可用；训练、预测和 Agent 工具需要 Flask 后端在线。",
            "button": "查看数据",
            "path": "pages/2_data_visualization.py",
        }
    if model_total == 0:
        return {
            "title": "可以开始第一次建模",
            "detail": "数据和后端都已准备好，选择目标列后即可训练模型。",
            "button": "去回归预测",
            "path": "pages/4_regression.py",
        }
    return {
        "title": "继续分析或比较模型",
        "detail": "已有模型版本，可以继续预测、切换版本，或让 Agent 帮你解释结果。",
        "button": "去大模型分析",
        "path": "pages/9_llm_analysis.py",
    }


def _render_info_panels(data, backend_ok, model_total, active_models, latest_label, latest):
    action = _next_action(data, backend_ok, model_total)
    left, middle, right = st.columns([1.2, 1, 1], gap="medium")

    with left:
        st.markdown(
            '<div class="home-panel home-panel-next">'
            '<div class="home-panel-title">下一步建议</div>'
            f'<div class="home-panel-desc">{_e(action["title"])}</div>'
            '<div class="home-kv">'
            '<div class="home-kv-label">原因</div>'
            f'<div class="home-kv-value">{_e(action["detail"])}</div>'
            '<div class="home-kv-label">建议入口</div>'
            f'<div class="home-kv-value">{_e(action["button"])}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        if st.button(action["button"], use_container_width=True):
            st.switch_page(action["path"])

    with middle:
        st.markdown(
            '<div class="home-panel home-panel-data">'
            '<div class="home-panel-title">数据概况</div>'
            '<div class="home-panel-desc">当前工作数据的基础结构。</div>'
            '<div class="home-kv">'
            '<div class="home-kv-label">数据集</div>'
            f'<div class="home-kv-value">{_e(data["name"])}</div>'
            '<div class="home-kv-label">规模</div>'
            f'<div class="home-kv-value">{_e(data["shape"])}</div>'
            '<div class="home-kv-label">数值列</div>'
            f'<div class="home-kv-value">{_e(data["numeric"])}</div>'
            '<div class="home-kv-label">类别列</div>'
            f'<div class="home-kv-value">{_e(data["categorical"])}</div>'
            '<div class="home-kv-label">缺失值</div>'
            f'<div class="home-kv-value">{_e(data["missing"])}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("查看或上传数据", use_container_width=True):
            st.switch_page("pages/1_data_load.py")

    with right:
        latest_text = "暂无模型版本"
        latest_detail = "训练完成后，这里会显示最近模型。"
        if latest:
            latest_text = f"{latest_label} · {latest.get('version_id', '')[:18]}"
            dataset = latest.get("dataset_name") or "unknown"
            created = str(latest.get("created_at", ""))[:16].replace("T", " ")
            latest_detail = f"{dataset} · {created}"
        active_text = "、".join(active_models) if active_models else "暂无"
        st.markdown(
            '<div class="home-panel home-panel-model">'
            '<div class="home-panel-title">模型资产</div>'
            '<div class="home-panel-desc">本地已保存模型版本和当前激活情况。</div>'
            '<div class="home-kv">'
            '<div class="home-kv-label">版本数</div>'
            f'<div class="home-kv-value">{_e(model_total)}</div>'
            '<div class="home-kv-label">激活模型</div>'
            f'<div class="home-kv-value">{_e(active_text)}</div>'
            '<div class="home-kv-label">最近版本</div>'
            f'<div class="home-kv-value">{_e(latest_text)}</div>'
            '<div class="home-kv-label">来源</div>'
            f'<div class="home-kv-value">{_e(latest_detail)}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("管理模型版本", use_container_width=True):
            st.switch_page("pages/4_regression.py")


def _render_workflow():
    st.markdown('<div class="home-section-title">推荐工作流</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-section-desc">从数据导入到模型解释，按这个顺序走最稳定。</div>', unsafe_allow_html=True)
    cols = st.columns(5, gap="small")
    for idx, (title, desc, path) in enumerate(WORKFLOW, start=1):
        with cols[idx - 1]:
            st.markdown(
                f'<div class="home-step home-step-{idx}">'
                f'<div class="home-step-index">{idx}</div>'
                f'<div class="home-step-title">{_e(title)}</div>'
                f'<div class="home-step-desc">{_e(desc)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("进入", key=f"workflow_{idx}", use_container_width=True):
                st.switch_page(path)


def _render_model_cards():
    st.markdown('<div class="home-section-title">功能入口</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-section-desc">常用建模、聚类和大模型分析入口集中在这里。</div>', unsafe_allow_html=True)
    rows = [MODEL_CARDS[:3], MODEL_CARDS[3:]]
    for row_idx, row in enumerate(rows):
        cols = st.columns(3, gap="medium")
        for col, (title, tag, desc, path, class_name) in zip(cols, row):
            with col:
                st.markdown(
                    f'<div class="home-model-card {class_name}">'
                    f'<div class="home-model-tag">{_e(tag)}</div>'
                    f'<div class="home-model-title">{_e(title)}</div>'
                    f'<div class="home-model-desc">{_e(desc)}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("打开", key=f"model_{row_idx}_{title}", use_container_width=True):
                    st.switch_page(path)


_install_home_styles()

registry = _model_registry()
data = _data_profile()
backend_ok = is_backend_connected()
model_total, active_models = _model_summary(registry)
latest_label, latest = _latest_model(registry)

st.markdown(
    '<div class="home-hero">'
    '<div class="home-title">Indeterminate</div>'
    '<div class="home-subtitle">'
    '面向数据加载、清洗、可视化、机器学习训练和大模型分析的一体化工作台。'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

_status_cards(data, backend_ok, model_total, active_models)
_render_info_panels(data, backend_ok, model_total, active_models, latest_label, latest)
_render_workflow()
_render_model_cards()

st.divider()
st.caption("开发者：Jiayang Li")
