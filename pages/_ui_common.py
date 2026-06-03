"""Shared Streamlit UI helpers for consistent page layout."""
import html
import streamlit as st


def install_common_styles():
    """Install lightweight shared CSS used across tool pages."""
    st.markdown(
        """
        <style>
        [data-testid="stickyNav"], [data-testid="stSidebarNav"] {display: none;}
        .app-page-header {
            border-bottom: 1px solid #d0d7de;
            padding-bottom: 10px;
            margin-bottom: 14px;
        }
        .app-page-header h1 {
            margin-bottom: 4px;
        }
        .app-page-subtitle {
            color: #57606a;
            font-size: 14px;
        }
        .app-section {
            border-top: 1px solid #e5e7eb;
            padding-top: 14px;
            margin-top: 18px;
            margin-bottom: 8px;
        }
        .app-section-title {
            font-size: 18px;
            font-weight: 650;
            margin-bottom: 2px;
        }
        .app-section-desc {
            color: #57606a;
            font-size: 13px;
        }
        .app-status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin: 8px 0 14px 0;
        }
        .app-status-cell {
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 8px 10px;
            min-height: 58px;
        }
        .app-status-label {
            font-size: 12px;
            color: #6b7280;
            margin-bottom: 2px;
        }
        .app-status-value {
            font-size: 14px;
            overflow-wrap: anywhere;
        }
        .app-kv-panel {
            border: 1px solid #d0d7de;
            border-top-width: 4px;
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
            min-height: 148px;
            margin: 8px 0 12px 0;
        }
        .app-kv-title {
            font-size: 17px;
            font-weight: 680;
            margin-bottom: 3px;
        }
        .app-kv-desc {
            color: #57606a;
            font-size: 13px;
            margin-bottom: 10px;
        }
        .app-kv-grid {
            display: grid;
            grid-template-columns: 92px minmax(0, 1fr);
            gap: 6px 10px;
            font-size: 13px;
        }
        .app-kv-label {
            color: #6b7280;
        }
        .app-kv-value {
            color: #111827;
            overflow-wrap: anywhere;
        }
        .app-prediction-panel {
            border: 1px solid #c4b5fd;
            border-top: 4px solid #7c3aed;
            border-radius: 8px;
            padding: 14px;
            margin: 12px 0 16px 0;
            background: #faf5ff;
        }
        .app-prediction-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .app-prediction-desc {
            color: #57606a;
            font-size: 13px;
            margin-bottom: 10px;
        }
        .app-prediction-main {
            font-size: 24px;
            font-weight: 750;
            margin: 4px 0 12px 0;
            overflow-wrap: anywhere;
        }
        .app-prediction-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin-top: 8px;
        }
        .app-prediction-cell {
            border: 1px solid #ddd6fe;
            border-radius: 6px;
            background: #ffffff;
            padding: 8px 10px;
        }
        .app-prediction-label {
            color: #6b7280;
            font-size: 12px;
            margin-bottom: 2px;
        }
        .app-prediction-value {
            color: #111827;
            font-size: 14px;
            overflow-wrap: anywhere;
        }
        @media (max-width: 820px) {
            .app-status-grid { grid-template-columns: 1fr; }
            .app-kv-grid { grid-template-columns: 1fr; }
            .app-prediction-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle=""):
    install_common_styles()
    subtitle_html = f'<div class="app-page-subtitle">{html.escape(str(subtitle))}</div>' if subtitle else ""
    st.markdown(
        f'<div class="app-page-header">'
        f'<h1>{html.escape(str(title))}</h1>'
        f'{subtitle_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_section_header(title, description=""):
    desc = f'<div class="app-section-desc">{html.escape(str(description))}</div>' if description else ""
    st.markdown(
        f'<div class="app-section">'
        f'<div class="app-section-title">{html.escape(str(title))}</div>'
        f'{desc}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_status_strip(items):
    """Render a three-column status strip.

    items: iterable of (label, value)
    """
    cells = []
    for label, value in items:
        cells.append(
            '<div class="app-status-cell">'
            f'<div class="app-status-label">{html.escape(str(label))}</div>'
            f'<div class="app-status-value">{html.escape(str(value))}</div>'
            '</div>'
        )
    st.markdown(f'<div class="app-status-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def render_notice(title, detail, level="warning"):
    text = f"{title}：{detail}" if detail else str(title)
    if level == "error":
        st.error(text)
    elif level == "info":
        st.info(text)
    elif level == "success":
        st.success(text)
    else:
        st.warning(text)


def render_metric_row(metrics):
    """Render metrics in a stable row.

    metrics: iterable of (label, value)
    """
    metrics = list(metrics)
    if not metrics:
        return
    cols = st.columns(min(len(metrics), 4))
    for idx, (label, value) in enumerate(metrics):
        with cols[idx % len(cols)]:
            st.metric(str(label), value)


def render_kv_panel(title, items, description="", tone="default"):
    """Render a compact key-value panel for page workbenches."""
    tone_colors = {
        "data": "#2563eb",
        "model": "#7c3aed",
        "train": "#0d9488",
        "risk": "#d97706",
        "default": "#d0d7de",
    }
    accent = tone_colors.get(tone, tone_colors["default"])
    desc_html = f'<div class="app-kv-desc">{html.escape(str(description))}</div>' if description else ""
    rows = []
    for label, value in items:
        rows.append(
            '<div class="app-kv-label">'
            f'{html.escape(str(label))}'
            '</div><div class="app-kv-value">'
            f'{html.escape(str(value))}'
            '</div>'
        )
    st.markdown(
        f'<div class="app-kv-panel" style="border-top-color:{accent};">'
        f'<div class="app-kv-title">{html.escape(str(title))}</div>'
        f'{desc_html}'
        f'<div class="app-kv-grid">{"".join(rows)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_prediction_panel(title, main_value, details=None, description=""):
    """Render a persistent prediction result panel."""
    details = details or []
    desc_html = f'<div class="app-prediction-desc">{html.escape(str(description))}</div>' if description else ""
    cells = []
    for label, value in details:
        cells.append(
            '<div class="app-prediction-cell">'
            f'<div class="app-prediction-label">{html.escape(str(label))}</div>'
            f'<div class="app-prediction-value">{html.escape(str(value))}</div>'
            '</div>'
        )
    grid_html = f'<div class="app-prediction-grid">{"".join(cells)}</div>' if cells else ""
    st.markdown(
        f'<div class="app-prediction-panel">'
        f'<div class="app-prediction-title">{html.escape(str(title))}</div>'
        f'{desc_html}'
        f'<div class="app-prediction-main">{html.escape(str(main_value))}</div>'
        f'{grid_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
