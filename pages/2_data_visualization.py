"""Data visualization page — all local Plotly rendering, no backend calls needed."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pages._prepare import render_sidebar, data_uploader
from pages._ui_common import render_notice, render_page_header, render_section_header, render_status_strip


LARGE_POINT_ROWS = 30000
VERY_LARGE_POINT_ROWS = 100000
HIGH_CARDINALITY = 50
VERY_HIGH_CARDINALITY = 200
PIE_MAX_USEFUL_CATEGORIES = 20
GROUP_MAX_USEFUL_CATEGORIES = 30


def _column_cardinality(df, col):
    if not col or col == "无" or col == "⟳ 自动索引" or col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


def _is_id_like(df, col):
    if not col or col not in df.columns or len(df) == 0:
        return False
    unique = _column_cardinality(df, col)
    return unique > HIGH_CARDINALITY and unique / max(len(df), 1) > 0.85


def _warn_visual_risks(warnings):
    if not warnings:
        return
    with st.expander("绘图风险提示", expanded=True):
        for level, title, detail in warnings:
            render_notice(title, detail, level=level)


def _common_large_point_warnings(df, chart_name, color_col="无", size_col="无"):
    warnings = []
    n_rows = len(df)
    if n_rows > VERY_LARGE_POINT_ROWS:
        warnings.append((
            "error",
            f"{chart_name}数据量过大",
            f"当前有 {n_rows:,} 行，直接绘制可能卡顿、遮挡严重且难以读图。建议先抽样、聚合或筛选。",
        ))
    elif n_rows > LARGE_POINT_ROWS:
        warnings.append((
            "warning",
            f"{chart_name}点数较多",
            f"当前有 {n_rows:,} 行，图形可能出现过度遮挡。建议启用聚合、抽样，或改用密度/分箱类图表。",
        ))

    color_unique = _column_cardinality(df, color_col)
    if color_unique > VERY_HIGH_CARDINALITY:
        warnings.append((
            "error",
            "颜色分组类别过多",
            f"「{color_col}」有 {color_unique:,} 个不同取值，图例会失控且颜色不可读。建议换成低基数字段或先合并类别。",
        ))
    elif color_unique > GROUP_MAX_USEFUL_CATEGORIES:
        warnings.append((
            "warning",
            "颜色分组较多",
            f"「{color_col}」有 {color_unique:,} 个类别，颜色区分会变弱。建议只保留 Top 类别或改用筛选。",
        ))

    if size_col != "无" and _is_id_like(df, size_col):
        warnings.append((
            "warning",
            "气泡大小字段疑似 ID",
            f"「{size_col}」几乎每行都不同，用作气泡大小通常没有解释意义。建议选择连续度量列。",
        ))
    return warnings


def _category_warnings(df, col, chart_name, max_useful=GROUP_MAX_USEFUL_CATEGORIES):
    warnings = []
    unique = _column_cardinality(df, col)
    if unique > VERY_HIGH_CARDINALITY:
        warnings.append((
            "error",
            f"{chart_name}类别数过多",
            f"「{col}」有 {unique:,} 个不同取值，直接绘制会生成大量图元和不可读标签。建议先聚合 Top N、分组或筛选。",
        ))
    elif unique > max_useful:
        warnings.append((
            "warning",
            f"{chart_name}类别较多",
            f"「{col}」有 {unique:,} 个类别，标签和图例可能拥挤。建议只展示 Top {max_useful} 或合并长尾类别。",
        ))
    if _is_id_like(df, col):
        warnings.append((
            "warning",
            "字段疑似唯一标识",
            f"「{col}」接近一行一个取值，通常不适合作为类别轴或分组字段。",
        ))
    return warnings

st.set_page_config(page_title="数据可视化", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/2_data_visualization.py")
render_page_header("数据可视化", "基于当前数据集快速生成常用探索图表。")

df = data_uploader(upload_to_backend=False)

if df is not None:
    st.success("数据加载成功！现在可以创建可视化图表。")
    df = df.copy()  # Don't mutate shared session state
    df.columns = df.columns.astype(str)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()

    render_status_strip([
        ("当前数据集", f"{len(df)} 行 · {len(df.columns)} 列"),
        ("数值列", f"{len(numeric_cols)} 列"),
        ("类别列", f"{len(categorical_cols)} 列"),
    ])

    if not numeric_cols:
        st.warning("未检测到数值列 — 部分图表类型将不可用。")

    chart_type = st.selectbox("选择图表类型", [
        "散点图", "折线图", "柱状图", "面积图", "直方图", "箱线图", "小提琴图",
        "二维密度热力图", "饼图 / 环形图", "成对关系图", "相关性热力图", "3D 散点图",
    ])

    config_col, chart_col = st.columns([1, 2.5], gap="medium")

    with config_col:
        render_section_header("图表设置", "选择图表字段、分组和视觉参数。")
        with st.expander("🎨 全局设置", expanded=True):
            chart_title = st.text_input("图表标题", value="")
            color_template = st.selectbox("配色模板", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation"], index=1)
            fig_width = st.slider("宽度", 500, 1400, 900, 50)
            fig_height = st.slider("高度", 300, 900, 550, 50)

    # ── Scatter plot ──
    if chart_type == "散点图":
        with config_col:
            st.subheader("📊 散点图设置")
            if len(numeric_cols) < 2:
                st.warning("散点图需要至少 2 个数值列才有意义，当前只有 1 个数值列，X/Y 轴将相同。")
            x_col = st.selectbox("X 轴", numeric_cols if numeric_cols else all_cols)
            y_col = st.selectbox("Y 轴", numeric_cols if numeric_cols else all_cols, index=min(1, len(numeric_cols) - 1) if numeric_cols else 0)
            color_col = st.selectbox("颜色分组 (可选)", ["无"] + all_cols)
            size_col = st.selectbox("气泡大小 (可选)", ["无"] + numeric_cols)
            hover_cols = st.multiselect("悬停信息 (额外列)", all_cols, default=all_cols[:min(3, len(all_cols))])
            trendline_label = st.selectbox("趋势线", ["无", "OLS 线性回归", "LOWESS 平滑曲线"])
            marginal_label = st.selectbox("边缘分布", ["无", "直方图", "箱线图", "小提琴图"])
        with chart_col:
            warnings = _common_large_point_warnings(df, "散点图", color_col, size_col)
            if trendline_label == "LOWESS 平滑曲线" and len(df) > 10000:
                warnings.append((
                    "warning",
                    "LOWESS 趋势线计算较重",
                    f"当前有 {len(df):,} 行，LOWESS 平滑可能明显变慢。建议先抽样，或改用 OLS 线性回归。",
                ))
            _warn_visual_risks(warnings)
            trendline_map = {"无": None, "OLS 线性回归": "ols", "LOWESS 平滑曲线": "lowess"}
            marginal_map = {"无": None, "直方图": "histogram", "箱线图": "box", "小提琴图": "violin"}
            tl = trendline_map[trendline_label]
            if x_col == y_col and tl is not None:
                st.error(f"X 轴和 Y 轴选择了相同的列「{x_col}」，无法添加趋势线。请选择不同的列，或将趋势线设为「无」。")
            elif x_col == y_col:
                st.warning(f"X 轴和 Y 轴选择了相同的列「{x_col}」，散点图将退化为对角线。建议选择不同的列。")
                fig = px.scatter(df, x=x_col, y=y_col, title=chart_title or f"{x_col} vs {y_col}", template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)
            else:
                kwargs = dict(x=x_col, y=y_col, title=chart_title or f"{x_col} vs {y_col}", template=color_template, width=fig_width, height=fig_height)
                if color_col != "无": kwargs["color"] = color_col
                if size_col != "无": kwargs["size"] = size_col; kwargs["size_max"] = 20
                if hover_cols: kwargs["hover_data"] = {c: True for c in hover_cols}
                if tl: kwargs["trendline"] = tl
                mg = marginal_map[marginal_label]
                if mg: kwargs["marginal_x"] = mg; kwargs["marginal_y"] = mg
                fig = px.scatter(df, **kwargs)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "折线图":
        with config_col:
            st.subheader("📈 折线图设置")
            x_options = ["⟳ 自动索引"] + all_cols
            x_col = st.selectbox("X 轴", x_options)
            y_cols = st.multiselect("Y 轴 (可多选)", numeric_cols, default=numeric_cols[:min(2, len(numeric_cols))])
            group_col = st.selectbox("分组 / 颜色 (可选)", ["无"] + all_cols)
            agg_label = st.selectbox("聚合方式", ["无", "均值", "求和", "计数", "中位数", "最小值", "最大值"])
        with chart_col:
            if y_cols:
                warnings = []
                if agg_label == "无" and len(df) > LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "折线图行数较多",
                        f"当前有 {len(df):,} 行且未聚合，折线可能过密、渲染变慢。建议按时间/类别聚合后再绘制。",
                    ))
                warnings.extend(_category_warnings(df, group_col, "折线图分组"))
                if x_col != "⟳ 自动索引" and agg_label == "无" and _is_id_like(df, x_col):
                    warnings.append((
                        "warning",
                        "X 轴疑似唯一标识",
                        f"「{x_col}」接近一行一个取值，折线图通常难以呈现趋势。建议选择时间列、排序列或先聚合。",
                    ))
                _warn_visual_risks(warnings)
                x_label = None if x_col == "⟳ 自动索引" else x_col
                color_arg = group_col if group_col != "无" else None
                if x_label is not None and x_label == group_col:
                    st.error(f"X 轴列「{x_col}」与分组列相同，聚合时会产生重复列名导致错误。请选择不同的分组列，或将聚合设为「无」。")
                else:
                    plot_df = df.copy()
                    if agg_label != "无" and x_col != "⟳ 自动索引":
                        agg_map = {"均值": "mean", "求和": "sum", "计数": "count", "中位数": "median", "最小值": "min", "最大值": "max"}
                        group_keys = [x_col]
                        if color_arg: group_keys.append(group_col)
                        plot_df = df.groupby(group_keys, as_index=False)[y_cols].agg(agg_map[agg_label])
                    elif agg_label != "无":
                        st.caption("使用自动索引时，聚合按默认索引分组，效果与不聚合相同。")
                    fig = px.line(plot_df, x=x_label, y=y_cols, color=color_arg, title=chart_title or "折线图", template=color_template, width=fig_width, height=fig_height)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请至少选择一个 Y 轴列。")

    elif chart_type == "柱状图":
        with config_col:
            st.subheader("📊 柱状图设置")
            x_col = st.selectbox("X 轴 (类别)", all_cols)
            y_col = st.selectbox("Y 轴 (数值)", numeric_cols if numeric_cols else all_cols)
            color_col = st.selectbox("颜色 / 分组 (可选)", ["无"] + all_cols)
            agg_label = st.selectbox("聚合方式", ["无", "均值", "求和", "计数", "中位数", "最小值", "最大值"]) if numeric_cols else "无"
            orient_label = st.radio("方向", ["垂直", "水平"], horizontal=True)
            bar_mode_label = st.selectbox("柱状模式", ["分组", "堆叠", "相对比例"])
        with chart_col:
            if not numeric_cols:
                st.error("柱状图需要数值列作为 Y 轴，但数据集中未检测到数值列。")
            elif agg_label != "无" and x_col == color_col:
                st.error(f"X 轴列「{x_col}」与颜色分组列相同，聚合时会产生重复列名导致错误。请选择不同的分组列，或将聚合设为「无」。")
            else:
                warnings = _category_warnings(df, x_col, "柱状图")
                warnings.extend(_category_warnings(df, color_col, "柱状图颜色分组"))
                if agg_label == "无" and len(df) > LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "柱状图未聚合且行数较多",
                        f"当前有 {len(df):,} 行，直接绘制每行柱形可能不可读。建议选择聚合方式或先筛选。",
                    ))
                _warn_visual_risks(warnings)
                plot_df = df.copy()
                if agg_label != "无":
                    agg_map = {"均值": "mean", "求和": "sum", "计数": "count", "中位数": "median", "最小值": "min", "最大值": "max"}
                    group_keys = [x_col]
                    if color_col != "无": group_keys.append(color_col)
                    plot_df = df.groupby(group_keys, as_index=False)[y_col].agg(agg_map[agg_label])
                orient_map = {"垂直": "v", "水平": "h"}
                bar_mode_map = {"分组": "group", "堆叠": "stack", "相对比例": "relative"}
                kwargs = dict(x=x_col, y=y_col, orientation=orient_map[orient_label], title=chart_title or f"{y_col} 按 {x_col}", template=color_template, width=fig_width, height=fig_height)
                if color_col != "无": kwargs["color"] = color_col; kwargs["barmode"] = bar_mode_map[bar_mode_label]
                fig = px.bar(plot_df, **kwargs)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "面积图":
        with config_col:
            st.subheader("🏔️ 面积图设置")
            x_options = ["⟳ 自动索引"] + all_cols
            x_col = st.selectbox("X 轴", x_options)
            y_cols = st.multiselect("Y 轴 (可多选)", numeric_cols, default=numeric_cols[:min(2, len(numeric_cols))])
            group_col = st.selectbox("颜色分组 (可选)", ["无"] + all_cols)
            area_mode_label = st.selectbox("堆叠模式", ["无 (叠加)", "堆叠", "百分比 (100%)"])
        with chart_col:
            if y_cols:
                warnings = []
                if len(df) > LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "面积图点数较多",
                        f"当前有 {len(df):,} 行，面积层叠可能遮挡趋势。建议先聚合或筛选时间范围。",
                    ))
                warnings.extend(_category_warnings(df, group_col, "面积图分组"))
                if x_col != "⟳ 自动索引" and _is_id_like(df, x_col):
                    warnings.append((
                        "warning",
                        "X 轴疑似唯一标识",
                        f"「{x_col}」接近一行一个取值，面积图通常难以表达连续趋势。建议选择时间列或聚合后的字段。",
                    ))
                _warn_visual_risks(warnings)
                x_label = None if x_col == "⟳ 自动索引" else x_col
                group = group_col if group_col != "无" else None
                area_mode_map = {"无 (叠加)": None, "堆叠": "stack", "百分比 (100%)": "percent"}
                kwargs = dict(x=x_label, y=y_cols, title=chart_title or "面积图", template=color_template, width=fig_width, height=fig_height)
                if group: kwargs["color"] = group
                if area_mode_map[area_mode_label]: kwargs["groupnorm"] = area_mode_map[area_mode_label]
                fig = px.area(df, **kwargs)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请至少选择一个 Y 轴列。")

    elif chart_type == "直方图":
        with config_col:
            st.subheader("📊 直方图设置")
            col = st.selectbox("选择列", numeric_cols if numeric_cols else all_cols)
            nbins = st.slider("柱数", 5, 200, 40)
            hist_color = st.selectbox("分组 (可选)", ["无"] + all_cols)
            marginal_label = st.selectbox("边缘图", ["无", "箱线图", "须图", "小提琴图"])
            histnorm_label = st.selectbox("归一化方式", ["计数", "百分比", "概率", "密度"])
            cumulative = st.checkbox("累积")
        with chart_col:
            if not numeric_cols:
                st.error("直方图需要数值列，但数据集中未检测到数值列。请上传包含数值列的数据集。")
            else:
                warnings = []
                if nbins > 120:
                    warnings.append((
                        "warning",
                        "直方图柱数过多",
                        f"当前设置为 {nbins} 个柱，可能造成噪声和标签拥挤。建议先使用 20-80 个柱观察分布。",
                    ))
                warnings.extend(_category_warnings(df, hist_color, "直方图分组"))
                _warn_visual_risks(warnings)
                marginal_map = {"无": None, "箱线图": "box", "须图": "rug", "小提琴图": "violin"}
                histnorm_map = {"计数": None, "百分比": "percent", "概率": "probability", "密度": "density"}
                color_arg = hist_color if hist_color != "无" else None
                fig = px.histogram(df, x=col, nbins=nbins, color=color_arg, marginal=marginal_map[marginal_label], histnorm=histnorm_map[histnorm_label], cumulative=cumulative, title=chart_title or f"直方图: {col}", template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "箱线图":
        with config_col:
            st.subheader("📦 箱线图设置")
            y_col = st.selectbox("Y 轴 (数值)", numeric_cols if numeric_cols else all_cols)
            x_col = st.selectbox("X 轴 / 分组 (可选)", ["无"] + all_cols)
            color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols)
            points_label = st.selectbox("显示数据点", ["仅异常值", "全部点", "疑似异常值", "不显示"])
            orient_label = st.radio("方向", ["垂直", "水平"], horizontal=True)
        with chart_col:
            if not numeric_cols:
                st.error("箱线图需要数值列作为 Y 轴，但数据集中未检测到数值列。")
            else:
                warnings = _category_warnings(df, x_col, "箱线图分组")
                warnings.extend(_category_warnings(df, color_col, "箱线图颜色分组"))
                if points_label == "全部点" and len(df) > LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "箱线图叠加点数过多",
                        f"当前有 {len(df):,} 行且显示全部点，图形可能被点覆盖。建议只显示异常值或关闭数据点。",
                    ))
                _warn_visual_risks(warnings)
                x_arg = x_col if x_col != "无" else None
                color_arg = color_col if color_col != "无" else None
                points_map = {"仅异常值": "outliers", "全部点": "all", "疑似异常值": "suspectedoutliers", "不显示": False}
                orient_map = {"垂直": "v", "水平": "h"}
                fig = px.box(df, x=x_arg, y=y_col, color=color_arg, points=points_map[points_label], orientation=orient_map[orient_label], title=chart_title or f"箱线图: {y_col}", template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "小提琴图":
        with config_col:
            st.subheader("🎻 小提琴图设置")
            y_col = st.selectbox("Y 轴 (数值)", numeric_cols if numeric_cols else all_cols)
            x_col = st.selectbox("X 轴 / 分组 (可选)", ["无"] + all_cols)
            color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols)
            box_inside = st.checkbox("内部显示箱线图", value=True)
            points_label = st.selectbox("显示数据点", ["仅异常值", "全部点", "疑似异常值", "不显示"])
        with chart_col:
            if not numeric_cols:
                st.error("小提琴图需要数值列作为 Y 轴，但数据集中未检测到数值列。")
            else:
                warnings = _category_warnings(df, x_col, "小提琴图分组")
                warnings.extend(_category_warnings(df, color_col, "小提琴图颜色分组"))
                if points_label == "全部点" and len(df) > LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "小提琴图叠加点数过多",
                        f"当前有 {len(df):,} 行且显示全部点，图形可能拥挤并拖慢渲染。建议只显示异常值或关闭数据点。",
                    ))
                _warn_visual_risks(warnings)
                x_arg = x_col if x_col != "无" else None
                color_arg = color_col if color_col != "无" else None
                points_map = {"仅异常值": "outliers", "全部点": "all", "疑似异常值": "suspectedoutliers", "不显示": False}
                fig = px.violin(df, x=x_arg, y=y_col, color=color_arg, box=box_inside, points=points_map[points_label], title=chart_title or f"小提琴图: {y_col}", template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "二维密度热力图":
        with config_col:
            st.subheader("🔥 二维密度设置")
            x_col = st.selectbox("X 轴", numeric_cols if numeric_cols else all_cols)
            idx = min(1, len(numeric_cols) - 1) if len(numeric_cols) > 1 else 0
            y_col = st.selectbox("Y 轴", numeric_cols if numeric_cols else all_cols, index=idx)
            marginal_label = st.selectbox("边缘图", ["无", "直方图", "箱线图", "小提琴图"])
            color_scale = st.selectbox("色彩映射", ["Viridis", "Plasma", "Inferno", "Magma", "Blues", "Reds", "Greens", "Turbo", "Hot", "Jet"])
        with chart_col:
            marginal_map = {"无": None, "直方图": "histogram", "箱线图": "box", "小提琴图": "violin"}
            if x_col == y_col:
                st.error(f"X 轴和 Y 轴选择了相同的列「{x_col}」，无法绘制二维密度热力图。请选择不同的列。")
            else:
                warnings = []
                if len(df) > VERY_LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "二维密度数据量较大",
                        f"当前有 {len(df):,} 行，密度计算可能较慢。建议先抽样或筛选范围。",
                    ))
                _warn_visual_risks(warnings)
                fig = px.density_heatmap(df, x=x_col, y=y_col, marginal_x=marginal_map[marginal_label], marginal_y=marginal_map[marginal_label], color_continuous_scale=color_scale, title=chart_title or f"二维密度: {x_col} vs {y_col}", template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "饼图 / 环形图":
        with config_col:
            st.subheader("🥧 饼图设置")
            names_col = st.selectbox("类别列 (标签)", all_cols)
            values_col = st.selectbox("数值列 (可选 — 默认计数)", ["⟳ 计数"] + numeric_cols)
            hole_size = st.slider("空心半径 (0 = 饼图, > 0 = 环形图)", 0.0, 0.8, 0.0, 0.05)
        with chart_col:
            warnings = []
            unique_names = _column_cardinality(df, names_col)
            if unique_names > VERY_HIGH_CARDINALITY:
                warnings.append((
                    "error",
                    "饼图不适合当前类别列",
                    f"「{names_col}」有 {unique_names:,} 个类别，饼图会产生大量碎片且几乎不可解释。建议改用柱状图、Top N 排名或分组聚合。",
                ))
            elif unique_names > PIE_MAX_USEFUL_CATEGORIES:
                warnings.append((
                    "warning",
                    "饼图类别过多",
                    f"「{names_col}」有 {unique_names:,} 个类别，超过饼图通常可读范围。建议只展示 Top {PIE_MAX_USEFUL_CATEGORIES}，其余合并为“其他”。",
                ))
            if _is_id_like(df, names_col):
                warnings.append((
                    "warning",
                    "饼图标签疑似唯一标识",
                    f"「{names_col}」接近一行一个取值，用作饼图标签会产生大量无意义切片。建议选择低基数类别列。",
                ))
            if values_col != "⟳ 计数" and unique_names < len(df):
                warnings.append((
                    "warning",
                    "饼图标签存在重复",
                    f"「{names_col}」有重复标签，直接按原始行绘制可能产生同名切片。建议先按类别聚合「{values_col}」。",
                ))
            _warn_visual_risks(warnings)
            if values_col == "⟳ 计数":
                vc = df[names_col].value_counts().reset_index()
                values_name = "count"
                while values_name == names_col: values_name = "_" + values_name
                vc.columns = [names_col, values_name]
                fig = px.pie(vc, names=names_col, values=values_name, hole=hole_size, title=chart_title or f"饼图: {names_col}", template=color_template, width=fig_width, height=fig_height)
            else:
                fig = px.pie(df, names=names_col, values=values_col, hole=hole_size, title=chart_title or f"饼图: {names_col}", template=color_template, width=fig_width, height=fig_height)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "成对关系图":
        with config_col:
            st.subheader("🔮 成对关系设置")
            defaults = numeric_cols[:min(4, len(numeric_cols))] if numeric_cols else all_cols[:min(4, len(all_cols))]
            matrix_cols = st.multiselect("选择列 (建议 2-6 个)", numeric_cols if numeric_cols else all_cols, default=defaults)
            color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols)
        with chart_col:
            if not numeric_cols:
                st.error("成对关系图需要数值列，但数据集中未检测到数值列。")
            elif len(matrix_cols) >= 2:
                warnings = []
                if len(matrix_cols) > 6:
                    warnings.append((
                        "warning",
                        "成对关系图列数较多",
                        f"当前选择 {len(matrix_cols)} 列，会生成 {len(matrix_cols) ** 2} 个子图。建议控制在 2-6 列。",
                    ))
                if len(df) > LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "成对关系图数据量较大",
                        f"当前有 {len(df):,} 行，散点矩阵会渲染大量点。建议先抽样或筛选。",
                    ))
                warnings.extend(_category_warnings(df, color_col, "成对关系图颜色分组"))
                _warn_visual_risks(warnings)
                color_arg = color_col if color_col != "无" else None
                fig = px.scatter_matrix(df, dimensions=matrix_cols, color=color_arg, title=chart_title or "成对关系图", template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请至少选择 2 个列。")

    elif chart_type == "相关性热力图":
        with config_col:
            st.subheader("🌡️ 热力图设置")
            if numeric_cols:
                st.caption(f"共 {len(numeric_cols)} 个数值列可用")
                show_annot = st.checkbox("显示数值标注", value=True)
                cmap = st.selectbox("色彩映射", ["RdBu", "RdBu_r", "Viridis", "Blues", "Reds", "coolwarm", "Spectral"])
            else:
                st.warning("需要至少 2 个数值列。")
        with chart_col:
            if len(numeric_cols) >= 2:
                warnings = []
                if len(numeric_cols) > 80:
                    warnings.append((
                        "error",
                        "相关性矩阵列数过多",
                        f"当前有 {len(numeric_cols)} 个数值列，会生成 {len(numeric_cols) ** 2:,} 个格子。建议先选择关键字段或做特征筛选。",
                    ))
                elif len(numeric_cols) > 30:
                    warnings.append((
                        "warning",
                        "相关性热力图列数较多",
                        f"当前有 {len(numeric_cols)} 个数值列，标签会拥挤。建议先选择重点字段。",
                    ))
                if show_annot and len(numeric_cols) > 15:
                    warnings.append((
                        "warning",
                        "数值标注可能遮挡",
                        f"当前有 {len(numeric_cols)} 个数值列，显示标注会让格子文字重叠。建议关闭数值标注。",
                    ))
                _warn_visual_risks(warnings)
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f" if show_annot else False, aspect="auto", color_continuous_scale=cmap, title=chart_title or "相关性热力图", template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("数值列不足，无法绘制相关性热力图 (需要 ≥ 2)。")

    elif chart_type == "3D 散点图":
        with config_col:
            st.subheader("🧊 3D 散点图设置")
            if len(numeric_cols) >= 3:
                x_col = st.selectbox("X 轴", numeric_cols, key="3dx")
                y_col = st.selectbox("Y 轴", numeric_cols, key="3dy", index=min(1, len(numeric_cols) - 1))
                z_col = st.selectbox("Z 轴", numeric_cols, key="3dz", index=min(2, len(numeric_cols) - 1))
                color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols, key="3dc")
                size_col = st.selectbox("气泡大小 (可选)", ["无"] + numeric_cols, key="3ds")
            else:
                st.warning("需要至少 3 个数值列。")
        with chart_col:
            if len(numeric_cols) >= 3:
                warnings = _common_large_point_warnings(df, "3D 散点图", color_col, size_col)
                if len(df) > LARGE_POINT_ROWS:
                    warnings.append((
                        "warning",
                        "3D 散点交互可能卡顿",
                        f"当前有 {len(df):,} 行，3D 旋转和悬停响应可能变慢。建议先抽样到几千行以内。",
                    ))
                _warn_visual_risks(warnings)
                if len({x_col, y_col, z_col}) < 3:
                    dup_cols = [c for c in [x_col, y_col, z_col] if [x_col, y_col, z_col].count(c) > 1]
                    st.error(f"X/Y/Z 轴存在重复的列「{set(dup_cols)}」，请为每个轴选择不同的列。")
                else:
                    kwargs = dict(x=x_col, y=y_col, z=z_col, title=chart_title or f"3D: {x_col} × {y_col} × {z_col}", template=color_template, width=fig_width, height=fig_height)
                    if color_col != "无": kwargs["color"] = color_col
                    if size_col != "无": kwargs["size"] = size_col; kwargs["size_max"] = 15
                    fig = px.scatter_3d(df, **kwargs)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("数值列不足，无法绘制 3D 散点图 (需要 ≥ 3)。")
else:
    st.warning("请上传数据集以开始可视化。支持 CSV 和 Excel 格式。")
