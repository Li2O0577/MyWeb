import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pages._prepare import render_sidebar, data_uploader

# 1. Page config
st.set_page_config(page_title="数据可视化", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/2_data_visualization.py")
st.title("📈 数据可视化")

# 2. 数据加载
df = data_uploader()

if df is not None:
    st.success("数据加载成功！现在可以创建可视化图表。")

    df.columns = df.columns.astype(str)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()

    if not numeric_cols:
        st.warning("未检测到数值列 — 部分图表类型将不可用。")

    # 3. 图表类型选择
    chart_type = st.selectbox("选择图表类型", [
        "散点图",
        "折线图",
        "柱状图",
        "面积图",
        "直方图",
        "箱线图",
        "小提琴图",
        "二维密度热力图",
        "饼图 / 环形图",
        "成对关系图",
        "相关性热力图",
        "3D 散点图",
    ])

    # 4. 布局分栏
    config_col, chart_col = st.columns([1, 2.5], gap="medium")

    # === 全局设置 ===
    with config_col:
        st.subheader("⚙️ 图表设置")
        with st.expander("🎨 全局设置", expanded=True):
            chart_title = st.text_input("图表标题", value="")
            color_template = st.selectbox(
                "配色模板",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation"],
                index=1
            )
            fig_width = st.slider("宽度", 500, 1400, 900, 50)
            fig_height = st.slider("高度", 300, 900, 550, 50)

    # === 各图表类型的设置与渲染 ===

    # ---------- 1. 散点图 ----------
    if chart_type == "散点图":
        with config_col:
            st.subheader("📊 散点图设置")
            x_col = st.selectbox("X 轴", numeric_cols if numeric_cols else all_cols)
            y_col = st.selectbox("Y 轴", numeric_cols if numeric_cols else all_cols,
                                 index=min(1, len(numeric_cols) - 1) if numeric_cols else 0)
            color_col = st.selectbox("颜色分组 (可选)", ["无"] + all_cols)
            size_col = st.selectbox("气泡大小 (可选)", ["无"] + numeric_cols)
            hover_cols = st.multiselect("悬停信息 (额外列)", all_cols,
                                        default=all_cols[:min(3, len(all_cols))])
            trendline_label = st.selectbox("趋势线", ["无", "OLS 线性回归", "LOWESS 平滑曲线"])
            marginal_label = st.selectbox("边缘分布", ["无", "直方图", "箱线图", "小提琴图"])

        with chart_col:
            trendline_map = {"无": None, "OLS 线性回归": "ols", "LOWESS 平滑曲线": "lowess"}
            marginal_map = {"无": None, "直方图": "histogram", "箱线图": "box", "小提琴图": "violin"}

            kwargs = dict(x=x_col, y=y_col,
                          title=chart_title or f"{x_col} vs {y_col}",
                          template=color_template, width=fig_width, height=fig_height)
            if color_col != "无":
                kwargs["color"] = color_col
            if size_col != "无":
                kwargs["size"] = size_col
                kwargs["size_max"] = 20
            if hover_cols:
                kwargs["hover_data"] = {c: True for c in hover_cols}
            tl = trendline_map[trendline_label]
            if tl:
                kwargs["trendline"] = tl
            mg = marginal_map[marginal_label]
            if mg:
                kwargs["marginal_x"] = mg
                kwargs["marginal_y"] = mg

            fig = px.scatter(df, **kwargs)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 2. 折线图 ----------
    elif chart_type == "折线图":
        with config_col:
            st.subheader("📈 折线图设置")
            x_options = ["⟳ 自动索引"] + all_cols
            x_col = st.selectbox("X 轴", x_options)
            y_cols = st.multiselect("Y 轴 (可多选)", numeric_cols,
                                    default=numeric_cols[:min(2, len(numeric_cols))])
            group_col = st.selectbox("分组 / 颜色 (可选)", ["无"] + all_cols)
            agg_label = st.selectbox("聚合方式 (按 X 轴对 Y 聚合)",
                                     ["无", "均值", "求和", "计数", "中位数", "最小值", "最大值"])

        with chart_col:
            if y_cols:
                plot_df = df.copy()
                x_label = None if x_col == "⟳ 自动索引" else x_col
                color_arg = group_col if group_col != "无" else None

                if agg_label != "无" and x_col != "⟳ 自动索引":
                    agg_map = {"均值": "mean", "求和": "sum", "计数": "count",
                               "中位数": "median", "最小值": "min", "最大值": "max"}
                    group_keys = [x_col]
                    if color_arg:
                        group_keys.append(group_col)
                    plot_df = df.groupby(group_keys, as_index=False)[y_cols].agg(agg_map[agg_label])

                fig = px.line(plot_df, x=x_label, y=y_cols, color=color_arg,
                              title=chart_title or "折线图",
                              template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请至少选择一个 Y 轴列。")

    # ---------- 3. 柱状图 ----------
    elif chart_type == "柱状图":
        with config_col:
            st.subheader("📊 柱状图设置")
            x_col = st.selectbox("X 轴 (类别)", all_cols)
            y_col = st.selectbox("Y 轴 (数值)", numeric_cols if numeric_cols else all_cols)
            color_col = st.selectbox("颜色 / 分组 (可选)", ["无"] + all_cols)
            if numeric_cols:
                agg_label = st.selectbox("聚合方式",
                                         ["无", "均值", "求和", "计数", "中位数", "最小值", "最大值"])
            else:
                agg_label = "无"
            orient_label = st.radio("方向", ["垂直", "水平"], horizontal=True)
            bar_mode_label = st.selectbox("柱状模式 (启用颜色分组时生效)", ["分组", "堆叠", "相对比例"])

        with chart_col:
            plot_df = df.copy()
            if agg_label != "无":
                agg_map = {"均值": "mean", "求和": "sum", "计数": "count",
                           "中位数": "median", "最小值": "min", "最大值": "max"}
                group_keys = [x_col]
                if color_col != "无":
                    group_keys.append(color_col)
                plot_df = df.groupby(group_keys, as_index=False)[y_col].agg(agg_map[agg_label])

            orient_map = {"垂直": "v", "水平": "h"}
            bar_mode_map = {"分组": "group", "堆叠": "stack", "相对比例": "relative"}

            kwargs = dict(x=x_col, y=y_col, orientation=orient_map[orient_label],
                          title=chart_title or f"{y_col} 按 {x_col}",
                          template=color_template, width=fig_width, height=fig_height)
            if color_col != "无":
                kwargs["color"] = color_col
                kwargs["barmode"] = bar_mode_map[bar_mode_label]

            fig = px.bar(plot_df, **kwargs)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 4. 面积图 ----------
    elif chart_type == "面积图":
        with config_col:
            st.subheader("🏔️ 面积图设置")
            x_options = ["⟳ 自动索引"] + all_cols
            x_col = st.selectbox("X 轴", x_options)
            y_cols = st.multiselect("Y 轴 (可多选)", numeric_cols,
                                    default=numeric_cols[:min(2, len(numeric_cols))])
            group_col = st.selectbox("颜色分组 (可选)", ["无"] + all_cols)
            area_mode_label = st.selectbox("堆叠模式", ["无 (叠加)", "堆叠", "百分比 (100%)"])

        with chart_col:
            if y_cols:
                x_label = None if x_col == "⟳ 自动索引" else x_col
                group = group_col if group_col != "无" else None
                area_mode_map = {"无 (叠加)": None, "堆叠": "stack", "百分比 (100%)": "percent"}
                grp_mode = area_mode_map[area_mode_label]

                kwargs = dict(x=x_label, y=y_cols,
                              title=chart_title or "面积图",
                              template=color_template, width=fig_width, height=fig_height)
                if group:
                    kwargs["color"] = group
                if grp_mode:
                    kwargs["groupnorm"] = grp_mode

                fig = px.area(df, **kwargs)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请至少选择一个 Y 轴列。")

    # ---------- 5. 直方图 ----------
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
            marginal_map = {"无": None, "箱线图": "box", "须图": "rug", "小提琴图": "violin"}
            histnorm_map = {"计数": None, "百分比": "percent", "概率": "probability", "密度": "density"}
            color_arg = hist_color if hist_color != "无" else None
            marg = marginal_map[marginal_label]

            fig = px.histogram(df, x=col, nbins=nbins, color=color_arg,
                               marginal=marg, histnorm=histnorm_map[histnorm_label],
                               cumulative=cumulative,
                               title=chart_title or f"直方图: {col}",
                               template=color_template, width=fig_width, height=fig_height)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 6. 箱线图 ----------
    elif chart_type == "箱线图":
        with config_col:
            st.subheader("📦 箱线图设置")
            y_col = st.selectbox("Y 轴 (数值)", numeric_cols if numeric_cols else all_cols)
            x_col = st.selectbox("X 轴 / 分组 (可选)", ["无"] + all_cols)
            color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols)
            points_label = st.selectbox("显示数据点",
                                        ["仅异常值", "全部点", "疑似异常值", "不显示"])
            orient_label = st.radio("方向", ["垂直", "水平"], horizontal=True)

        with chart_col:
            x_arg = x_col if x_col != "无" else None
            color_arg = color_col if color_col != "无" else None
            points_map = {"仅异常值": "outliers", "全部点": "all",
                          "疑似异常值": "suspectedoutliers", "不显示": False}
            orient_map = {"垂直": "v", "水平": "h"}

            fig = px.box(df, x=x_arg, y=y_col, color=color_arg,
                         points=points_map[points_label], orientation=orient_map[orient_label],
                         title=chart_title or f"箱线图: {y_col}",
                         template=color_template, width=fig_width, height=fig_height)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 7. 小提琴图 ----------
    elif chart_type == "小提琴图":
        with config_col:
            st.subheader("🎻 小提琴图设置")
            y_col = st.selectbox("Y 轴 (数值)", numeric_cols if numeric_cols else all_cols)
            x_col = st.selectbox("X 轴 / 分组 (可选)", ["无"] + all_cols)
            color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols)
            box_inside = st.checkbox("内部显示箱线图", value=True)
            points_label = st.selectbox("显示数据点",
                                        ["仅异常值", "全部点", "疑似异常值", "不显示"])

        with chart_col:
            x_arg = x_col if x_col != "无" else None
            color_arg = color_col if color_col != "无" else None
            points_map = {"仅异常值": "outliers", "全部点": "all",
                          "疑似异常值": "suspectedoutliers", "不显示": False}

            fig = px.violin(df, x=x_arg, y=y_col, color=color_arg,
                            box=box_inside, points=points_map[points_label],
                            title=chart_title or f"小提琴图: {y_col}",
                            template=color_template, width=fig_width, height=fig_height)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 8. 二维密度热力图 ----------
    elif chart_type == "二维密度热力图":
        with config_col:
            st.subheader("🔥 二维密度设置")
            x_col = st.selectbox("X 轴", numeric_cols if numeric_cols else all_cols)
            idx = min(1, len(numeric_cols) - 1) if len(numeric_cols) > 1 else 0
            y_col = st.selectbox("Y 轴", numeric_cols if numeric_cols else all_cols, index=idx)
            marginal_label = st.selectbox("边缘图", ["无", "直方图", "箱线图", "小提琴图"])
            color_scale = st.selectbox("色彩映射",
                                       ["Viridis", "Plasma", "Inferno", "Magma", "Blues",
                                        "Reds", "Greens", "Turbo", "Hot", "Jet"])

        with chart_col:
            marginal_map = {"无": None, "直方图": "histogram", "箱线图": "box", "小提琴图": "violin"}
            marg = marginal_map[marginal_label]
            fig = px.density_heatmap(df, x=x_col, y=y_col,
                                     marginal_x=marg, marginal_y=marg,
                                     color_continuous_scale=color_scale,
                                     title=chart_title or f"二维密度: {x_col} vs {y_col}",
                                     template=color_template, width=fig_width, height=fig_height)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 9. 饼图 / 环形图 ----------
    elif chart_type == "饼图 / 环形图":
        with config_col:
            st.subheader("🥧 饼图设置")
            names_col = st.selectbox("类别列 (标签)", all_cols)
            values_col = st.selectbox("数值列 (可选 — 默认计数)", ["⟳ 计数"] + numeric_cols)
            hole_size = st.slider("空心半径 (0 = 饼图, > 0 = 环形图)", 0.0, 0.8, 0.0, 0.05)

        with chart_col:
            if values_col == "⟳ 计数":
                vc = df[names_col].value_counts().reset_index()
                values_name = "count"
                while values_name == names_col:
                    values_name = "_" + values_name
                vc.columns = [names_col, values_name]
                fig = px.pie(vc, names=names_col, values=values_name, hole=hole_size,
                             title=chart_title or f"饼图: {names_col}",
                             template=color_template, width=fig_width, height=fig_height)
            else:
                fig = px.pie(df, names=names_col, values=values_col, hole=hole_size,
                             title=chart_title or f"饼图: {names_col}",
                             template=color_template, width=fig_width, height=fig_height)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 10. 成对关系图 ----------
    elif chart_type == "成对关系图":
        with config_col:
            st.subheader("🔮 成对关系设置")
            defaults = numeric_cols[:min(4, len(numeric_cols))] if numeric_cols else all_cols[:min(4, len(all_cols))]
            matrix_cols = st.multiselect("选择列 (建议 2-6 个)",
                                         numeric_cols if numeric_cols else all_cols,
                                         default=defaults)
            color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols)

        with chart_col:
            if len(matrix_cols) >= 2:
                color_arg = color_col if color_col != "无" else None
                fig = px.scatter_matrix(df, dimensions=matrix_cols, color=color_arg,
                                        title=chart_title or "成对关系图",
                                        template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请至少选择 2 个列。")

    # ---------- 11. 相关性热力图 ----------
    elif chart_type == "相关性热力图":
        with config_col:
            st.subheader("🌡️ 热力图设置")
            if numeric_cols:
                st.caption(f"共 {len(numeric_cols)} 个数值列可用")
                show_annot = st.checkbox("显示数值标注", value=True)
                cmap = st.selectbox("色彩映射",
                                    ["RdBu", "RdBu_r", "Viridis", "Blues", "Reds", "coolwarm", "Spectral"])
            else:
                st.warning("需要至少 2 个数值列。")

        with chart_col:
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f" if show_annot else False,
                                aspect="auto", color_continuous_scale=cmap,
                                title=chart_title or "相关性热力图",
                                template=color_template, width=fig_width, height=fig_height)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("数值列不足，无法绘制相关性热力图 (需要 ≥ 2)。")

    # ---------- 12. 3D 散点图 ----------
    elif chart_type == "3D 散点图":
        with config_col:
            st.subheader("🧊 3D 散点图设置")
            if len(numeric_cols) >= 3:
                x_col = st.selectbox("X 轴", numeric_cols, key="3dx")
                y_col = st.selectbox("Y 轴", numeric_cols, key="3dy",
                                     index=min(1, len(numeric_cols) - 1))
                z_col = st.selectbox("Z 轴", numeric_cols, key="3dz",
                                     index=min(2, len(numeric_cols) - 1))
                color_col = st.selectbox("颜色 (可选)", ["无"] + all_cols, key="3dc")
                size_col = st.selectbox("气泡大小 (可选)", ["无"] + numeric_cols, key="3ds")
            else:
                st.warning("需要至少 3 个数值列。")

        with chart_col:
            if len(numeric_cols) >= 3:
                kwargs = dict(x=x_col, y=y_col, z=z_col,
                              title=chart_title or f"3D: {x_col} × {y_col} × {z_col}",
                              template=color_template, width=fig_width, height=fig_height)
                if color_col != "无":
                    kwargs["color"] = color_col
                if size_col != "无":
                    kwargs["size"] = size_col
                    kwargs["size_max"] = 15
                fig = px.scatter_3d(df, **kwargs)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("数值列不足，无法绘制 3D 散点图 (需要 ≥ 3)。")

else:
    st.warning("请上传数据集以开始可视化。支持 CSV 和 Excel 格式。")
