import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  AreaChart,
  BarChart3,
  Box,
  CircleDot,
  Clipboard,
  Download,
  FileJson,
  Grid3X3,
  Loader2,
  PieChart,
  RefreshCcw,
  ScatterChart,
  Sigma,
  SlidersHorizontal,
  TrendingUp
} from "lucide-react";
import { fetchVisualization } from "../services/api";
import type { ApiJson, DataProfile } from "../types";

interface Props {
  profile: DataProfile | null;
}

type ChartType =
  | "scatter"
  | "line"
  | "bar"
  | "area"
  | "histogram"
  | "box"
  | "violin"
  | "density_heatmap"
  | "pie"
  | "scatter_matrix"
  | "corr_heatmap"
  | "scatter3d";

interface ChartWarning {
  level: "warning" | "error" | string;
  title: string;
  detail: string;
}

interface ChartResponse {
  chart_type: ChartType;
  title?: string;
  n_rows?: number;
  warnings?: ChartWarning[];
  data?: any;
}

const chartOptions: Array<{ id: ChartType; label: string; icon: typeof ScatterChart; need: string }> = [
  { id: "scatter", label: "散点图", icon: ScatterChart, need: "趋势线 / 悬停 / 气泡" },
  { id: "line", label: "折线图", icon: TrendingUp, need: "多 Y 轴 / 聚合" },
  { id: "bar", label: "柱状图", icon: BarChart3, need: "聚合 / 分组 / 方向" },
  { id: "area", label: "面积图", icon: AreaChart, need: "多序列 / 堆叠" },
  { id: "histogram", label: "直方图", icon: BarChart3, need: "分箱 / 归一化" },
  { id: "box", label: "箱线图", icon: Box, need: "分组分布" },
  { id: "violin", label: "小提琴图", icon: SlidersHorizontal, need: "密度 + 箱线" },
  { id: "density_heatmap", label: "二维密度热力图", icon: Grid3X3, need: "分箱热力" },
  { id: "pie", label: "饼图 / 环形图", icon: PieChart, need: "Top N / 环形" },
  { id: "scatter_matrix", label: "成对关系图", icon: Grid3X3, need: "2-6 个数值列" },
  { id: "corr_heatmap", label: "相关性热力图", icon: Sigma, need: "标注 / 色阶" },
  { id: "scatter3d", label: "3D 散点图", icon: CircleDot, need: "真实 3D 交互" }
];

const aggOptions = [
  { value: "none", label: "无" },
  { value: "mean", label: "均值" },
  { value: "sum", label: "求和" },
  { value: "count", label: "计数" },
  { value: "median", label: "中位数" },
  { value: "min", label: "最小值" },
  { value: "max", label: "最大值" }
];

const templates = ["plotly_white", "plotly", "ggplot2", "seaborn", "simple_white", "presentation", "plotly_dark"];
const colorScales = ["Viridis", "Plasma", "Inferno", "Magma", "Blues", "Reds", "Greens", "Turbo", "Hot", "Jet", "RdBu"];
const palette = ["#a4512a", "#34785a", "#5b6f91", "#a76b16", "#7d5a8a", "#52796f", "#b94a43", "#6f6a60"];

function first(values: string[], fallback = "") {
  return values[0] ?? fallback;
}

function distinctAt(values: string[], index: number, avoid = "") {
  return values.find((value, i) => i >= index && value !== avoid) ?? values.find((value) => value !== avoid) ?? first(values);
}

function valueText(value: unknown) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function colorFor(label: unknown) {
  const text = String(label ?? "");
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) hash = (hash * 31 + text.charCodeAt(i)) % 997;
  return palette[hash % palette.length];
}

function numberOrNull(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function linearTrend(points: Array<Record<string, unknown>>) {
  const rows = points
    .map((point) => ({ x: numberOrNull(point.x), y: numberOrNull(point.y) }))
    .filter((point): point is { x: number; y: number } => point.x !== null && point.y !== null);
  if (rows.length < 2) return null;
  const meanX = rows.reduce((sum, point) => sum + point.x, 0) / rows.length;
  const meanY = rows.reduce((sum, point) => sum + point.y, 0) / rows.length;
  const denom = rows.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
  if (denom === 0) return null;
  const slope = rows.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0) / denom;
  const intercept = meanY - slope * meanX;
  const minX = Math.min(...rows.map((point) => point.x));
  const maxX = Math.max(...rows.map((point) => point.x));
  return { x: [minX, maxX], y: [slope * minX + intercept, slope * maxX + intercept] };
}

function smoothTrend(points: Array<Record<string, unknown>>) {
  const rows = points
    .map((point) => ({ x: numberOrNull(point.x), y: numberOrNull(point.y) }))
    .filter((point): point is { x: number; y: number } => point.x !== null && point.y !== null)
    .sort((a, b) => a.x - b.x);
  if (rows.length < 5) return null;
  const windowSize = Math.max(3, Math.min(31, Math.round(Math.sqrt(rows.length))));
  const half = Math.floor(windowSize / 2);
  const smoothed = rows.map((row, index) => {
    const slice = rows.slice(Math.max(0, index - half), Math.min(rows.length, index + half + 1));
    return {
      x: row.x,
      y: slice.reduce((sum, item) => sum + item.y, 0) / slice.length
    };
  });
  return { x: smoothed.map((row) => row.x), y: smoothed.map((row) => row.y) };
}

function hoverText(point: Record<string, unknown>) {
  const hover = point.hover;
  if (!hover || typeof hover !== "object") return "";
  return Object.entries(hover as Record<string, unknown>)
    .map(([key, value]) => `${key}: ${valueText(value)}`)
    .join("<br>");
}

function rowsFromChart(result: ChartResponse | null) {
  if (!result?.data) return { columns: [] as string[], rows: [] as Array<Record<string, unknown>> };
  const data = result.data;
  if (Array.isArray(data.points)) {
    return { columns: ["x", "y", "z", "color", "size"].filter((column) => data.points.some((row: any) => row[column] !== undefined)), rows: data.points };
  }
  if (Array.isArray(data.bars)) return { columns: ["label", "value", "group"], rows: data.bars };
  if (Array.isArray(data.bins)) return { columns: ["label", "start", "end", "count"], rows: data.bins };
  if (Array.isArray(data.slices)) return { columns: ["label", "value"], rows: data.slices };
  if (Array.isArray(data.cells)) return { columns: ["x", "y", "value"], rows: data.cells };
  if (Array.isArray(data.groups)) return { columns: ["label", "lower", "q1", "median", "q3", "upper"], rows: data.groups };
  if (Array.isArray(data.series)) {
    const rows = data.series.flatMap((series: any) => (series.points ?? []).map((point: any) => ({ series: series.name, ...point })));
    return { columns: ["series", "x", "y"], rows };
  }
  return { columns: [], rows: [] };
}

function safeFileName(value: string) {
  const cleaned = value.trim().replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "-");
  return cleaned || "chart";
}

function csvCell(value: unknown) {
  const text = valueText(value);
  if (/[",\r\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function downloadTextFile(fileName: string, text: string, type = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function chartRowsToCsv(result: ChartResponse | null) {
  const { columns, rows } = rowsFromChart(result);
  if (!columns.length || !rows.length) return "";
  return [
    columns.map(csvCell).join(","),
    ...rows.map((row: Record<string, unknown>) => columns.map((column) => csvCell(row[column])).join(","))
  ].join("\n");
}

function ChartDataTable({ result }: { result: ChartResponse | null }) {
  const { columns, rows } = rowsFromChart(result);
  if (!rows.length) return null;
  return (
    <details className="chart-data-panel">
      <summary>查看图表数据 · {rows.length.toLocaleString()} 条</summary>
      <div className="table-wrap compact-table chart-data-table">
        <table className="data-table">
          <thead>
            <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
          </thead>
          <tbody>
            {rows.slice(0, 100).map((row: Record<string, unknown>, rowIndex: number) => (
              <tr key={rowIndex}>
                {columns.map((column) => <td key={column}>{valueText(row[column])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function MultiSelect({
  columns,
  selected,
  onChange,
  emptyText = "暂无可选字段"
}: {
  columns: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  emptyText?: string;
}) {
  if (!columns.length) return <div className="empty-list">{emptyText}</div>;
  return (
    <div className="column-select compact">
      {columns.map((column) => {
        const active = selected.includes(column);
        return (
          <button
            className={active ? "column-choice selected" : "column-choice"}
            key={column}
            type="button"
            onClick={() => onChange(active ? selected.filter((item) => item !== column) : [...selected, column])}
          >
            {column}
          </button>
        );
      })}
    </div>
  );
}

function EmptyChart({ text }: { text: string }) {
  return <div className="chart-empty">{text}</div>;
}

function makeMatrixSubplots(data: any) {
  const columns = (data?.columns ?? []) as string[];
  const pairs = (data?.pairs ?? []) as Array<Record<string, any>>;
  if (!columns.length || !pairs.length) return { traces: [], layout: {} };
  const n = columns.length;
  const gap = 0.018;
  const cell = (1 - gap * (n - 1)) / n;
  const traces: any[] = [];
  const layout: Record<string, any> = {};
  columns.forEach((_x, xIndex) => {
    columns.forEach((_y, yIndex) => {
      const axisIndex = yIndex * n + xIndex + 1;
      const pair = pairs.find((item) => item.x_col === columns[xIndex] && item.y_col === columns[yIndex]);
      const points = pair?.points ?? [];
      const xAxis = axisIndex === 1 ? "x" : `x${axisIndex}`;
      const yAxis = axisIndex === 1 ? "y" : `y${axisIndex}`;
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: `${columns[xIndex]} / ${columns[yIndex]}`,
        x: points.map((point: any) => point.x),
        y: points.map((point: any) => point.y),
        marker: { size: 4, color: points.map((point: any) => colorFor(point.color ?? "default")), opacity: 0.62 },
        hovertemplate: `${columns[xIndex]}=%{x}<br>${columns[yIndex]}=%{y}<extra></extra>`,
        showlegend: false,
        xaxis: xAxis,
        yaxis: yAxis
      });
      layout[xAxis === "x" ? "xaxis" : `xaxis${axisIndex}`] = {
        domain: [xIndex * (cell + gap), xIndex * (cell + gap) + cell],
        title: yIndex === n - 1 ? columns[xIndex] : "",
        showticklabels: yIndex === n - 1,
        zeroline: false
      };
      layout[yAxis === "y" ? "yaxis" : `yaxis${axisIndex}`] = {
        domain: [1 - (yIndex + 1) * cell - yIndex * gap, 1 - yIndex * (cell + gap)],
        title: xIndex === 0 ? columns[yIndex] : "",
        showticklabels: xIndex === 0,
        zeroline: false
      };
    });
  });
  return { traces, layout };
}

function buildPlotlyFigure(
  result: ChartResponse | null,
  options: {
    title: string;
    template: string;
    height: number;
    trendline: string;
    barMode: string;
    orientation: "v" | "h";
    hole: number;
    colorScale: string;
    showAnnot: boolean;
    marginal: string;
  }
) {
  if (!result) return null;
  const data = result.data ?? {};
  const title = options.title || result.title || "";
  const baseLayout: any = {
    title: { text: title, x: 0.02, xanchor: "left" },
    template: options.template,
    height: options.height,
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 54, r: 24, t: title ? 54 : 24, b: 54 },
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: "#1f1e1a" },
    legend: { orientation: "h", y: -0.18 },
    hovermode: "closest"
  };

  if (result.chart_type === "scatter") {
    const points = (data.points ?? []) as Array<Record<string, unknown>>;
    const groups = new Map<string, Array<Record<string, unknown>>>();
    points.forEach((point) => {
      const key = String(point.color ?? "数据点");
      groups.set(key, [...(groups.get(key) ?? []), point]);
    });
    const traces: any[] = Array.from(groups.entries()).map(([name, rows]) => ({
      type: "scattergl",
      mode: "markers",
      name,
      x: rows.map((point) => point.x),
      y: rows.map((point) => point.y),
      text: rows.map((point) => point.color ?? ""),
      customdata: rows.map((point) => hoverText(point)),
      marker: {
        size: rows.map((point) => Math.max(5, Math.min(28, Number(point.size) || 8))),
        color: colorFor(name),
        opacity: 0.74,
        line: { color: "rgba(255,255,255,.85)", width: 0.8 }
      },
      hovertemplate: `${data.x_label ?? "X"}=%{x}<br>${data.y_label ?? "Y"}=%{y}<br>%{customdata}<extra></extra>`
    }));
    const trend = options.trendline === "ols" ? linearTrend(points) : options.trendline === "smooth" ? smoothTrend(points) : null;
    if (trend) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: options.trendline === "smooth" ? "平滑趋势线" : "OLS 趋势线",
        x: trend.x,
        y: trend.y,
        line: { color: "#843f20", width: 2.5 }
      });
    }
    const layout: Record<string, unknown> = { ...baseLayout, xaxis: { title: data.x_label }, yaxis: { title: data.y_label } };
    if (options.marginal !== "none") {
      const marginalType = options.marginal === "histogram" ? "histogram" : options.marginal;
      traces.push({
        type: marginalType,
        name: "X 边缘分布",
        x: points.map((point) => point.x),
        marker: { color: "rgba(164,81,42,0.42)" },
        yaxis: "y2",
        xaxis: "x",
        showlegend: false
      });
      traces.push({
        type: marginalType,
        name: "Y 边缘分布",
        y: points.map((point) => point.y),
        marker: { color: "rgba(52,120,90,0.42)" },
        xaxis: "x2",
        yaxis: "y",
        orientation: "h",
        showlegend: false
      });
      Object.assign(layout, {
        xaxis: { title: data.x_label, domain: [0, 0.82] },
        yaxis: { title: data.y_label, domain: [0, 0.82] },
        xaxis2: { domain: [0.84, 1], showticklabels: false },
        yaxis2: { domain: [0.84, 1], showticklabels: false },
        bargap: 0.05
      });
    }
    return { data: traces, layout };
  }

  if (result.chart_type === "scatter3d") {
    const points = (data.points ?? []) as Array<Record<string, unknown>>;
    const groups = new Map<string, Array<Record<string, unknown>>>();
    points.forEach((point) => {
      const key = String(point.color ?? "数据点");
      groups.set(key, [...(groups.get(key) ?? []), point]);
    });
    return {
      data: Array.from(groups.entries()).map(([name, rows]) => ({
        type: "scatter3d",
        mode: "markers",
        name,
        x: rows.map((point) => point.x),
        y: rows.map((point) => point.y),
        z: rows.map((point) => point.z),
        text: rows.map((point) => point.color ?? ""),
        customdata: rows.map((point) => hoverText(point)),
        marker: { size: rows.map((point) => Math.max(3, Math.min(16, Number(point.size) || 5))), color: colorFor(name), opacity: 0.78 },
        hovertemplate: `${data.x_label ?? "X"}=%{x}<br>${data.y_label ?? "Y"}=%{y}<br>${data.z_label ?? "Z"}=%{z}<br>%{customdata}<extra></extra>`
      })),
      layout: {
        ...baseLayout,
        scene: { xaxis: { title: data.x_label }, yaxis: { title: data.y_label }, zaxis: { title: data.z_label } },
        margin: { l: 0, r: 0, t: title ? 54 : 12, b: 0 }
      }
    };
  }

  if (result.chart_type === "line" || result.chart_type === "area") {
    const series = (data.series ?? []) as Array<{ name: string; points: Array<{ x: unknown; y: unknown }> }>;
    return {
      data: series.map((item) => ({
        type: "scatter",
        mode: "lines+markers",
        name: item.name,
        x: item.points.map((point) => point.x),
        y: item.points.map((point) => point.y),
        fill: result.chart_type === "area" ? "tozeroy" : undefined,
        line: { width: 2.2 }
      })),
      layout: { ...baseLayout, hovermode: "x unified", xaxis: { title: data.x_label ?? "X" }, yaxis: { title: "Y" } }
    };
  }

  if (result.chart_type === "bar" || result.chart_type === "histogram") {
    const rows = result.chart_type === "bar" ? (data.bars ?? []) : (data.bins ?? []);
    const groupNames = Array.from(new Set(rows.map((row: any) => String(row.group ?? "数据"))));
    return {
      data: groupNames.map((group) => {
        const groupRows = rows.filter((row: any) => String(row.group ?? "数据") === group);
        const labels = groupRows.map((row: any) => row.label);
        const values = groupRows.map((row: any) => row.value ?? row.count);
        return {
        type: "bar",
        name: group,
        orientation: options.orientation,
        x: options.orientation === "h" ? values : labels,
        y: options.orientation === "h" ? labels : values,
        marker: { color: colorFor(group), opacity: 0.86 },
        hovertemplate: options.orientation === "h" ? "%{y}<br>%{x}<extra></extra>" : "%{x}<br>%{y}<extra></extra>"
        };
      }),
      layout: { ...baseLayout, barmode: options.barMode, xaxis: { title: options.orientation === "h" ? data.y_label : data.x_label }, yaxis: { title: options.orientation === "h" ? data.x_label : data.y_label } }
    };
  }

  if (result.chart_type === "box" || result.chart_type === "violin") {
    const groups = (data.groups ?? []) as Array<Record<string, unknown>>;
    return {
      data: groups.map((group) => ({
        type: result.chart_type === "violin" ? "violin" : "box",
        name: String(group.label),
        y: Array.isArray(group.values) ? group.values : undefined,
        q1: Array.isArray(group.values) ? undefined : [group.q1],
        median: Array.isArray(group.values) ? undefined : [group.median],
        q3: Array.isArray(group.values) ? undefined : [group.q3],
        lowerfence: Array.isArray(group.values) ? undefined : [group.lower],
        upperfence: Array.isArray(group.values) ? undefined : [group.upper],
        box: result.chart_type === "violin" ? { visible: true } : undefined,
        meanline: result.chart_type === "violin" ? { visible: true } : undefined,
        points: result.chart_type === "violin" ? "outliers" : undefined,
        boxpoints: result.chart_type === "box" ? "outliers" : undefined,
        marker: { color: colorFor(group.label) },
        hovertemplate: "%{y}<extra></extra>"
      })),
      layout: { ...baseLayout, yaxis: { title: data.y_label ?? "value" } }
    };
  }

  if (result.chart_type === "density_heatmap") {
    const cells = (data.cells ?? []) as Array<Record<string, unknown>>;
    const xCount = Math.max(0, ...cells.map((cell) => Number(cell.x))) + 1;
    const yCount = Math.max(0, ...cells.map((cell) => Number(cell.y))) + 1;
    const z = Array.from({ length: yCount }, () => Array.from({ length: xCount }, () => 0));
    cells.forEach((cell) => {
      z[Number(cell.y)][Number(cell.x)] = Number(cell.value) || 0;
    });
    return {
      data: [{ type: "heatmap", z, colorscale: options.colorScale, hovertemplate: "x bin %{x}<br>y bin %{y}<br>count %{z}<extra></extra>" }],
      layout: { ...baseLayout, xaxis: { title: data.x_label }, yaxis: { title: data.y_label } }
    };
  }

  if (result.chart_type === "corr_heatmap") {
    const columns = (data.columns ?? []) as string[];
    const z = columns.map((y: string) => columns.map((x: string) => {
      const cell = (data.cells ?? []).find((item: any) => item.x === x && item.y === y);
      return Number(cell?.value ?? 0);
    }));
    return {
      data: [{
        type: "heatmap",
        x: columns,
        y: columns,
        z,
        colorscale: options.colorScale,
        zmin: -1,
        zmax: 1,
        text: options.showAnnot ? z.map((row) => row.map((value) => value.toFixed(2))) : undefined,
        texttemplate: options.showAnnot ? "%{text}" : undefined,
        hovertemplate: "%{x} / %{y}<br>corr=%{z:.3f}<extra></extra>"
      }],
      layout: { ...baseLayout, xaxis: { automargin: true }, yaxis: { automargin: true } }
    };
  }

  if (result.chart_type === "pie") {
    const slices = data.slices ?? [];
    return {
      data: [{
        type: "pie",
        labels: slices.map((slice: any) => slice.label),
        values: slices.map((slice: any) => slice.value),
        hole: options.hole,
        textinfo: "label+percent",
        hovertemplate: "%{label}<br>%{value}<br>%{percent}<extra></extra>"
      }],
      layout: { ...baseLayout, margin: { l: 24, r: 24, t: title ? 54 : 24, b: 24 } }
    };
  }

  if (result.chart_type === "scatter_matrix") {
    const matrix = makeMatrixSubplots(data);
    return { data: matrix.traces, layout: { ...baseLayout, ...matrix.layout, showlegend: false, height: Math.max(options.height, 620), margin: { l: 48, r: 20, t: title ? 54 : 24, b: 48 } } };
  }

  return null;
}

interface PlotlyChartHandle {
  downloadPng: (fileName: string) => Promise<void>;
}

const PlotlyChart = forwardRef<PlotlyChartHandle, { result: ChartResponse | null; options: Parameters<typeof buildPlotlyFigure>[1] }>(
function PlotlyChart({ result, options }, forwardedRef) {
  const ref = useRef<HTMLDivElement | null>(null);
  const figure = useMemo(() => buildPlotlyFigure(result, options), [result, options]);
  const [plotlyReady, setPlotlyReady] = useState(false);

  useImperativeHandle(forwardedRef, () => ({
    async downloadPng(fileName: string) {
      if (!ref.current || !figure) throw new Error("当前没有可导出的图表。");
      const module = await import("plotly.js-dist-min");
      const url = await module.default.toImage(ref.current, {
        format: "png",
        width: 1400,
        height: Math.max(720, Number(figure.layout?.height) || 720),
        scale: 2
      });
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = fileName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    }
  }), [figure]);

  useEffect(() => {
    if (!ref.current || !figure) return;
    let cancelled = false;
    let current: HTMLDivElement | null = ref.current;
    import("plotly.js-dist-min").then((module) => {
      if (cancelled || !current) return;
      setPlotlyReady(true);
      void module.default.react(current, figure.data, figure.layout, {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d"]
      });
    });
    return () => {
      cancelled = true;
      if (current) {
        import("plotly.js-dist-min").then((module) => module.default.purge(current as HTMLElement));
      }
      current = null;
    };
  }, [figure]);

  if (!result) return <EmptyChart text="选择图表类型和字段后生成图表。" />;
  if (!figure) return <EmptyChart text="当前图表暂未实现 Plotly 渲染。" />;
  return (
    <>
      {!plotlyReady ? <div className="chart-empty">正在加载 Plotly 图表引擎...</div> : null}
      <div className="plotly-canvas" ref={ref} />
    </>
  );
});

export default function DataVisualization({ profile }: Props) {
  const allCols = profile?.columns ?? [];
  const numericCols = profile?.numeric_cols ?? [];
  const categoricalCols = profile?.categorical_cols ?? [];
  const categoryOrAll = categoricalCols.length ? categoricalCols : allCols;

  const [chartType, setChartType] = useState<ChartType>("scatter");
  const [title, setTitle] = useState("");
  const [template, setTemplate] = useState("plotly_white");
  const [height, setHeight] = useState(560);
  const [xCol, setXCol] = useState(first(numericCols, first(allCols)));
  const [yCol, setYCol] = useState(distinctAt(numericCols, 1, xCol));
  const [zCol, setZCol] = useState(distinctAt(numericCols, 2, yCol));
  const [colorCol, setColorCol] = useState("无");
  const [sizeCol, setSizeCol] = useState("无");
  const [groupCol, setGroupCol] = useState("无");
  const [yCols, setYCols] = useState<string[]>(numericCols.slice(0, 2));
  const [matrixCols, setMatrixCols] = useState<string[]>(numericCols.slice(0, 4));
  const [agg, setAgg] = useState("none");
  const [bins, setBins] = useState(40);
  const [norm, setNorm] = useState("count");
  const [cumulative, setCumulative] = useState(false);
  const [topN, setTopN] = useState(20);
  const [trendline, setTrendline] = useState("none");
  const [marginal, setMarginal] = useState("none");
  const [hoverCols, setHoverCols] = useState<string[]>(allCols.slice(0, 3));
  const [barMode, setBarMode] = useState("group");
  const [orientation, setOrientation] = useState<"v" | "h">("v");
  const [hole, setHole] = useState(0);
  const [colorScale, setColorScale] = useState("Viridis");
  const [showAnnot, setShowAnnot] = useState(true);
  const [result, setResult] = useState<ChartResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const plotlyRef = useRef<PlotlyChartHandle | null>(null);

  useEffect(() => {
    if (!profile) return;
    const x = first(profile.numeric_cols, first(profile.columns));
    setXCol(x);
    setYCol(distinctAt(profile.numeric_cols, 1, x));
    setZCol(distinctAt(profile.numeric_cols, 2, x));
    setYCols(profile.numeric_cols.slice(0, 2));
    setMatrixCols(profile.numeric_cols.slice(0, 4));
    setHoverCols(profile.columns.slice(0, 3));
    setColorCol("无");
    setGroupCol("无");
    setSizeCol("无");
  }, [profile?.session_id]);

  useEffect(() => {
    if (!profile) return;
    const nums = profile.numeric_cols;
    const cats = profile.categorical_cols.length ? profile.categorical_cols : profile.columns;
    const x = first(nums, first(profile.columns));
    const y = distinctAt(nums, 1, x);
    setError("");
    setResult(null);
    if (chartType === "bar") {
      setXCol(first(cats, first(profile.columns)));
      setYCol(x);
      setAgg((current) => (current === "none" ? "mean" : current));
    } else if (chartType === "pie") {
      setXCol(first(cats, first(profile.columns)));
      setYCol("无");
    } else if (chartType === "histogram") {
      setXCol(x);
    } else if (chartType === "box" || chartType === "violin") {
      setYCol(x);
      setGroupCol("无");
    } else if (chartType === "scatter_matrix") {
      setMatrixCols(nums.slice(0, 4));
    } else if (chartType === "corr_heatmap") {
      setMatrixCols(nums.slice(0, 10));
    } else if (chartType === "scatter3d") {
      setXCol(x);
      setYCol(y);
      setZCol(distinctAt(nums, 2, y));
    } else if (chartType === "density_heatmap" || chartType === "scatter") {
      setXCol(x);
      setYCol(y);
    } else if (chartType === "line" || chartType === "area") {
      setXCol("无");
      setYCols(nums.slice(0, 3));
      setAgg("none");
    }
  }, [chartType, profile?.session_id]);

  const selectedChart = useMemo(() => chartOptions.find((item) => item.id === chartType) ?? chartOptions[0], [chartType]);
  const SelectedIcon = selectedChart.icon;
  const plotOptions = useMemo(() => ({ title, template, height, trendline, barMode, orientation, hole, colorScale, showAnnot, marginal }), [title, template, height, trendline, barMode, orientation, hole, colorScale, showAnnot, marginal]);

  const buildPayload = (): ApiJson => {
    const noneToUndefined = (value: string) => (value === "无" ? undefined : value);
    if (chartType === "line" || chartType === "area") {
      return { chart_type: chartType, title, x_col: noneToUndefined(xCol), y_cols: yCols, group_col: noneToUndefined(groupCol), agg };
    }
    if (chartType === "bar") {
      return { chart_type: chartType, title, x_col: xCol, y_col: yCol, color_col: noneToUndefined(colorCol), agg: agg === "none" ? "mean" : agg, top_n: topN };
    }
    if (chartType === "histogram") {
      return { chart_type: chartType, title, col: xCol, group_col: noneToUndefined(groupCol), bins, norm, cumulative };
    }
    if (chartType === "box" || chartType === "violin") {
      return { chart_type: chartType, title, y_col: yCol, x_col: noneToUndefined(groupCol), color_col: noneToUndefined(colorCol) };
    }
    if (chartType === "density_heatmap") {
      return { chart_type: chartType, title, x_col: xCol, y_col: yCol, bins };
    }
    if (chartType === "pie") {
      return { chart_type: chartType, title, names_col: xCol, values_col: noneToUndefined(yCol), top_n: topN };
    }
    if (chartType === "scatter_matrix" || chartType === "corr_heatmap") {
      return { chart_type: chartType, title, cols: matrixCols, color_col: noneToUndefined(colorCol) };
    }
    if (chartType === "scatter3d") {
      return { chart_type: chartType, title, x_col: xCol, y_col: yCol, z_col: zCol, color_col: noneToUndefined(colorCol), size_col: noneToUndefined(sizeCol), hover_cols: hoverCols };
    }
    return { chart_type: chartType, title, x_col: xCol, y_col: yCol, color_col: noneToUndefined(colorCol), size_col: noneToUndefined(sizeCol), hover_cols: hoverCols };
  };

  const generateChart = async () => {
    if (!profile?.session_id) return;
    setLoading(true);
    setError("");
    setActionMessage("");
    try {
      const payload = await fetchVisualization(profile.session_id, buildPayload());
      setResult(payload as unknown as ChartResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "图表生成失败");
    } finally {
      setLoading(false);
    }
  };

  const exportPng = async () => {
    if (!result) return;
    setActionMessage("");
    try {
      await plotlyRef.current?.downloadPng(`${safeFileName(title || selectedChart.label)}.png`);
      setActionMessage("PNG 已导出。");
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : "PNG 导出失败。");
    }
  };

  const exportCsv = () => {
    const csv = chartRowsToCsv(result);
    if (!csv) {
      setActionMessage("当前图表没有可导出的表格数据。");
      return;
    }
    downloadTextFile(`${safeFileName(title || selectedChart.label)}-data.csv`, csv, "text/csv;charset=utf-8");
    setActionMessage("图表数据 CSV 已导出。");
  };

  const copyConfig = async () => {
    const text = JSON.stringify(buildPayload(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setActionMessage("当前图表配置已复制。");
    } catch {
      downloadTextFile(`${safeFileName(title || selectedChart.label)}-config.json`, text, "application/json;charset=utf-8");
      setActionMessage("浏览器剪贴板不可用，已改为下载配置 JSON。");
    }
  };

  useEffect(() => {
    if (!profile?.session_id) return;
    generateChart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profile?.session_id]);

  if (!profile) {
    return (
      <section className="visual-section" aria-label="数据可视化">
        <div className="empty-list">暂无数据。</div>
      </section>
    );
  }

  const renderAxisControls = () => {
    if (chartType === "line" || chartType === "area") {
      return (
        <>
          <label className="form-field">
            <span>X 轴</span>
            <select value={xCol} onChange={(event) => setXCol(event.target.value)}>
              <option value="无">自动索引</option>
              {allCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="form-field full">
            <span>Y 轴，可多选</span>
            <MultiSelect columns={numericCols} selected={yCols} onChange={setYCols} emptyText="当前无数值列" />
          </label>
          <label className="form-field">
            <span>分组 / 颜色</span>
            <select value={groupCol} onChange={(event) => setGroupCol(event.target.value)}>
              <option>无</option>
              {allCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="form-field">
            <span>聚合方式</span>
            <select value={agg} onChange={(event) => setAgg(event.target.value)}>
              {aggOptions.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </label>
        </>
      );
    }

    if (chartType === "histogram") {
      return (
        <>
          <label className="form-field">
            <span>选择数值列</span>
            <select value={xCol} onChange={(event) => setXCol(event.target.value)}>
              {numericCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="form-field">
            <span>分箱数：{bins}</span>
            <input min={5} max={200} type="range" value={bins} onChange={(event) => setBins(Number(event.target.value))} />
          </label>
          <label className="form-field">
            <span>归一化</span>
            <select value={norm} onChange={(event) => setNorm(event.target.value)}>
              <option value="count">计数</option>
              <option value="percent">百分比</option>
              <option value="probability">概率</option>
              <option value="density">密度</option>
            </select>
          </label>
          <label className="form-field">
            <span>分组</span>
            <select value={groupCol} onChange={(event) => setGroupCol(event.target.value)}>
              <option>无</option>
              {allCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="checkbox-line">
            <input checked={cumulative} type="checkbox" onChange={(event) => setCumulative(event.target.checked)} />
            <span>累积</span>
          </label>
        </>
      );
    }

    if (chartType === "box" || chartType === "violin") {
      return (
        <>
          <label className="form-field">
            <span>Y 轴数值列</span>
            <select value={yCol} onChange={(event) => setYCol(event.target.value)}>
              {numericCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="form-field">
            <span>X 轴 / 分组</span>
            <select value={groupCol} onChange={(event) => setGroupCol(event.target.value)}>
              <option>无</option>
              {allCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
        </>
      );
    }

    if (chartType === "pie") {
      return (
        <>
          <label className="form-field">
            <span>类别列</span>
            <select value={xCol} onChange={(event) => setXCol(event.target.value)}>
              {categoryOrAll.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="form-field">
            <span>数值列</span>
            <select value={yCol} onChange={(event) => setYCol(event.target.value)}>
              <option>无</option>
              {numericCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="form-field">
            <span>Top N：{topN}</span>
            <input min={3} max={50} type="range" value={topN} onChange={(event) => setTopN(Number(event.target.value))} />
          </label>
          <label className="form-field">
            <span>空心半径：{hole.toFixed(2)}</span>
            <input min={0} max={0.8} step={0.05} type="range" value={hole} onChange={(event) => setHole(Number(event.target.value))} />
          </label>
        </>
      );
    }

    if (chartType === "scatter_matrix" || chartType === "corr_heatmap") {
      return (
        <>
          <label className="form-field full">
            <span>选择数值列</span>
            <MultiSelect columns={numericCols} selected={matrixCols} onChange={setMatrixCols} emptyText="当前无数值列" />
          </label>
          {chartType === "scatter_matrix" ? (
            <label className="form-field">
              <span>颜色分组</span>
              <select value={colorCol} onChange={(event) => setColorCol(event.target.value)}>
                <option>无</option>
                {allCols.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
          ) : (
            <label className="checkbox-line">
              <input checked={showAnnot} type="checkbox" onChange={(event) => setShowAnnot(event.target.checked)} />
              <span>显示相关系数标注</span>
            </label>
          )}
        </>
      );
    }

    return (
      <>
        <label className="form-field">
          <span>X 轴</span>
          <select value={xCol} onChange={(event) => setXCol(event.target.value)}>
            {(chartType === "bar" ? categoryOrAll : numericCols).map((column) => <option key={column}>{column}</option>)}
          </select>
        </label>
        <label className="form-field">
          <span>Y 轴</span>
          <select value={yCol} onChange={(event) => setYCol(event.target.value)}>
            {numericCols.map((column) => <option key={column}>{column}</option>)}
          </select>
        </label>
        {chartType === "scatter3d" ? (
          <label className="form-field">
            <span>Z 轴</span>
            <select value={zCol} onChange={(event) => setZCol(event.target.value)}>
              {numericCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
        ) : null}
        {chartType === "scatter" ? (
          <>
            <label className="form-field">
              <span>趋势线</span>
              <select value={trendline} onChange={(event) => setTrendline(event.target.value)}>
                <option value="none">无</option>
                <option value="ols">OLS 线性回归</option>
                <option value="smooth">平滑趋势线</option>
              </select>
            </label>
            <label className="form-field">
              <span>边缘分布</span>
              <select value={marginal} onChange={(event) => setMarginal(event.target.value)}>
                <option value="none">无</option>
                <option value="histogram">直方图</option>
                <option value="box">箱线图</option>
                <option value="violin">小提琴图</option>
              </select>
            </label>
          </>
        ) : null}
        {chartType === "bar" ? (
          <>
            <label className="form-field">
              <span>聚合方式</span>
              <select value={agg === "none" ? "mean" : agg} onChange={(event) => setAgg(event.target.value)}>
                {aggOptions.filter((item) => item.value !== "none").map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>方向</span>
              <select value={orientation} onChange={(event) => setOrientation(event.target.value as "v" | "h")}>
                <option value="v">垂直</option>
                <option value="h">水平</option>
              </select>
            </label>
            <label className="form-field">
              <span>柱状模式</span>
              <select value={barMode} onChange={(event) => setBarMode(event.target.value)}>
                <option value="group">分组</option>
                <option value="stack">堆叠</option>
                <option value="relative">相对比例</option>
              </select>
            </label>
            <label className="form-field">
              <span>Top N：{topN}</span>
              <input min={5} max={50} type="range" value={topN} onChange={(event) => setTopN(Number(event.target.value))} />
            </label>
          </>
        ) : null}
        {chartType === "density_heatmap" ? (
          <label className="form-field">
            <span>分箱数：{bins}</span>
            <input min={8} max={60} type="range" value={bins} onChange={(event) => setBins(Number(event.target.value))} />
          </label>
        ) : null}
        {chartType === "scatter" || chartType === "scatter3d" || chartType === "bar" ? (
          <label className="form-field">
            <span>颜色分组</span>
            <select value={colorCol} onChange={(event) => setColorCol(event.target.value)}>
              <option>无</option>
              {allCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
        ) : null}
        {chartType === "scatter" || chartType === "scatter3d" ? (
          <>
            <label className="form-field">
              <span>气泡大小</span>
              <select value={sizeCol} onChange={(event) => setSizeCol(event.target.value)}>
                <option>无</option>
                {numericCols.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            <label className="form-field full">
              <span>悬停信息</span>
              <MultiSelect columns={allCols} selected={hoverCols} onChange={setHoverCols} />
            </label>
          </>
        ) : null}
      </>
    );
  };

  return (
    <section className="visual-workspace" aria-label="数据可视化">
      <div className="visual-toolbar">
        <div>
          <span>当前数据集</span>
          <strong>{profile.n_rows.toLocaleString()} 行 · {profile.n_cols} 列</strong>
        </div>
        <div>
          <span>字段结构</span>
          <strong>{numericCols.length} 数值 · {categoricalCols.length} 类别</strong>
        </div>
        <button className="button primary" type="button" onClick={generateChart} disabled={loading}>
          {loading ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <RefreshCcw size={15} aria-hidden="true" />}
          生成图表
        </button>
      </div>

      <div className="visual-layout">
        <aside className="visual-config" aria-label="图表设置">
          <div className="panel-title">
            <span>
              <SlidersHorizontal size={17} aria-hidden="true" />
              <h2>图表设置</h2>
            </span>
          </div>

          <div className="chart-type-grid">
            {chartOptions.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  className={item.id === chartType ? "chart-type selected" : "chart-type"}
                  key={item.id}
                  type="button"
                  onClick={() => setChartType(item.id)}
                >
                  <Icon size={15} aria-hidden="true" />
                  <span>{item.label}</span>
                  <small>{item.need}</small>
                </button>
              );
            })}
          </div>

          <label className="form-field">
            <span>图表标题</span>
            <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder={selectedChart.label} />
          </label>

          <div className="operation-columns two">
            <label className="form-field">
              <span>配色模板</span>
              <select value={template} onChange={(event) => setTemplate(event.target.value)}>
                {templates.map((item) => <option key={item}>{item}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>色彩映射</span>
              <select value={colorScale} onChange={(event) => setColorScale(event.target.value)}>
                {colorScales.map((item) => <option key={item}>{item}</option>)}
              </select>
            </label>
          </div>

          <label className="form-field">
            <span>图表高度：{height}px</span>
            <input min={360} max={900} step={20} type="range" value={height} onChange={(event) => setHeight(Number(event.target.value))} />
          </label>

          <div className="visual-control-grid">
            {renderAxisControls()}
          </div>
        </aside>

        <article className="visual-result" aria-label="图表结果">
          <div className="panel-title split">
            <span>
              <SelectedIcon size={17} aria-hidden="true" />
              <h2>{title || selectedChart.label}</h2>
            </span>
            <small>{result?.n_rows ? `基于 ${result.n_rows.toLocaleString()} 行数据生成` : "等待生成"}</small>
          </div>

          <div className="chart-actionbar" aria-label="图表操作">
            <button className="button ghost" type="button" onClick={exportPng} disabled={!result || loading}>
              <Download size={15} aria-hidden="true" />
              导出 PNG
            </button>
            <button className="button ghost" type="button" onClick={exportCsv} disabled={!result || loading}>
              <Download size={15} aria-hidden="true" />
              导出 CSV
            </button>
            <button className="button ghost" type="button" onClick={copyConfig} disabled={loading}>
              <Clipboard size={15} aria-hidden="true" />
              复制配置
            </button>
            <button className="button ghost" type="button" onClick={() => downloadTextFile(`${safeFileName(title || selectedChart.label)}-config.json`, JSON.stringify(buildPayload(), null, 2), "application/json;charset=utf-8")} disabled={loading}>
              <FileJson size={15} aria-hidden="true" />
              下载 JSON
            </button>
            {actionMessage ? <span className="chart-action-status">{actionMessage}</span> : null}
          </div>

          {error ? <div className="inline-error">{error}</div> : null}
          {result?.warnings?.length ? (
            <div className="visual-warnings">
              {result.warnings.map((warning, index) => (
                <div className={warning.level === "error" ? "visual-warning error" : "visual-warning"} key={`${warning.title}-${index}`}>
                  <AlertTriangle size={15} aria-hidden="true" />
                  <div>
                    <strong>{warning.title}</strong>
                    <span>{warning.detail}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          <div className="chart-canvas plotly-shell">
            <PlotlyChart ref={plotlyRef} result={result} options={plotOptions} />
          </div>
          <ChartDataTable result={result} />
        </article>
      </div>
    </section>
  );
}
