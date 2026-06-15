import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  AreaChart,
  BarChart3,
  Box,
  CircleDot,
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
  { id: "scatter", label: "散点图", icon: ScatterChart, need: "2 个数值列" },
  { id: "line", label: "折线图", icon: TrendingUp, need: "1+ 个数值列" },
  { id: "bar", label: "柱状图", icon: BarChart3, need: "类别 + 数值" },
  { id: "area", label: "面积图", icon: AreaChart, need: "1+ 个数值列" },
  { id: "histogram", label: "直方图", icon: BarChart3, need: "1 个数值列" },
  { id: "box", label: "箱线图", icon: Box, need: "1 个数值列" },
  { id: "violin", label: "小提琴图", icon: SlidersHorizontal, need: "1 个数值列" },
  { id: "density_heatmap", label: "二维密度热力图", icon: Grid3X3, need: "2 个数值列" },
  { id: "pie", label: "饼图 / 环形图", icon: PieChart, need: "类别列" },
  { id: "scatter_matrix", label: "成对关系图", icon: Grid3X3, need: "2-6 个数值列" },
  { id: "corr_heatmap", label: "相关性热力图", icon: Sigma, need: "2+ 个数值列" },
  { id: "scatter3d", label: "3D 散点图", icon: CircleDot, need: "3 个数值列" }
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

const palette = ["#a4512a", "#34785a", "#5b6f91", "#a76b16", "#7d5a8a", "#52796f", "#b94a43", "#6f6a60"];

function first(values: string[], fallback = "") {
  return values[0] ?? fallback;
}

function second(values: string[], fallback = "") {
  return values[1] ?? first(values, fallback);
}

function third(values: string[], fallback = "") {
  return values[2] ?? second(values, fallback);
}

function toNumber(value: unknown) {
  const next = Number(value);
  return Number.isFinite(next) ? next : 0;
}

function scale(value: number, min: number, max: number, start: number, end: number) {
  if (max === min) return (start + end) / 2;
  return start + ((value - min) / (max - min)) * (end - start);
}

function colorFor(label: unknown) {
  const text = String(label ?? "");
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) hash = (hash * 31 + text.charCodeAt(i)) % 997;
  return palette[hash % palette.length];
}

function valueText(value: unknown) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return String(value);
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

function ChartShell({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div className="chart-canvas" aria-label={label}>
      {children}
    </div>
  );
}

function ScatterSvg({ data, threeD = false }: { data: any; threeD?: boolean }) {
  const points = (data?.points ?? []) as Array<Record<string, unknown>>;
  if (!points.length) return <EmptyChart text="没有可绘制的点。" />;
  const xs = points.map((point) => toNumber(point.x) + (threeD ? toNumber(point.z) * 0.28 : 0));
  const ys = points.map((point) => toNumber(point.y) - (threeD ? toNumber(point.z) * 0.18 : 0));
  const sizes = points.map((point) => Math.max(3, Math.min(10, toNumber(point.size) || 4)));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return (
    <ChartShell label={threeD ? "3D 散点图投影" : "散点图"}>
      <svg viewBox="0 0 720 360" className="chart-svg rich">
        <line x1="54" y1="312" x2="676" y2="312" />
        <line x1="54" y1="42" x2="54" y2="312" />
        {points.map((point, index) => {
          const x = scale(xs[index], minX, maxX, 60, 666);
          const y = scale(ys[index], minY, maxY, 304, 48);
          return <circle key={index} cx={x} cy={y} r={sizes[index]} fill={colorFor(point.color ?? "default")} opacity="0.76" />;
        })}
        <text x="54" y="342">{data?.x_label ?? "X"}</text>
        <text x="18" y="48">{threeD ? `${data?.y_label ?? "Y"} / ${data?.z_label ?? "Z"}` : data?.y_label ?? "Y"}</text>
      </svg>
    </ChartShell>
  );
}

function LineAreaSvg({ data, area = false }: { data: any; area?: boolean }) {
  const series = (data?.series ?? []) as Array<{ name: string; points: Array<{ x: unknown; y: unknown }> }>;
  const valid = series.filter((item) => item.points?.length);
  if (!valid.length) return <EmptyChart text="没有可绘制的序列。" />;
  const yValues = valid.flatMap((item) => item.points.map((point) => toNumber(point.y)));
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const maxLen = Math.max(...valid.map((item) => item.points.length));
  return (
    <ChartShell label={area ? "面积图" : "折线图"}>
      <svg viewBox="0 0 720 360" className="chart-svg rich">
        <line x1="54" y1="312" x2="676" y2="312" />
        <line x1="54" y1="42" x2="54" y2="312" />
        {valid.slice(0, 8).map((item, seriesIndex) => {
          const path = item.points
            .map((point, index) => {
              const x = scale(index, 0, Math.max(1, maxLen - 1), 60, 666);
              const y = scale(toNumber(point.y), minY, maxY, 304, 48);
              return `${index === 0 ? "M" : "L"} ${x} ${y}`;
            })
            .join(" ");
          const fillPath = `${path} L 666 312 L 60 312 Z`;
          const stroke = palette[seriesIndex % palette.length];
          return (
            <g key={item.name}>
              {area ? <path d={fillPath} fill={stroke} opacity="0.12" /> : null}
              <path d={path} fill="none" stroke={stroke} strokeWidth="2.2" />
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        {valid.slice(0, 8).map((item, index) => (
          <span key={item.name}><i style={{ background: palette[index % palette.length] }} />{item.name}</span>
        ))}
      </div>
    </ChartShell>
  );
}

function BarSvg({ bars, label = "柱状图" }: { bars: Array<Record<string, unknown>>; label?: string }) {
  if (!bars.length) return <EmptyChart text="没有可绘制的柱。" />;
  const max = Math.max(1, ...bars.map((item) => Math.abs(toNumber(item.value ?? item.count))));
  const barWidth = 600 / Math.max(1, bars.length);
  return (
    <ChartShell label={label}>
      <svg viewBox="0 0 720 360" className="chart-svg rich">
        <line x1="54" y1="312" x2="676" y2="312" />
        {bars.map((item, index) => {
          const value = toNumber(item.value ?? item.count);
          const height = Math.abs(value / max) * 236;
          const x = 60 + index * barWidth;
          const y = 312 - height;
          return (
            <g key={`${item.label}-${index}`}>
              <rect x={x} y={y} width={Math.max(6, barWidth - 7)} height={height} rx="4" fill={colorFor(item.group ?? item.label)} />
              {barWidth > 28 ? <text x={x} y="334">{String(item.label ?? "").slice(0, 10)}</text> : null}
            </g>
          );
        })}
      </svg>
    </ChartShell>
  );
}

function BoxSvg({ data, violin = false }: { data: any; violin?: boolean }) {
  const groups = (data?.groups ?? []) as Array<Record<string, any>>;
  if (!groups.length) return <EmptyChart text="没有可绘制的分布。" />;
  const allValues = groups.flatMap((group) => [group.lower, group.q1, group.median, group.q3, group.upper].map(toNumber));
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const slot = 620 / Math.max(1, groups.length);
  return (
    <ChartShell label={violin ? "小提琴图" : "箱线图"}>
      <svg viewBox="0 0 720 360" className="chart-svg rich">
        <line x1="54" y1="312" x2="676" y2="312" />
        <line x1="54" y1="42" x2="54" y2="312" />
        {groups.map((group, index) => {
          const cx = 70 + index * slot + slot / 2;
          const lowerY = scale(toNumber(group.lower), min, max, 304, 52);
          const q1Y = scale(toNumber(group.q1), min, max, 304, 52);
          const medY = scale(toNumber(group.median), min, max, 304, 52);
          const q3Y = scale(toNumber(group.q3), min, max, 304, 52);
          const upperY = scale(toNumber(group.upper), min, max, 304, 52);
          const density = (group.density ?? []) as Array<Record<string, unknown>>;
          const maxDensity = Math.max(1, ...density.map((item) => toNumber(item.count)));
          return (
            <g key={`${group.label}-${index}`}>
              {violin ? density.map((item, densityIndex) => {
                const y = scale((toNumber(item.start) + toNumber(item.end)) / 2, min, max, 304, 52);
                const width = (toNumber(item.count) / maxDensity) * Math.min(34, slot / 2.4);
                return <line key={densityIndex} x1={cx - width} x2={cx + width} y1={y} y2={y} stroke={palette[index % palette.length]} strokeWidth="6" opacity="0.24" />;
              }) : null}
              <line x1={cx} x2={cx} y1={lowerY} y2={upperY} stroke={palette[index % palette.length]} strokeWidth="2" />
              <rect x={cx - Math.min(28, slot / 3)} y={q3Y} width={Math.min(56, slot / 1.5)} height={Math.max(3, q1Y - q3Y)} rx="5" fill={palette[index % palette.length]} opacity="0.28" />
              <line x1={cx - Math.min(28, slot / 3)} x2={cx + Math.min(28, slot / 3)} y1={medY} y2={medY} stroke={palette[index % palette.length]} strokeWidth="3" />
              <text x={cx - Math.min(28, slot / 3)} y="334">{String(group.label).slice(0, 9)}</text>
            </g>
          );
        })}
      </svg>
    </ChartShell>
  );
}

function HeatmapSvg({ data, corr = false }: { data: any; corr?: boolean }) {
  const cells = (data?.cells ?? []) as Array<Record<string, unknown>>;
  if (!cells.length) return <EmptyChart text="没有可绘制的热力图数据。" />;
  if (corr) {
    const columns = (data?.columns ?? []) as string[];
    const size = Math.max(14, Math.min(44, 560 / Math.max(1, columns.length)));
    return (
      <ChartShell label="相关性热力图">
        <svg viewBox="0 0 720 360" className="chart-svg rich heatmap-svg">
          {cells.map((cell, index) => {
            const x = columns.indexOf(String(cell.x));
            const y = columns.indexOf(String(cell.y));
            const value = Math.max(-1, Math.min(1, toNumber(cell.value)));
            const color = value >= 0 ? `rgba(164, 81, 42, ${0.15 + Math.abs(value) * 0.72})` : `rgba(52, 120, 90, ${0.15 + Math.abs(value) * 0.72})`;
            return <rect key={index} x={86 + x * size} y={42 + y * size} width={size - 2} height={size - 2} rx="3" fill={color} />;
          })}
          {columns.slice(0, 18).map((column, index) => <text key={column} x={86 + index * size} y="32">{column.slice(0, 7)}</text>)}
        </svg>
      </ChartShell>
    );
  }
  const max = Math.max(1, ...cells.map((cell) => toNumber(cell.value)));
  const xCount = Math.max(...cells.map((cell) => toNumber(cell.x))) + 1;
  const yCount = Math.max(...cells.map((cell) => toNumber(cell.y))) + 1;
  const width = 600 / Math.max(1, xCount);
  const height = 250 / Math.max(1, yCount);
  return (
    <ChartShell label="二维密度热力图">
      <svg viewBox="0 0 720 360" className="chart-svg rich heatmap-svg">
        {cells.map((cell, index) => (
          <rect
            key={index}
            x={60 + toNumber(cell.x) * width}
            y={46 + (yCount - 1 - toNumber(cell.y)) * height}
            width={Math.max(2, width - 1)}
            height={Math.max(2, height - 1)}
            rx="2"
            fill={`rgba(164, 81, 42, ${0.12 + (toNumber(cell.value) / max) * 0.78})`}
          />
        ))}
      </svg>
    </ChartShell>
  );
}

function PieSvg({ data }: { data: any }) {
  const slices = (data?.slices ?? []) as Array<Record<string, unknown>>;
  const total = slices.reduce((sum, item) => sum + Math.max(0, toNumber(item.value)), 0);
  if (!slices.length || total <= 0) return <EmptyChart text="没有可绘制的饼图数据。" />;
  let cursor = -Math.PI / 2;
  const radius = 112;
  const cx = 190;
  const cy = 170;
  return (
    <ChartShell label="饼图">
      <svg viewBox="0 0 720 360" className="chart-svg rich">
        {slices.map((slice, index) => {
          const angle = (Math.max(0, toNumber(slice.value)) / total) * Math.PI * 2;
          const start = cursor;
          const end = cursor + angle;
          cursor = end;
          const x1 = cx + Math.cos(start) * radius;
          const y1 = cy + Math.sin(start) * radius;
          const x2 = cx + Math.cos(end) * radius;
          const y2 = cy + Math.sin(end) * radius;
          const large = angle > Math.PI ? 1 : 0;
          const path = `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2} Z`;
          return <path key={`${slice.label}-${index}`} d={path} fill={palette[index % palette.length]} opacity="0.84" />;
        })}
        <circle cx={cx} cy={cy} r="54" fill="#fffdf8" />
        <text x="165" y="176">{total.toLocaleString()}</text>
        {slices.slice(0, 10).map((slice, index) => (
          <g key={`${slice.label}-legend`}>
            <rect x="360" y={62 + index * 24} width="10" height="10" rx="2" fill={palette[index % palette.length]} />
            <text x="378" y={72 + index * 24}>{String(slice.label).slice(0, 22)} · {valueText(slice.value)}</text>
          </g>
        ))}
      </svg>
    </ChartShell>
  );
}

function MatrixSvg({ data }: { data: any }) {
  const columns = (data?.columns ?? []) as string[];
  const pairs = (data?.pairs ?? []) as Array<Record<string, any>>;
  if (!columns.length || !pairs.length) return <EmptyChart text="没有可绘制的成对关系数据。" />;
  const cell = Math.min(112, 600 / columns.length);
  return (
    <ChartShell label="成对关系图">
      <svg viewBox="0 0 720 360" className="chart-svg rich matrix-svg">
        {pairs.map((pair, pairIndex) => {
          const xIndex = columns.indexOf(String(pair.x_col));
          const yIndex = columns.indexOf(String(pair.y_col));
          const points = (pair.points ?? []) as Array<Record<string, unknown>>;
          const xs = points.map((point) => toNumber(point.x));
          const ys = points.map((point) => toNumber(point.y));
          const minX = Math.min(...xs);
          const maxX = Math.max(...xs);
          const minY = Math.min(...ys);
          const maxY = Math.max(...ys);
          const left = 62 + xIndex * cell;
          const top = 36 + yIndex * cell;
          return (
            <g key={pairIndex}>
              <rect x={left} y={top} width={cell - 6} height={cell - 6} rx="4" fill="#fffdf8" stroke="rgba(222, 214, 200, 0.8)" />
              {points.slice(0, 120).map((point, pointIndex) => (
                <circle
                  key={pointIndex}
                  cx={scale(toNumber(point.x), minX, maxX, left + 8, left + cell - 14)}
                  cy={scale(toNumber(point.y), minY, maxY, top + cell - 14, top + 8)}
                  r="1.6"
                  fill={colorFor(point.color ?? "default")}
                  opacity="0.58"
                />
              ))}
            </g>
          );
        })}
        {columns.map((column, index) => <text key={column} x={62 + index * cell} y="28">{column.slice(0, 8)}</text>)}
      </svg>
    </ChartShell>
  );
}

function ChartRenderer({ result }: { result: ChartResponse | null }) {
  if (!result) return <EmptyChart text="选择图表类型和字段后生成图表。" />;
  const data = result.data ?? {};
  if (result.chart_type === "scatter") return <ScatterSvg data={data} />;
  if (result.chart_type === "scatter3d") return <ScatterSvg data={data} threeD />;
  if (result.chart_type === "line") return <LineAreaSvg data={data} />;
  if (result.chart_type === "area") return <LineAreaSvg data={data} area />;
  if (result.chart_type === "bar") return <BarSvg bars={data.bars ?? []} />;
  if (result.chart_type === "histogram") return <BarSvg bars={data.bins ?? []} label="直方图" />;
  if (result.chart_type === "box") return <BoxSvg data={data} />;
  if (result.chart_type === "violin") return <BoxSvg data={data} violin />;
  if (result.chart_type === "density_heatmap") return <HeatmapSvg data={data} />;
  if (result.chart_type === "corr_heatmap") return <HeatmapSvg data={data} corr />;
  if (result.chart_type === "pie") return <PieSvg data={data} />;
  if (result.chart_type === "scatter_matrix") return <MatrixSvg data={data} />;
  return <EmptyChart text="当前图表暂未实现渲染。" />;
}

export default function DataVisualization({ profile }: Props) {
  const allCols = profile?.columns ?? [];
  const numericCols = profile?.numeric_cols ?? [];
  const categoricalCols = profile?.categorical_cols ?? [];
  const categoryOrAll = categoricalCols.length ? categoricalCols : allCols;

  const [chartType, setChartType] = useState<ChartType>("scatter");
  const [title, setTitle] = useState("");
  const [xCol, setXCol] = useState(first(numericCols, first(allCols)));
  const [yCol, setYCol] = useState(second(numericCols, first(allCols)));
  const [zCol, setZCol] = useState(third(numericCols, first(allCols)));
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
  const [result, setResult] = useState<ChartResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!profile) return;
    setXCol(first(profile.numeric_cols, first(profile.columns)));
    setYCol(second(profile.numeric_cols, first(profile.columns)));
    setZCol(third(profile.numeric_cols, first(profile.columns)));
    setYCols(profile.numeric_cols.slice(0, 2));
    setMatrixCols(profile.numeric_cols.slice(0, 4));
    setColorCol("无");
    setGroupCol("无");
    setSizeCol("无");
  }, [profile?.session_id]);

  const selectedChart = useMemo(() => chartOptions.find((item) => item.id === chartType) ?? chartOptions[0], [chartType]);
  const SelectedIcon = selectedChart.icon;

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
      return { chart_type: chartType, title, x_col: xCol, y_col: yCol, z_col: zCol, color_col: noneToUndefined(colorCol), size_col: noneToUndefined(sizeCol) };
    }
    return { chart_type: chartType, title, x_col: xCol, y_col: yCol, color_col: noneToUndefined(colorCol), size_col: noneToUndefined(sizeCol) };
  };

  const generateChart = async () => {
    if (!profile?.session_id) return;
    setLoading(true);
    setError("");
    try {
      const payload = await fetchVisualization(profile.session_id, buildPayload());
      setResult(payload as unknown as ChartResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "图表生成失败");
    } finally {
      setLoading(false);
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
          <label className="form-field">
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
          ) : null}
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
        {chartType === "bar" ? (
          <>
            <label className="form-field">
              <span>聚合方式</span>
              <select value={agg === "none" ? "mean" : agg} onChange={(event) => setAgg(event.target.value)}>
                {aggOptions.filter((item) => item.value !== "none").map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
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
          <label className="form-field">
            <span>气泡大小</span>
            <select value={sizeCol} onChange={(event) => setSizeCol(event.target.value)}>
              <option>无</option>
              {numericCols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
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

          <ChartRenderer result={result} />
        </article>
      </div>
    </section>
  );
}
