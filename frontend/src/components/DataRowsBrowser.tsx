import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Rows3 } from "lucide-react";

import { fetchDataRows } from "../services/api";
import type { DataRowsResponse } from "../types";


function valueToText(value: unknown) {
  if (value === null || value === undefined) return "空";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "非有限值";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export default function DataRowsBrowser({ sessionId }: { sessionId: string }) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [data, setData] = useState<DataRowsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setPage(1);
    setData(null);
  }, [sessionId]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    setError("");
    fetchDataRows(sessionId, page, pageSize, controller.signal)
      .then((payload) => {
        if (active) setData(payload);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        if (active) {
          setData(null);
          setError(err instanceof Error ? err.message : "读取数据失败");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [page, pageSize, sessionId]);

  const firstRow = data && data.total_rows ? (data.page - 1) * data.page_size + 1 : 0;
  const lastRow = data ? Math.min(data.page * data.page_size, data.total_rows) : 0;

  return (
    <section className="all-data-panel" aria-label="查看全部数据">
      <div className="all-data-toolbar">
        <div>
          <Rows3 size={17} aria-hidden="true" />
          <span><strong>全部数据</strong><small>{data ? `${firstRow}-${lastRow} / ${data.total_rows.toLocaleString()} 行` : "正在读取"}</small></span>
        </div>
        <div className="pagination-controls">
          <label>
            <span>每页</span>
            <select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>500</option>
            </select>
          </label>
          <button className="icon-button" type="button" aria-label="上一页" title="上一页" disabled={loading || page <= 1} onClick={() => setPage((value) => value - 1)}>
            <ChevronLeft size={16} />
          </button>
          <span className="page-indicator">{data ? `${data.page} / ${data.total_pages}` : "- / -"}</span>
          <button className="icon-button" type="button" aria-label="下一页" title="下一页" disabled={loading || !data || page >= data.total_pages} onClick={() => setPage((value) => value + 1)}>
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {error ? <div className="inline-error">{error}</div> : null}
      <div className="table-wrap all-data-table-wrap" aria-busy={loading}>
        <table className="data-table all-data-table">
          <thead><tr>{(data?.columns ?? []).map((column) => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>
            {data?.rows.map((row, rowIndex) => (
              <tr key={`${data.page}-${rowIndex}`}>
                {data.columns.map((column) => <td key={column}>{valueToText(row[column])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
        {loading ? <div className="table-loading">正在读取数据...</div> : null}
      </div>
    </section>
  );
}
