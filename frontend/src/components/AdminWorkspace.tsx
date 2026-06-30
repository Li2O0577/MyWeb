import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, RefreshCcw, ShieldCheck, UserCheck, Users } from "lucide-react";

import { fetchAdminUsers, fetchAuditLogs, updateAdminUser } from "../services/api";
import type { AuditLogEntry, AuthUser, ManagedUser } from "../types";


const actionLabels: Record<string, string> = {
  "auth.register": "注册账号",
  "auth.login": "登录",
  "auth.login_failed": "登录失败",
  "auth.logout": "退出登录",
  "admin.user_access_updated": "更新账号权限"
};

function formatTime(value: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(value * 1000));
}

function detailText(log: AuditLogEntry) {
  if (log.action === "admin.user_access_updated") {
    const role = log.details.role === "admin" ? "管理员" : "普通用户";
    const status = log.details.disabled ? "已禁用" : "可用";
    return `${role} · ${status}`;
  }
  if (log.action === "auth.login_failed" && typeof log.details.username === "string") {
    return `用户名：${log.details.username}`;
  }
  return log.source_ip ? `来源：${log.source_ip}` : "无附加信息";
}

export default function AdminWorkspace({ currentUser }: { currentUser: AuthUser }) {
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setError("");
    const [nextUsers, nextLogs] = await Promise.all([
      fetchAdminUsers(signal),
      fetchAuditLogs(100, signal)
    ]);
    setUsers(nextUsers);
    setLogs(nextLogs);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    refresh(controller.signal)
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "读取管理数据失败");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [refresh]);

  const metrics = useMemo(() => ({
    total: users.length,
    admins: users.filter((user) => user.role === "admin" && !user.disabled).length,
    disabled: users.filter((user) => user.disabled).length,
    sessions: users.reduce((sum, user) => sum + user.active_sessions, 0)
  }), [users]);

  const changeUser = async (user: ManagedUser, changes: { role?: "admin" | "user"; disabled?: boolean }) => {
    const action = changes.role
      ? `将“${user.username}”设置为${changes.role === "admin" ? "管理员" : "普通用户"}`
      : `${changes.disabled ? "禁用" : "启用"}账号“${user.username}”`;
    if (!window.confirm(`确定${action}吗？`)) return;
    setBusyUserId(user.user_id);
    setError("");
    try {
      await updateAdminUser(user.user_id, changes);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新账号失败");
    } finally {
      setBusyUserId(null);
    }
  };

  return (
    <section className="admin-workspace" aria-label="账号管理">
      <div className="admin-metrics">
        <div><Users size={17} /><span>账号总数</span><strong>{metrics.total}</strong></div>
        <div><ShieldCheck size={17} /><span>可用管理员</span><strong>{metrics.admins}</strong></div>
        <div><UserCheck size={17} /><span>活跃登录</span><strong>{metrics.sessions}</strong></div>
        <div><Activity size={17} /><span>禁用账号</span><strong>{metrics.disabled}</strong></div>
      </div>

      {error ? <div className="inline-error">{error}</div> : null}

      <article className="admin-panel">
        <div className="panel-title split">
          <span><Users size={17} /><h2>账号与权限</h2></span>
          <button className="button ghost" type="button" onClick={() => refresh().catch((err) => setError(String(err)))} disabled={loading}>
            <RefreshCcw size={15} />刷新
          </button>
        </div>
        <div className="table-wrap admin-table-wrap">
          <table className="data-table admin-table">
            <thead><tr><th>账号</th><th>角色</th><th>状态</th><th>活跃登录</th><th>创建时间</th></tr></thead>
            <tbody>
              {users.map((user) => {
                const isSelf = user.user_id === currentUser.user_id;
                const busy = busyUserId === user.user_id;
                return (
                  <tr key={user.user_id}>
                    <td><strong>{user.username}</strong>{isSelf ? <small>当前账号</small> : null}</td>
                    <td>
                      <select
                        className="admin-select"
                        value={user.role}
                        disabled={busy || isSelf}
                        aria-label={`设置 ${user.username} 的角色`}
                        onChange={(event) => changeUser(user, { role: event.target.value as "admin" | "user" })}
                      >
                        <option value="user">普通用户</option>
                        <option value="admin">管理员</option>
                      </select>
                    </td>
                    <td>
                      <label className="account-toggle">
                        <input
                          type="checkbox"
                          checked={!user.disabled}
                          disabled={busy || isSelf}
                          onChange={(event) => changeUser(user, { disabled: !event.target.checked })}
                        />
                        <span>{user.disabled ? "已禁用" : "可用"}</span>
                      </label>
                    </td>
                    <td>{user.active_sessions}</td>
                    <td>{user.created_at ? formatTime(user.created_at) : "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </article>

      <article className="admin-panel">
        <div className="panel-title"><Activity size={17} /><h2>最近审计事件</h2></div>
        <div className="audit-list">
          {logs.length ? logs.map((log) => (
            <div className="audit-row" key={log.audit_id}>
              <span className={`audit-mark ${log.action === "auth.login_failed" ? "danger" : ""}`} />
              <div>
                <strong>{actionLabels[log.action] ?? log.action}</strong>
                <span>{log.actor_username ?? "未认证来源"}{log.target_username ? ` → ${log.target_username}` : ""}</span>
              </div>
              <small>{detailText(log)}</small>
              <time>{formatTime(log.created_at)}</time>
            </div>
          )) : <div className="empty-list">暂无审计事件。</div>}
        </div>
      </article>
    </section>
  );
}
