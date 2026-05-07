import { useState, useEffect, useRef, useCallback } from 'react'
import { RadialBarChart, RadialBar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import './index.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

const AGENTS = [
  { id: 'orchestrator', name: 'Orchestrator', role: 'Task Decomposition & Routing', icon: '🧠' },
  { id: 'planner',      name: 'Planner',      role: 'ReAct Loop & Strategy',        icon: '📋' },
  { id: 'executor',     name: 'Executor',      role: 'Tool Invocation & Actions',    icon: '⚡' },
  { id: 'critic',       name: 'Critic',        role: 'Quality Evaluation & Safety',  icon: '🔍' },
]

const ADV_TYPES = [
  { id: 'prompt_injection', label: '💉 Injection',   desc: 'Override system prompt' },
  { id: 'jailbreak',        label: '🔓 Jailbreak',   desc: 'Safety bypass attempts' },
  { id: 'hallucination',    label: '🌀 Hallucination', desc: 'Fabrication probes' },
  { id: 'token_smuggling',  label: '🕵️ Smuggling',  desc: 'Unicode/whitespace attacks' },
]

const EXAMPLE_TASKS = [
  "Explain the transformer attention mechanism and calculate the number of parameters in a 7B model",
  "Search Wikipedia for information about quantum entanglement and summarize key points",
  "Write and execute Python code that generates the first 20 Fibonacci numbers",
  "Analyze the trade-offs between ReAct and chain-of-thought prompting strategies",
]

function formatTime(date) {
  return new Date(date).toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function ScoreRing({ score, size = 60 }) {
  const r = size / 2 - 6
  const circ = 2 * Math.PI * r
  const fill = circ * (1 - (score || 0))
  const color = score >= 0.8 ? '#34d399' : score >= 0.6 ? '#fbbf24' : '#f87171'
  return (
    <div className="score-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={5} />
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={5}
          strokeDasharray={circ} strokeDashoffset={fill} strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
      </svg>
      <div className="score-center" style={{ color }}>{score ? Math.round(score * 100) : '—'}</div>
    </div>
  )
}

function ScoreDimensions({ evalResult }) {
  if (!evalResult?.scores) return null
  const dims = [
    { key: 'accuracy',     label: 'Accuracy' },
    { key: 'completeness', label: 'Complete' },
    { key: 'coherence',    label: 'Coherence' },
    { key: 'safety',       label: 'Safety' },
    { key: 'efficiency',   label: 'Efficiency' },
  ]
  return (
    <div className="score-bars">
      {dims.map(d => {
        const val = evalResult.scores[d.key] || 0
        return (
          <div className="score-bar-row" key={d.key}>
            <span className="score-bar-label">{d.label}</span>
            <div className="score-bar-track">
              <div className="score-bar-fill" style={{ width: `${val * 100}%` }} />
            </div>
            <span className="score-bar-value">{Math.round(val * 100)}</span>
          </div>
        )
      })}
    </div>
  )
}

export default function App() {
  const [task, setTask]             = useState('')
  const [loading, setLoading]       = useState(false)
  const [result, setResult]         = useState(null)
  const [activeTab, setActiveTab]   = useState('output')
  const [enableEval, setEnableEval] = useState(true)
  const [events, setEvents]         = useState([])
  const [evalMetrics, setEvalMetrics] = useState(null)
  const [agentStatus, setAgentStatus]  = useState({})
  const [wsStatus, setWsStatus]     = useState('disconnected')
  const [advTypes, setAdvTypes]     = useState([])
  const [advRunning, setAdvRunning] = useState(false)
  const [advReport, setAdvReport]   = useState(null)
  const [scoreTrend, setScoreTrend] = useState([])
  const [totalTasks, setTotalTasks] = useState(0)

  const wsRef = useRef(null)
  const eventLogRef = useRef(null)

  const addEvent = useCallback((msg, type = 'info') => {
    setEvents(prev => [{
      id: Date.now() + Math.random(),
      time: new Date(),
      msg,
      type,
    }, ...prev].slice(0, 60))
  }, [])

  // WebSocket
  useEffect(() => {
    let ws, pingInterval, reconnectTimeout

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL)
        wsRef.current = ws

        ws.onopen = () => {
          setWsStatus('connected')
          addEvent('Connected to orchestration system', 'success')
          pingInterval = setInterval(() => ws.readyState === 1 && ws.send(JSON.stringify({ type: 'ping' })), 25000)
        }

        ws.onmessage = (e) => {
          try {
            const data = JSON.parse(e.data)
            if (data.type === 'pong' || data.type === 'heartbeat') return
            if (data.agents) setAgentStatus(data.agents)
            if (data.eval_metrics) {
              setEvalMetrics(data.eval_metrics)
              if (data.eval_metrics.score_trend) setScoreTrend(data.eval_metrics.score_trend.map((s, i) => ({ i, score: s })))
            }
            if (data.type === 'task_started')    addEvent(`Task started: ${data.task?.slice(0, 60)}…`, 'info')
            if (data.type === 'task_completed')  addEvent(`Task done — score: ${data.eval_score ? Math.round(data.eval_score * 100) : 'N/A'}`, 'success')
          } catch {}
        }

        ws.onerror = () => setWsStatus('error')
        ws.onclose = () => {
          setWsStatus('disconnected')
          clearInterval(pingInterval)
          reconnectTimeout = setTimeout(connect, 4000)
        }
      } catch {}
    }

    connect()
    return () => {
      clearInterval(pingInterval)
      clearTimeout(reconnectTimeout)
      ws?.close()
    }
  }, [addEvent])

  // Poll eval metrics
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/eval/metrics`)
        if (r.ok) {
          const d = await r.json()
          setEvalMetrics(d)
          if (d.score_trend) setScoreTrend(d.score_trend.map((s, i) => ({ i, score: s })))
        }
      } catch {}
    }
    fetchMetrics()
    const id = setInterval(fetchMetrics, 10000)
    return () => clearInterval(id)
  }, [])

  const runTask = async () => {
    if (!task.trim() || loading) return
    setLoading(true)
    setResult(null)
    setActiveTab('output')
    addEvent(`Submitting task: ${task.slice(0, 60)}…`)

    try {
      const res = await fetch(`${API_BASE}/api/v1/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, enable_eval: enableEval }),
      })
      const data = await res.json()
      setResult(data)
      setTotalTasks(t => t + 1)
      if (data.eval_score != null) {
        setScoreTrend(prev => [...prev, { i: prev.length, score: data.eval_score }].slice(-20))
      }
      addEvent(`Task complete — ${data.tokens_used} tokens, ${data.duration_seconds?.toFixed(2)}s`, 'success')
    } catch (err) {
      addEvent(`Task failed: ${err.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const runAdversarial = async () => {
    setAdvRunning(true)
    setAdvReport(null)
    addEvent('Starting adversarial test suite…', 'info')
    try {
      const res = await fetch(`${API_BASE}/api/v1/adversarial/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attack_types: advTypes.length ? advTypes : null }),
      })
      const data = await res.json()
      setAdvReport(data)
      const grade = data.summary?.grade || 'N/A'
      addEvent(`Adversarial suite done — Grade: ${grade}`, data.summary?.robustness_score >= 0.8 ? 'success' : 'error')
    } catch (err) {
      addEvent(`Adversarial test failed: ${err.message}`, 'error')
    } finally {
      setAdvRunning(false)
    }
  }

  const triggerImprove = async () => {
    addEvent('Triggering self-improvement cycle…', 'info')
    try {
      const res = await fetch(`${API_BASE}/api/v1/eval/improve`, { method: 'POST' })
      const data = await res.json()
      addEvent(data.is_degrading ? 'Quality degradation detected — patches applied' : 'System healthy — no patches needed',
        data.is_degrading ? 'error' : 'success')
    } catch {}
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">🤖</div>
          <div>
            <div className="header-title">Multi-Agent LLM Orchestration</div>
            <div className="header-subtitle">Production-Grade · Self-Improving · Adversarial-Robust</div>
          </div>
        </div>
        <div className="header-status">
          <div className={`status-dot ${wsStatus === 'connected' ? '' : 'disconnected'}`}>
            {wsStatus === 'connected' ? 'Live' : 'Offline'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--clr-text-muted)' }}>
            {new Date().toLocaleDateString()}
          </div>
        </div>
      </header>

      <div className="main-content">
        {/* ── Stat Tiles ── */}
        <div className="stats-grid">
          {[
            { label: 'Tasks Run',      value: totalTasks,                                                     cls: 'primary',  suffix: '' },
            { label: 'Avg Eval Score', value: evalMetrics?.avg_score ? Math.round(evalMetrics.avg_score*100) : '—', cls: 'success', suffix: evalMetrics?.avg_score ? '%' : '' },
            { label: 'Pass Rate',      value: evalMetrics?.pass_rate ? Math.round(evalMetrics.pass_rate*100) : '—', cls: 'warning', suffix: evalMetrics?.pass_rate ? '%' : '' },
            { label: 'Impr. Cycles',   value: evalMetrics?.improvement_cycles ?? 0,                          cls: 'danger',   suffix: '' },
          ].map(s => (
            <div key={s.label} className={`stat-tile ${s.cls}`}>
              <div className="stat-label">{s.label}</div>
              <div className="stat-value">{s.value}{s.suffix}</div>
              <div className="stat-change">{s.cls === 'success' && evalMetrics?.is_degrading ? '⚠ Degrading' : 'System active'}</div>
            </div>
          ))}
        </div>

        {/* ── Left Column ── */}
        <div className="task-panel">
          {/* Task Input */}
          <div className="card">
            <div className="card-header">
              <div className="card-title"><span className="card-title-icon">🎯</span> Submit Task</div>
            </div>
            <div className="task-input-wrapper">
              <textarea
                id="task-input"
                className="task-textarea"
                value={task}
                onChange={e => setTask(e.target.value)}
                onKeyDown={e => e.ctrlKey && e.key === 'Enter' && runTask()}
                placeholder="Enter a task for the agent system… (Ctrl+Enter to submit)"
                rows={5}
              />
            </div>

            {/* Example tasks */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, margin: '10px 0' }}>
              {EXAMPLE_TASKS.map((t, i) => (
                <button key={i} className="btn btn-secondary" style={{ fontSize: 11, padding: '4px 10px' }}
                  onClick={() => setTask(t)}>
                  Example {i + 1}
                </button>
              ))}
            </div>

            <div className="task-controls">
              <div className="toggle-group">
                <button id="toggle-eval" className={`toggle-chip ${enableEval ? 'active' : ''}`} onClick={() => setEnableEval(!enableEval)}>
                  🔍 Evaluate
                </button>
              </div>
              <button id="run-task-btn" className="btn btn-primary" onClick={runTask} disabled={loading || !task.trim()}>
                {loading ? <><div className="spinner" /> Running…</> : '▶ Run Task'}
              </button>
            </div>
          </div>

          {/* Results */}
          <div className="card" style={{ flex: 1 }}>
            <div className="result-tabs">
              {['output', 'reasoning', 'tools', 'eval'].map(tab => (
                <div key={tab} className={`result-tab ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </div>
              ))}
            </div>

            <div className="result-content">
              {loading && (
                <div className="result-empty">
                  <div className="thinking-dots"><span /><span /><span /></div>
                  <div style={{ marginTop: 12, color: 'var(--clr-text-muted)', fontSize: 13 }}>Agents thinking…</div>
                </div>
              )}

              {!loading && !result && (
                <div className="result-empty">
                  <div className="result-empty-icon">🤖</div>
                  <div>Submit a task to see results here</div>
                </div>
              )}

              {!loading && result && (
                <>
                  {activeTab === 'output' && (
                    <pre className="result-text">{result.output || 'No output generated'}</pre>
                  )}
                  {activeTab === 'reasoning' && (
                    <pre className="result-text">{result.reasoning || 'No reasoning captured'}</pre>
                  )}
                  {activeTab === 'tools' && (
                    result.tool_calls?.length > 0
                      ? <div className="tool-timeline">
                          {result.tool_calls.map((tc, i) => (
                            <div key={i} className={`tool-call-item ${tc.error ? 'error' : ''}`}>
                              <span className="tool-call-icon">{tc.error ? '❌' : '🔧'}</span>
                              <div className="tool-call-info">
                                <div className="tool-call-name">{tc.tool}</div>
                                <div className="tool-call-result">{tc.result_summary || tc.error || 'No result'}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      : <div className="result-empty"><div>No tool calls in this task</div></div>
                  )}
                  {activeTab === 'eval' && (
                    result.eval_result
                      ? <>
                          <div className="score-display">
                            <ScoreRing score={result.eval_result.overall_score} />
                            <div className="score-details">
                              <div className="score-label">Overall Quality Score</div>
                              <ScoreDimensions evalResult={result.eval_result} />
                            </div>
                          </div>
                          {result.eval_result.issues_found?.length > 0 && (
                            <div style={{ marginTop: 8 }}>
                              <div style={{ fontSize: 11, color: 'var(--clr-warning)', marginBottom: 4 }}>⚠ Issues Found</div>
                              {result.eval_result.issues_found.map((issue, i) => (
                                <div key={i} style={{ fontSize: 12, color: 'var(--clr-text-dim)', padding: '3px 0' }}>• {issue}</div>
                              ))}
                            </div>
                          )}
                        </>
                      : <div className="result-empty"><div>Enable evaluation to see scores</div></div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Score Trend Chart */}
          {scoreTrend.length > 2 && (
            <div className="card">
              <div className="card-header">
                <div className="card-title"><span className="card-title-icon">📈</span> Eval Score Trend</div>
              </div>
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={scoreTrend} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
                  <CartesianGrid stroke="rgba(99,132,255,0.08)" />
                  <XAxis dataKey="i" hide />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
                  <Tooltip formatter={v => `${Math.round(v * 100)}%`} contentStyle={{ background: '#0d1220', border: '1px solid rgba(99,132,255,0.2)', borderRadius: 8, fontSize: 12 }} />
                  <Line type="monotone" dataKey="score" stroke="#6384ff" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Adversarial Testing */}
          <div className="card">
            <div className="card-header">
              <div className="card-title"><span className="card-title-icon">⚔️</span> Adversarial Robustness Testing</div>
            </div>
            <div className="adv-grid" style={{ marginBottom: 12 }}>
              {ADV_TYPES.map(t => (
                <div key={t.id} className={`adv-type-chip ${advTypes.includes(t.id) ? 'selected' : ''}`}
                  onClick={() => setAdvTypes(prev => prev.includes(t.id) ? prev.filter(x => x !== t.id) : [...prev, t.id])}>
                  {t.label}
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <button id="run-adversarial-btn" className="btn btn-danger" onClick={runAdversarial} disabled={advRunning}>
                {advRunning ? <><div className="spinner" /> Running…</> : '🎯 Run Tests'}
              </button>
              {advReport?.summary && (
                <div style={{ fontSize: 12, color: 'var(--clr-text-dim)' }}>
                  Score: <span style={{ color: advReport.summary.robustness_score >= 0.8 ? 'var(--clr-success)' : 'var(--clr-danger)', fontWeight: 700 }}>
                    {Math.round(advReport.summary.robustness_score * 100)}%
                  </span> — {advReport.summary.grade}
                </div>
              )}
            </div>
            {advReport?.summary && (
              <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Total Tests',  value: advReport.summary.total_tests, color: 'var(--clr-text)' },
                  { label: 'Detected',     value: advReport.summary.detected,    color: 'var(--clr-success)' },
                  { label: 'Bypassed',     value: advReport.summary.bypassed,    color: 'var(--clr-danger)' },
                ].map(s => (
                  <div key={s.label} style={{ textAlign: 'center', padding: '8px', background: 'var(--clr-surface-2)', borderRadius: 8 }}>
                    <div style={{ fontSize: 20, fontWeight: 800, color: s.color, fontFamily: 'JetBrains Mono, monospace' }}>{s.value}</div>
                    <div style={{ fontSize: 10, color: 'var(--clr-text-muted)', marginTop: 2 }}>{s.label}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── Right Sidebar ── */}
        <div className="sidebar">
          {/* Agent Status */}
          <div className="card">
            <div className="card-header">
              <div className="card-title"><span className="card-title-icon">👥</span> Agent Status</div>
            </div>
            <div className="agent-list">
              {AGENTS.map(a => (
                <div key={a.id} className="agent-item">
                  <div className={`agent-avatar ${a.id}`}>{a.icon}</div>
                  <div className="agent-info">
                    <div className="agent-name">{a.name}</div>
                    <div className="agent-role">{a.role}</div>
                  </div>
                  <div className={`agent-status-badge badge-${agentStatus?.status || 'idle'}`}>
                    {loading && a.id === 'orchestrator' ? 'running' : 'idle'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Eval Loop */}
          <div className="card">
            <div className="card-header">
              <div className="card-title"><span className="card-title-icon">🔄</span> Self-Improvement Loop</div>
            </div>
            {evalMetrics ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  { label: 'Total Evals',   value: evalMetrics.total_evaluations ?? 0 },
                  { label: 'Window Avg',    value: evalMetrics.recent_window_avg != null ? `${Math.round(evalMetrics.recent_window_avg * 100)}%` : '—' },
                  { label: 'Degrading',     value: evalMetrics.is_degrading ? '⚠ YES' : '✅ No', warn: evalMetrics.is_degrading },
                  { label: 'Best Examples', value: evalMetrics.best_examples_count ?? 0 },
                ].map(m => (
                  <div key={m.label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '6px 0', borderBottom: '1px solid var(--clr-border)' }}>
                    <span style={{ color: 'var(--clr-text-muted)' }}>{m.label}</span>
                    <span style={{ fontWeight: 600, color: m.warn ? 'var(--clr-warning)' : 'var(--clr-text)' }}>{m.value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: 'var(--clr-text-muted)', fontSize: 12, textAlign: 'center', padding: 16 }}>Run a task to see eval data</div>
            )}
            <button id="improve-btn" className="btn btn-secondary" style={{ width: '100%', marginTop: 12, justifyContent: 'center' }} onClick={triggerImprove}>
              🚀 Trigger Improvement Cycle
            </button>
          </div>

          {/* Event Log */}
          <div className="card" style={{ flex: 1 }}>
            <div className="card-header">
              <div className="card-title"><span className="card-title-icon">📡</span> Live Event Log</div>
              <button className="btn btn-secondary" style={{ fontSize: 11, padding: '3px 8px' }} onClick={() => setEvents([])}>
                Clear
              </button>
            </div>
            <div className="event-log" ref={eventLogRef}>
              {events.length === 0 && (
                <div style={{ color: 'var(--clr-text-muted)', fontSize: 12, textAlign: 'center', padding: 12 }}>
                  Waiting for events…
                </div>
              )}
              {events.map(ev => (
                <div key={ev.id} className="event-item">
                  <span className="event-time">{formatTime(ev.time)}</span>
                  <span className={`event-msg ${ev.type}`}>{ev.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
