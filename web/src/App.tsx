import {
  AlertCircle,
  ArrowRight,
  CalendarClock,
  Check,
  ChevronDown,
  CircleDot,
  Database,
  FileText,
  History,
  Mail,
  MessageSquareText,
  Pause,
  Play,
  Search,
  ShieldCheck,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

type Evidence = {
  evidence_id: string
  source_id: string
  message_id: string
  source_type: string
  observed_at: string
  speaker: string
  text: string
}

type MemoryState = {
  status: string
  label: string
  value: string
  evidence: string[]
}

type CaseSummary = {
  case_id: string
  title: string
  summary: string
  capability: string
  difficulty: string
  as_of: string
}

type DemoCase = CaseSummary & {
  question: string
  source_events: Evidence[]
  memory_states: MemoryState[]
  answer: {
    status: string
    body: string
    confidence: number
    evidence: string[]
    abstention_reason: string | null
  }
  review: {
    result: string
    failure_type: string | null
    explanation: string
    expected_answer: string
  }
}

type Run = {
  series_id: string
  model: string
  status: string
  requests: number
  input_tokens: number
  output_tokens: number
  cost: string
  note: string
}

type Scorecard = {
  series_id: string
  baseline_id: string
  label: string
  status: string
  answer_accuracy: number | null
  evidence_recall: number | null
  abstention_precision: number | null
  recall_at_10: number | null
  latency_ms: number | null
  failures: number | null
}

type View = 'explorer' | 'scorecard'
type MobilePanel = 'timeline' | 'memory' | 'answer'

const sourceIcon = (type: string) => {
  if (type === 'email') return Mail
  if (type === 'calendar') return CalendarClock
  return MessageSquareText
}

const formatDate = (value: string) => new Intl.DateTimeFormat('en', {
  month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
}).format(new Date(value))

const percent = (value: number | null) => value === null ? 'Not scored' : `${Math.round(value * 100)}%`

function StatusTag({ value }: { value: string }) {
  return <span className={`status-tag status-${value}`}>{value.replaceAll('_', ' ')}</span>
}

function App() {
  const [view, setView] = useState<View>('explorer')
  const [summaries, setSummaries] = useState<CaseSummary[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [demoCase, setDemoCase] = useState<DemoCase | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [scorecards, setScorecards] = useState<Scorecard[]>([])
  const [replayPosition, setReplayPosition] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>('timeline')
  const [scoreFilter, setScoreFilter] = useState('all')
  const [baselineFilter, setBaselineFilter] = useState('all')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/demo/cases').then((response) => response.ok ? response.json() : Promise.reject(response)),
      fetch('/api/runs').then((response) => response.ok ? response.json() : Promise.reject(response)),
      fetch('/api/scorecards').then((response) => response.ok ? response.json() : Promise.reject(response)),
    ]).then(([caseRows, runRows, scoreRows]) => {
      setSummaries(caseRows)
      setRuns(runRows)
      setScorecards(scoreRows)
      setSelectedId(caseRows[0]?.case_id ?? '')
    }).catch(() => setError('The demo artifacts could not be loaded.'))
  }, [])

  useEffect(() => {
    if (!selectedId) return
    setError(null)
    fetch(`/api/demo/cases/${selectedId}`)
      .then((response) => response.ok ? response.json() : Promise.reject(response))
      .then((value: DemoCase) => {
        setDemoCase(value)
        setReplayPosition(value.source_events.length)
        setSelectedEvidence(value.answer.evidence[0] ?? null)
        setIsPlaying(false)
      })
      .catch(() => setError('This case could not be loaded.'))
  }, [selectedId])

  useEffect(() => {
    if (!isPlaying || !demoCase) return
    if (replayPosition >= demoCase.source_events.length) {
      setIsPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setReplayPosition((value) => value + 1), 850)
    return () => window.clearTimeout(timer)
  }, [isPlaying, replayPosition, demoCase])

  const visibleEvidence = useMemo(
    () => demoCase?.source_events.slice(0, replayPosition) ?? [],
    [demoCase, replayPosition],
  )
  const visibleIds = useMemo(() => new Set(visibleEvidence.map((item) => item.evidence_id)), [visibleEvidence])
  const filteredScores = scorecards.filter((item) =>
    (scoreFilter === 'all' || item.status === scoreFilter)
    && (baselineFilter === 'all' || item.baseline_id === baselineFilter),
  )
  const baselineOptions = Array.from(
    new Map(scorecards.map((score) => [score.baseline_id, score.label])).entries(),
  )

  const restartReplay = () => {
    setReplayPosition(0)
    setIsPlaying(true)
    setMobilePanel('timeline')
  }

  if (error && summaries.length === 0) {
    return <main className="fatal-state"><AlertCircle size={24} /><h1>Demo unavailable</h1><p>{error}</p></main>
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><Database size={19} /><span>Longitudinal Memory</span><small>benchmark</small></div>
        <nav className="view-tabs" aria-label="Primary views">
          <button className={view === 'explorer' ? 'active' : ''} onClick={() => setView('explorer')}><Search size={15} />Explorer</button>
          <button className={view === 'scorecard' ? 'active' : ''} onClick={() => setView('scorecard')}><FileText size={15} />Scorecard</button>
        </nav>
        <div className="artifact-mode"><ShieldCheck size={15} />Artifact replay</div>
      </header>

      {view === 'explorer' ? (
        <main className="explorer">
          <section className="casebar">
            <div className="case-picker">
              <label htmlFor="case-select">Guided case</label>
              <div className="select-wrap">
                <select id="case-select" value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
                  {summaries.map((item) => <option key={item.case_id} value={item.case_id}>{item.title}</option>)}
                </select>
                <ChevronDown size={15} />
              </div>
            </div>
            {demoCase && <>
              <div className="case-context"><StatusTag value={demoCase.capability} /><span>{demoCase.summary}</span></div>
              <button className="replay-button" onClick={restartReplay}><Play size={15} />Replay case</button>
            </>}
          </section>

          <div className="mobile-tabs" role="tablist" aria-label="Explorer panels">
            {(['timeline', 'memory', 'answer'] as MobilePanel[]).map((panel) => (
              <button key={panel} role="tab" aria-selected={mobilePanel === panel} className={mobilePanel === panel ? 'active' : ''} onClick={() => setMobilePanel(panel)}>{panel}</button>
            ))}
          </div>

          {demoCase ? <div className="panel-grid">
            <section className={`panel timeline-panel ${mobilePanel === 'timeline' ? 'mobile-active' : ''}`} aria-label="Source timeline">
              <div className="panel-heading"><div><span className="eyebrow">Source history</span><h2>Chronological replay</h2></div><span className="count">{visibleEvidence.length}/{demoCase.source_events.length}</span></div>
              <div className="replay-control">
                <button className="icon-button" title={isPlaying ? 'Pause replay' : 'Play replay'} aria-label={isPlaying ? 'Pause replay' : 'Play replay'} onClick={() => {
                  if (replayPosition >= demoCase.source_events.length) setReplayPosition(0)
                  setIsPlaying((value) => !value)
                }}>{isPlaying ? <Pause size={15} /> : <Play size={15} />}</button>
                <input aria-label="Replay position" type="range" min="0" max={demoCase.source_events.length} value={replayPosition} onChange={(event) => {
                  setIsPlaying(false)
                  setReplayPosition(Number(event.target.value))
                }} />
              </div>
              <div className="timeline" aria-live="polite">
                {visibleEvidence.length === 0 && <div className="empty-state"><History size={22} /><span>Replay has not started.</span></div>}
                {visibleEvidence.map((event, index) => {
                  const Icon = sourceIcon(event.source_type)
                  return <button key={event.evidence_id} className={`timeline-event ${selectedEvidence === event.evidence_id ? 'selected' : ''}`} onClick={() => {
                    setSelectedEvidence(event.evidence_id)
                    setMobilePanel('answer')
                  }}>
                    <span className="timeline-marker"><Icon size={14} /></span>
                    <span className="event-content"><span className="event-meta"><b>{event.speaker}</b><time>{formatDate(event.observed_at)}</time></span><span>{event.text}</span><small>{event.source_id}</small></span>
                    {index < visibleEvidence.length - 1 && <span className="timeline-line" />}
                  </button>
                })}
              </div>
            </section>

            <section className={`panel memory-panel ${mobilePanel === 'memory' ? 'mobile-active' : ''}`} aria-label="Memory states">
              <div className="panel-heading"><div><span className="eyebrow">Memory model</span><h2>Belief state over time</h2></div></div>
              <div className="memory-list">
                {demoCase.memory_states.map((memory, index) => {
                  const available = memory.evidence.length === 0 || memory.evidence.some((id) => visibleIds.has(id))
                  return <button key={`${memory.label}-${index}`} disabled={!available} className={`memory-row memory-${memory.status}`} onClick={() => {
                    const id = memory.evidence.find((item) => visibleIds.has(item))
                    if (id) setSelectedEvidence(id)
                    setMobilePanel('answer')
                  }}>
                    <span className="memory-step">{index + 1}</span>
                    <span className="memory-copy"><span className="memory-topline"><b>{memory.label}</b><StatusTag value={memory.status} /></span><span>{available ? memory.value : 'Waiting for source evidence...'}</span><small>{memory.evidence.length} evidence link{memory.evidence.length === 1 ? '' : 's'}</small></span>
                  </button>
                })}
              </div>
            </section>

            <section className={`panel answer-panel ${mobilePanel === 'answer' ? 'mobile-active' : ''}`} aria-label="Answer and evidence">
              <div className="panel-heading"><div><span className="eyebrow">Evaluation query</span><h2>{demoCase.question}</h2></div></div>
              <div className={`answer-block ${demoCase.answer.status}`}>
                <div className="answer-status">{demoCase.answer.status === 'abstained' ? <ShieldCheck size={17} /> : <Check size={17} />}<b>{demoCase.answer.status}</b><span>{Math.round(demoCase.answer.confidence * 100)}% confidence</span></div>
                <p>{demoCase.answer.body}</p>
                {demoCase.answer.abstention_reason && <small>{demoCase.answer.abstention_reason}</small>}
              </div>
              <div className="evidence-section">
                <h3>Retrieved evidence</h3>
                {demoCase.answer.evidence.length === 0 ? <div className="empty-evidence">No evidence retrieved. The answer abstains.</div> : demoCase.answer.evidence.map((id) => {
                  const evidence = demoCase.source_events.find((item) => item.evidence_id === id)
                  if (!evidence) return null
                  return <button key={id} className={`evidence-chip ${selectedEvidence === id ? 'selected' : ''}`} onClick={() => setSelectedEvidence(id)}><CircleDot size={13} /><span><b>{evidence.speaker}</b>{evidence.text}</span></button>
                })}
              </div>
              {selectedEvidence && (() => {
                const evidence = demoCase.source_events.find((item) => item.evidence_id === selectedEvidence)
                return evidence ? <div className="inspection"><span>Inspecting</span><b>{evidence.source_id}</b><code>{evidence.message_id}</code></div> : null
              })()}
              <div className={`review-block review-${demoCase.review.result}`}>
                <div><AlertCircle size={16} /><b>{demoCase.review.result === 'correct' ? 'Correct with review note' : 'Partial answer'}</b></div>
                <p>{demoCase.review.explanation}</p>
                {demoCase.review.result !== 'correct' && <details><summary>Expected answer</summary><p>{demoCase.review.expected_answer}</p></details>}
              </div>
            </section>
          </div> : <div className="loading-state">Loading case...</div>}
        </main>
      ) : (
        <main className="scorecard-view">
          <section className="scorecard-header">
            <div><span className="eyebrow">Step 10.4 status</span><h1>Benchmark evidence, without a composite score</h1><p>Development results, incomplete work, execution failures, token use, and cost remain separate.</p></div>
            <div className="score-filters">
              <div className="filter-control"><label htmlFor="baseline-filter">Baseline</label><div className="select-wrap"><select id="baseline-filter" value={baselineFilter} onChange={(event) => setBaselineFilter(event.target.value)}><option value="all">B0-B7</option>{baselineOptions.map(([baseline, label]) => <option key={baseline} value={baseline}>{baseline} - {label}</option>)}</select><ChevronDown size={15} /></div></div>
              <div className="filter-control"><label htmlFor="score-filter">Status</label><div className="select-wrap"><select id="score-filter" value={scoreFilter} onChange={(event) => setScoreFilter(event.target.value)}><option value="all">All statuses</option><option value="pilot_scored">Pilot scored</option><option value="development_scored">Development scored</option><option value="development_partial">Partial</option><option value="not_scored">Not scored</option></select><ChevronDown size={15} /></div></div>
            </div>
          </section>
          <section className="run-strip">
            {runs.map((run) => <article key={run.series_id} className="run-item"><div className="run-title"><span className="run-dot" /><div><b>{run.series_id}</b><span>{run.model}</span></div><StatusTag value={run.status} /></div><div className="run-stats"><span><b>{run.requests}</b>requests</span><span><b>{run.input_tokens.toLocaleString()}</b>input tokens</span><span><b>{run.output_tokens.toLocaleString()}</b>output tokens</span><span><b>{run.cost}</b>recorded cost</span></div><p>{run.note}</p></article>)}
          </section>
          <section className="score-table-wrap">
            <table className="score-table">
              <thead><tr><th>Series</th><th>Baseline</th><th>Status</th><th>Answer accuracy</th><th>Evidence recall</th><th>Recall@10</th><th>Abstention precision</th><th>Mean latency</th><th>Failures</th></tr></thead>
              <tbody>{filteredScores.map((score) => <tr key={`${score.series_id}:${score.baseline_id}`}><td><b>{score.series_id}</b></td><td><b>{score.baseline_id}</b><span>{score.label}</span></td><td><StatusTag value={score.status} /></td><td>{percent(score.answer_accuracy)}</td><td>{percent(score.evidence_recall)}</td><td>{percent(score.recall_at_10)}</td><td>{percent(score.abstention_precision)}</td><td>{score.latency_ms === null ? 'Not measured' : `${score.latency_ms.toFixed(1)} ms`}</td><td>{score.failures ?? 'Not reported'}</td></tr>)}</tbody>
            </table>
            {filteredScores.length === 0 && <div className="empty-state"><Search size={22} /><span>No baselines match this filter.</span></div>}
          </section>
          <section className="score-notes"><div><ArrowRight size={16} /><p><b>B1:</b> 92% strict answer accuracy, but only 50% message-level evidence recall. A correct answer can still have an incomplete audit trail.</p></div><div><ArrowRight size={16} /><p><b>B7:</b> 100% abstention recall but 25% precision on the small development slice. It avoided false answers by abstaining too often.</p></div></section>
        </main>
      )}
    </div>
  )
}

export default App
