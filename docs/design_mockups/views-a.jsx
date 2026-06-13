// Views 1-4: Overview, Pipeline, Browse States, Validation
const { useState, useEffect, useMemo } = React;
const D = window.VIEB_DATA;

// =============== OVERVIEW ===============
function OverviewView({ navigate }) {
  const [groupBy, setGroupBy] = useState('Genotype × Drug');
  const [showInd, setShowInd] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [cohortA, setCohortA] = useState('WT–Saline');
  const [cohortB, setCohortB] = useState('KO–CNO');

  return (
    <div className="view-pad">
      <div className="view-header">
        <div>
          <h1 className="view-title">Overview</h1>
          <p className="view-subtitle">How did the experiment go?</p>
        </div>
        <div style={{display:'flex', gap:8, alignItems:'center'}}>
          <span style={{fontSize:11, color:'var(--text-tertiary)', fontFamily:"'IBM Plex Mono', monospace"}}>FearCond_2026_Q2</span>
          <button className="btn">Export figures</button>
        </div>
      </div>

      <div className="stat-strip">
        <div className="stat-card">
          <div className="stat-num">{D.summary.videos}</div>
          <div className="stat-label">Total Videos</div>
          <div className="stat-delta">across 22 animals · 12 days</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">{D.summary.frames}</div>
          <div className="stat-label">Total Frames</div>
          <div className="stat-delta">@ 30 fps · 39.9 h</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">{D.summary.states}</div>
          <div className="stat-label">States Discovered</div>
          <div className="stat-delta">61 retained · 1 dominant (excluded)</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">{D.summary.noise}%</div>
          <div className="stat-label">Noise %</div>
          <div className="stat-delta">frames below confidence threshold</div>
        </div>
      </div>

      <div className="chart-card" style={{marginBottom: 20}}>
        <div className="chart-head-row">
          <div>
            <h2 className="chart-title">Did the animals learn?</h2>
            <p className="chart-subtitle">Discrimination ratio per day, by cohort. Above zero = preference for safe context. Below zero = preference for fear context.</p>
          </div>
          <div style={{display:'flex', gap:8, alignItems:'center'}}>
            <label style={{fontSize:11, color:'var(--text-secondary)', display:'flex', alignItems:'center', gap:6}}>
              <input type="checkbox" checked={showInd} onChange={e => setShowInd(e.target.checked)}/>
              show individuals
            </label>
            <select className="dropdown" value={groupBy} onChange={e => setGroupBy(e.target.value)}>
              <option>Genotype × Drug</option>
              <option>Sex</option>
              <option>Cage</option>
              <option>Pooled</option>
            </select>
          </div>
        </div>
        <LearningCurves data={D.learning} showIndividual={showInd}/>
        <div className="legend" style={{marginTop: 14}}>
          {D.learning.map((d, i) => (
            <span key={i} className="legend-item" style={{color: d.cohort.color}}>
              <span className="legend-swatch"></span>
              <span style={{color:'var(--text-secondary)'}}>{d.cohort.name}</span>
              <span style={{color:'var(--text-tertiary)'}}>n={d.cohort.n}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="two-col" style={{marginBottom: 20}}>
        <div className="chart-card">
          <h2 className="chart-title">State Occupancy by Cohort</h2>
          <p className="chart-subtitle">Mean fraction of frames spent in each state, ±SE across animals. Dominant state excluded.</p>
          <GroupedBars data={D.stateOccupancy} cohorts={D.COHORTS}/>
        </div>
        <div className="chart-card">
          <h2 className="chart-title">Fear-Enriched States</h2>
          <p className="chart-subtitle">Difference in state occupancy between Context A (fear) and Context B (safe), sorted. Red above zero = fear-enriched.</p>
          <DivergingBars data={D.fearStates}/>
        </div>
      </div>

      <div className="card">
        <div className="collapsible-head" onClick={() => setExpanded(e => !e)}>
          <div>
            <div style={{fontWeight:500, fontSize:13}}>Notable Distinctions</div>
            <div style={{fontSize:12, color:'var(--text-secondary)', marginTop:2}}>
              States with significant cohort differences ({expanded ? 'collapse' : 'expand'})
            </div>
          </div>
          <span style={{color:'var(--text-tertiary)', fontSize:18, transform: expanded ? 'rotate(90deg)' : '', transition: 'transform 0.15s'}}>›</span>
        </div>
        {expanded && (
          <div style={{padding: '14px 22px 20px'}}>
            <div style={{display:'flex', gap:10, alignItems:'center', marginBottom: 14}}>
              <span style={{fontSize:11, color:'var(--text-tertiary)', textTransform:'uppercase', letterSpacing:'0.1em'}}>compare</span>
              <select className="dropdown" value={cohortA} onChange={e => setCohortA(e.target.value)}>
                {D.COHORTS.map(c => <option key={c.id}>{c.name}</option>)}
              </select>
              <span style={{color:'var(--text-tertiary)'}}>vs</span>
              <select className="dropdown" value={cohortB} onChange={e => setCohortB(e.target.value)}>
                {D.COHORTS.map(c => <option key={c.id}>{c.name}</option>)}
              </select>
            </div>
            <table className="data-table">
              <thead><tr>
                <th>State</th><th>Label</th><th>{cohortA} (%)</th><th>{cohortB} (%)</th>
                <th>Δ</th><th>p-value</th><th>Cohen's d</th>
              </tr></thead>
              <tbody>
                {[
                  {s:37, l:'freezing-prolonged', a:8.42, b:2.18, d:6.24, p:'0.0008', cd:1.84},
                  {s:14, l:'stretch-attend',     a:4.91, b:1.05, d:3.86, p:'0.0021', cd:1.42},
                  {s:51, l:'darting',            a:0.62, b:3.81, d:-3.19, p:'0.0044', cd:-1.18},
                  {s:23, l:'wall-following',     a:11.21, b:7.04, d:4.17, p:'0.0091', cd:0.96},
                  {s:9,  l:'grooming-rear',      a:5.12, b:8.83, d:-3.71, p:'0.0163', cd:-0.81},
                  {s:42, l:'rearing-supported',  a:1.92, b:4.55, d:-2.63, p:'0.0285', cd:-0.72}
                ].map((r, i) => (
                  <tr key={i}>
                    <td>S{String(r.s).padStart(2,'0')}</td>
                    <td style={{fontFamily:'Inter', fontStyle:'italic', color:'var(--text-secondary)'}}>{r.l}</td>
                    <td>{r.a.toFixed(2)}</td>
                    <td>{r.b.toFixed(2)}</td>
                    <td className={r.d > 0 ? 'pos' : 'neg'}>{r.d > 0 ? '+' : ''}{r.d.toFixed(2)}</td>
                    <td>{r.p}</td>
                    <td>{r.cd > 0 ? '+' : ''}{r.cd.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// =============== PIPELINE ===============
function PipelineView() {
  const [open, setOpen] = useState(null);
  const stages = D.pipeline;

  return (
    <div className="view-pad">
      <div className="view-header">
        <div>
          <h1 className="view-title">Pipeline</h1>
          <p className="view-subtitle">Eleven stages from raw video to quantification. Run all, or pick up from a specific stage.</p>
        </div>
        <div style={{display:'flex', gap:8}}>
          <button className="btn">Pause</button>
          <button className="btn-primary btn">▶ Run All</button>
        </div>
      </div>

      <div className="pipeline-list">
        {stages.map((s, i) => {
          const isOpen = open === i;
          const active = s.status === 'running';
          return (
            <div key={i} className={`pipeline-row ${active ? 'active' : ''}`}>
              <div className="pipeline-head" onClick={() => setOpen(isOpen ? null : i)}>
                <div className="pipeline-num">{String(s.num).padStart(2,'0')}</div>
                <div>
                  <div className="pipeline-name">{s.name}</div>
                  <div className="pipeline-desc">{s.desc}</div>
                </div>
                <div className="pipeline-status">
                  <span className="run-from-here" onClick={e => e.stopPropagation()}>
                    <button className="btn" style={{padding:'3px 9px', fontSize:11}}>Run from here</button>
                  </span>
                  <span>{s.ts}</span>
                  <StatusIcon status={s.status}/>
                </div>
                <span style={{color:'var(--text-tertiary)', fontSize:14, transform: isOpen ? 'rotate(90deg)' : '', transition: 'transform 0.15s'}}>›</span>
              </div>
              {isOpen && <PipelineExpand stage={s}/>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StatusIcon({ status }) {
  if (status === 'complete') return <span className="status-icon status-complete">✓</span>;
  if (status === 'running')  return <span className="status-icon status-running" style={{
    background: 'transparent', border: '2px solid var(--warning)', borderTopColor: 'transparent',
    borderRadius: '50%', animation: 'spin 1s linear infinite'
  }}></span>;
  if (status === 'error') return <span className="status-icon status-error">✕</span>;
  return <span className="status-icon status-not-run"></span>;
}

function PipelineExpand({ stage }) {
  const params = paramsFor(stage.num);
  return (
    <div className="pipeline-expand">
      <div>
        <div className="section-label">CLI command</div>
        <div className="cli-block">
          <span className="prompt">$</span> vieb run <span className="flag">--stage</span> {stage.num} <span className="flag">--project</span> FearCond_2026_Q2 <span className="flag">--workers</span> 8
        </div>
        <div className="section-label" style={{marginTop: 18}}>Parameters</div>
        <div>
          {params.map((p, i) => (
            <div key={i} className="param-row">
              <label>{p.label}</label>
              {p.type === 'number' && <input className="input" type="number" defaultValue={p.value} style={{width:120}}/>}
              {p.type === 'text' && <input className="input" type="text" defaultValue={p.value}/>}
              {p.type === 'select' && (
                <select className="dropdown" defaultValue={p.value}>
                  {p.options.map(o => <option key={o}>{o}</option>)}
                </select>
              )}
              {p.type === 'check' && <label style={{fontSize:12, color:'var(--text-secondary)'}}><input type="checkbox" defaultChecked={p.value}/> {p.help}</label>}
            </div>
          ))}
        </div>
      </div>
      <div>
        <div className="section-label">Last 20 lines of log</div>
        <div className="log-block">
          {sampleLog(stage)}
        </div>
      </div>
    </div>
  );
}

function paramsFor(stage) {
  switch(stage) {
    case 0: return [
      {label: 'DLC project path', type: 'text', value: '~/vieb/dlc/mouse_topdown-2025-06'},
      {label: 'Model snapshot',   type: 'select', value: 'snapshot-450000', options: ['snapshot-300000','snapshot-450000','latest']},
      {label: 'Confidence cutoff',type: 'number', value: 0.6},
      {label: 'GPU acceleration', type: 'check',  value: true, help: 'use CUDA when available'}
    ];
    case 4: return [
      {label: 'Algorithm',        type: 'select', value: 'AR-HMM (autoregressive)', options:['AR-HMM (autoregressive)','GMM','k-means']},
      {label: 'Min cluster size', type: 'number', value: 80},
      {label: 'Max states',       type: 'number', value: 80},
      {label: 'Seed',             type: 'number', value: 42}
    ];
    case 10: return [
      {label: 'Correlation method', type: 'select', value: 'Pearson', options:['Pearson','Spearman','Kendall']},
      {label: 'p-threshold',        type: 'number', value: 0.05},
      {label: 'FDR correction',     type: 'check',  value: true, help: 'Benjamini–Hochberg'}
    ];
    default: return [
      {label: 'FPS',              type: 'number', value: 30},
      {label: 'Window (frames)',  type: 'number', value: 15},
      {label: 'Smoothing kernel', type: 'select', value: 'gaussian', options:['gaussian','median','none']}
    ];
  }
}

function sampleLog(stage) {
  if (stage.status === 'complete') {
    return (
      <>
        <div>14:21:01  loading 222 videos from raw/</div>
        <div>14:21:02  scanning poses_h5/  found 222 files</div>
        <div>14:21:02  computing kinematic features (window=15)</div>
        <div>14:21:04  [████████████████████] 100% · 4,311,872 frames</div>
        <div>14:21:07  writing features.parquet (1.2 GB)</div>
        <div>14:21:09  <span className="ok">✓ stage {stage.num} ({stage.name}) complete</span></div>
      </>
    );
  }
  if (stage.status === 'running') {
    return (
      <>
        <div>16:11:08  starting cohort aggregation across 22 animals</div>
        <div>16:11:09  building per-animal occupancy vectors</div>
        <div>16:11:11  ▏ M-2407 done</div>
        <div>16:11:13  ▎ M-2414 done</div>
        <div>16:11:14  ▍ M-2421 done</div>
        <div>16:11:16  <span className="warn">⚠ M-2428 missing day 7 video — skipping</span></div>
        <div>16:11:17  ▌ M-2435 done</div>
        <div>16:11:19  ▋ M-2442 done</div>
        <div>16:11:21  ▊ M-2449 done · 7/22 (32%)</div>
      </>
    );
  }
  return <div style={{color:'var(--text-tertiary)', fontStyle:'italic'}}>No log output. Run this stage to generate output.</div>;
}

// =============== BROWSE STATES ===============
function BrowseStatesView() {
  const [selected, setSelected] = useState(D.stateCatalog[36] || D.stateCatalog[0]);
  const [sort, setSort] = useState('id');
  const [search, setSearch] = useState('');
  const [modalThumb, setModalThumb] = useState(null);
  const [page, setPage] = useState(1);

  const sorted = useMemo(() => {
    let list = D.stateCatalog.slice();
    if (search) list = list.filter(s => s.label.includes(search.toLowerCase()) || String(s.id).includes(search));
    list.sort((a,b) => {
      if (sort === 'id') return a.id - b.id;
      if (sort === 'occupancy') return b.occupancy - a.occupancy;
      if (sort === 'duration') return b.duration - a.duration;
      return 0;
    });
    return list;
  }, [sort, search]);

  return (
    <div style={{padding: '24px 32px'}}>
      <div className="view-header">
        <div>
          <h1 className="view-title">Browse States</h1>
          <p className="view-subtitle">Inspect discovered behavioral states — kinematics, distribution, and exemplar clips.</p>
        </div>
      </div>

      <div className="browse-layout">
        <div className="state-list-panel">
          <div className="state-list-head">
            <input className="search-input" placeholder="Search states…" value={search} onChange={e => setSearch(e.target.value)}/>
            <select className="dropdown" value={sort} onChange={e => setSort(e.target.value)}>
              <option value="id">By ID</option>
              <option value="occupancy">By occupancy</option>
              <option value="duration">By bout duration</option>
            </select>
            <div style={{fontSize:11, color:'var(--text-tertiary)', fontFamily:"'IBM Plex Mono', monospace"}}>
              {sorted.length} states · 1 dominant excluded
            </div>
          </div>
          <div className="state-list-scroll">
            {sorted.map(s => (
              <div key={s.id}
                   className={`state-card ${selected.id === s.id ? 'selected' : ''}`}
                   onClick={() => setSelected(s)}>
                <div>
                  <div className="state-card-num">{s.id}</div>
                </div>
                <div>
                  <div className="state-card-label">{s.label}</div>
                  <div className="state-card-speed" style={{
                    background: '#ECECEC',
                    width: 60
                  }}>
                    <div style={{
                      width: `${s.speed * 100}%`, height: '100%',
                      background: '#4E79A7', borderRadius: 2
                    }}></div>
                  </div>
                </div>
                <div className={`ctx-badge ctx-${s.ctx}`}>{s.ctx}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="state-detail-panel">
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
            <div>
              <h2 style={{margin:0, fontSize:18, fontWeight:600}}>
                State {selected.id} <span style={{color:'var(--text-tertiary)', fontWeight:400}}>—</span> <span style={{fontStyle:'italic', color:'var(--text-secondary)', fontWeight:400}}>{selected.label}</span>
              </h2>
              <p style={{margin:'6px 0 0', fontSize:12, color:'var(--text-secondary)', fontFamily:"'IBM Plex Mono', monospace"}}>
                Found in {selected.animals} animals · {selected.sessions} sessions · {(selected.occupancy*100).toFixed(2)}% of frames
              </p>
            </div>
            <div style={{display:'flex', gap:8}}>
              <span className={`ctx-badge ctx-${selected.ctx}`}>context {selected.ctx}</span>
              <button className="btn">Add note</button>
            </div>
          </div>

          <div style={{marginTop:18}}>
            <div className="section-label">Kinematic signature</div>
            <KinematicBars kin={selected.kinematics}/>
          </div>

          <div className="filter-row">
            <span style={{fontSize:11, color:'var(--text-tertiary)', textTransform:'uppercase', letterSpacing:'0.1em'}}>filter</span>
            <span className="chip active">All Animals</span>
            <span className="chip">All Contexts</span>
            <span className="chip">All Days</span>
            <span style={{flex:1}}></span>
            <span style={{fontSize:11, color:'var(--text-tertiary)', fontFamily:"'IBM Plex Mono', monospace"}}>48 clips</span>
          </div>

          <div className="thumb-grid">
            {Array.from({length: 8}, (_, i) => (
              <div key={i} className="thumb" onClick={() => setModalThumb(i)}>
                <div className="thumb-overlay">
                  <div className="top">
                    <span>M-{(2400 + i*7).toString().padStart(4,'0')}</span>
                    <span className="ctx-badge ctx-A" style={{padding:'1px 4px'}}>A</span>
                  </div>
                  <div className="bot">
                    <span>D{(i%12)+1}</span>
                    <span>{(0.8 + i*0.4).toFixed(1)}s</span>
                  </div>
                </div>
                <div className="thumb-play"></div>
              </div>
            ))}
          </div>

          <div className="pagination">
            <span style={{cursor:'pointer'}} onClick={() => setPage(Math.max(1, page-1))}>← Previous</span>
            <span>{page} of 12</span>
            <span style={{cursor:'pointer'}} onClick={() => setPage(Math.min(12, page+1))}>Next →</span>
          </div>
        </div>
      </div>

      {modalThumb !== null && (
        <div className="modal-backdrop" onClick={() => setModalThumb(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <div style={{fontSize:13, fontWeight:500}}>M-{(2400 + modalThumb*7).toString().padStart(4,'0')} · S{selected.id} ({selected.label})</div>
                <div style={{fontSize:11, color:'var(--text-tertiary)', fontFamily:"'IBM Plex Mono', monospace", marginTop:2}}>
                  Day {(modalThumb%12)+1} · Context A · {(0.8 + modalThumb*0.4).toFixed(1)}s · frames 1402–1428
                </div>
              </div>
              <span style={{cursor:'pointer', color:'var(--text-tertiary)', fontSize:18}} onClick={() => setModalThumb(null)}>×</span>
            </div>
            <div className="video-stage">
              <KeypointOverlay/>
            </div>
            <div className="video-controls">
              <div className="play-circle"></div>
              <span>0:00</span>
              <div className="seek"><div className="seek-fill"></div></div>
              <span>0:03</span>
              <span style={{borderLeft:'1px solid var(--border)', paddingLeft:12, marginLeft:8}}>
                <span style={{cursor:'pointer'}}>⟲ loop</span>
                <span style={{margin:'0 12px', color:'var(--text-tertiary)'}}>|</span>
                <span style={{cursor:'pointer'}}>1× speed</span>
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// =============== VALIDATION ===============
function ValidationView() {
  const [labeled, setLabeled] = useState(24);
  const total = 50;
  const [currentLabel, setCurrentLabel] = useState(null);

  function pickLabel(lbl) {
    setCurrentLabel(lbl);
    if (labeled < total) setLabeled(l => l + 1);
    setTimeout(() => setCurrentLabel(null), 250);
  }

  return (
    <div className="view-pad">
      <div className="view-header">
        <div>
          <h1 className="view-title">Validation</h1>
          <p className="view-subtitle">Hand-label a sample of frames to estimate inter-rater agreement with discovered states.</p>
        </div>
      </div>

      <div className="validation-layout">
        {/* Left: sampling controls */}
        <div className="card card-pad" style={{display:'flex', flexDirection:'column', gap:14}}>
          <div className="section-label" style={{marginBottom:0}}>Sampling</div>
          <div>
            <label style={{fontSize:11, color:'var(--text-secondary)'}}>Rater</label>
            <input className="input" defaultValue="A. Tanaka" style={{width:'100%', marginTop:4}}/>
          </div>
          <div>
            <label style={{fontSize:11, color:'var(--text-secondary)'}}>Sample source</label>
            <select className="dropdown" defaultValue="By state" style={{width:'100%', marginTop:4}}>
              <option>By state</option>
              <option>By video</option>
            </select>
          </div>
          <div>
            <label style={{fontSize:11, color:'var(--text-secondary)'}}>State</label>
            <select className="dropdown" defaultValue="S37 — freezing-prolonged" style={{width:'100%', marginTop:4}}>
              <option>S37 — freezing-prolonged</option>
              <option>S14 — stretch-attend</option>
              <option>S51 — darting</option>
            </select>
          </div>
          <div>
            <label style={{fontSize:11, color:'var(--text-secondary)'}}>Video</label>
            <select className="dropdown" defaultValue="(any)" style={{width:'100%', marginTop:4}}>
              <option>(any)</option>
            </select>
          </div>
          <div>
            <label style={{fontSize:11, color:'var(--text-secondary)', display:'flex', justifyContent:'space-between'}}>
              <span>Frames</span><span style={{fontFamily:"'IBM Plex Mono', monospace"}}>{total}</span>
            </label>
            <input type="range" min="10" max="200" defaultValue={total} style={{width:'100%', accentColor:'#4E79A7'}}/>
          </div>
          <button className="btn-primary btn" style={{width:'100%', justifyContent:'center'}}>Sample Frames</button>

          <div>
            <div style={{fontSize:11, color:'var(--text-secondary)', display:'flex', justifyContent:'space-between'}}>
              <span>Progress</span>
              <span style={{fontFamily:"'IBM Plex Mono', monospace"}}>{labeled} of {total}</span>
            </div>
            <div className="progress-bar"><div className="progress-fill" style={{width: `${labeled/total*100}%`}}></div></div>
          </div>

          <div className="shortcut-card">
            <div style={{fontWeight:500, color:'var(--text-primary)', marginBottom:4}}>Keyboard</div>
            <div className="row"><span>Label freeze</span><span className="kbd">F</span></div>
            <div className="row"><span>Label walk</span><span className="kbd">W</span></div>
            <div className="row"><span>Label groom</span><span className="kbd">G</span></div>
            <div className="row"><span>Label rear</span><span className="kbd">R</span></div>
            <div className="row"><span>Other / skip</span><span className="kbd">O / S</span></div>
          </div>
        </div>

        {/* Center: frame */}
        <div className="frame-stage">
          <div className="frame-canvas">
            <KeypointOverlay/>
            <div style={{position:'absolute', top:10, left:12, fontFamily:"'IBM Plex Mono', monospace", fontSize:11, color:'rgba(255,255,255,0.85)'}}>
              frame 18,237 / 79,200
            </div>
            <div style={{position:'absolute', top:10, right:12, display:'flex', gap:6}}>
              <span className="ctx-badge ctx-A">A</span>
            </div>
            {currentLabel && (
              <div style={{
                position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center',
                background: 'rgba(78,121,167,0.18)', color: 'white',
                fontSize: 28, fontWeight: 600, letterSpacing: '0.05em', pointerEvents:'none'
              }}>
                {currentLabel}
              </div>
            )}
          </div>
          <div className="frame-info">
            <span>M-2421 · Day 6 · Context A · trial 3</span>
            <span>auto label: <span style={{color:'var(--c1)'}}>S37 freezing-prolonged</span> (conf 0.84)</span>
          </div>
          <div style={{marginTop:18}}>
            <div className="section-label">Running confusion matrix</div>
            <div style={{display:'flex', gap:24, alignItems:'flex-start'}}>
              <ConfusionMatrix/>
              <table className="data-table" style={{flex:1}}>
                <thead><tr><th>State</th><th>Auto</th><th>Manual</th><th>Agreement</th></tr></thead>
                <tbody>
                  {[
                    ['Freeze', 44, 42, 95.5],
                    ['Walk',   42, 38, 90.5],
                    ['Groom',  24, 22, 91.7],
                    ['Rear',   20, 18, 90.0],
                    ['Other',  19, 14, 73.7]
                  ].map((r, i) => (
                    <tr key={i}>
                      <td style={{fontFamily:'Inter'}}>{r[0]}</td>
                      <td>{r[1]}</td>
                      <td>{r[2]}</td>
                      <td>{r[3].toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{display:'flex', justifyContent:'flex-end', marginTop:14}}>
              <button className="btn">Export labels</button>
            </div>
          </div>
        </div>

        {/* Right: label buttons */}
        <div>
          <div className="section-label">Label this frame</div>
          <div className="label-stack">
            {[
              ['Freeze','F'], ['Walk','W'], ['Groom','G'], ['Rear','R'], ['Other','O']
            ].map(([lbl, k]) => (
              <div key={lbl} className="label-btn" onClick={() => pickLabel(lbl)}>
                <span>{lbl}</span>
                <span className="kbd">{k}</span>
              </div>
            ))}
            <div className="label-btn" style={{justifyContent:'center', color:'var(--text-secondary)'}} onClick={() => pickLabel('Skip')}>
              <span>Skip frame</span>
              <span className="kbd">S</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, {
  OverviewView, PipelineView, BrowseStatesView, ValidationView
});
