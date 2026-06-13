// Views 5-7: Quantification, Advanced, Settings
const D2 = window.VIEB_DATA;

// =============== QUANTIFICATION ===============
function QuantificationView() {
  const [tab, setTab] = useState('master');

  return (
    <div className="view-pad">
      <div className="view-header">
        <div>
          <h1 className="view-title">Quantification</h1>
          <p className="view-subtitle">Per-animal behavioral variables, learning curves, and molecular correlations.</p>
        </div>
        <div style={{display:'flex', gap:8}}>
          <button className="btn">Export CSV</button>
          <button className="btn-primary btn">Run Quantification</button>
        </div>
      </div>

      <div className="tabs">
        {[
          ['master','Master Table'],
          ['learning','Learning Curves'],
          ['jess','Import Jess']
        ].map(([k,v]) => (
          <div key={k} className={`tab ${tab===k?'active':''}`} onClick={() => setTab(k)}>{v}</div>
        ))}
      </div>

      {tab === 'master' && <MasterTable/>}
      {tab === 'learning' && <LearningTab/>}
      {tab === 'jess' && <JessTab/>}
    </div>
  );
}

function MasterTable() {
  const groups = [
    {label: 'Identity', cols: ['animal','cohort','sex'], span: 3},
    {label: 'Fear', cols: ['day'], span: 1},
    {label: 'Learning', cols: ['discr','thr','asymp'], span: 3},
    {label: 'Freezing', cols: ['freezing','bouts'], span: 2},
    {label: 'Behavioral', cols: ['occ'], span: 1},
    {label: 'Transitions', cols: ['trans'], span: 1},
    {label: 'Deviation', cols: ['dev'], span: 1}
  ];
  const colHeaders = {
    animal:'Animal', cohort:'Cohort', sex:'Sex',
    day:'Probe day', discr:'Disc. ratio', thr:'Threshold (d)', asymp:'Asymptote',
    freezing:'Freezing %', bouts:'# Bouts', occ:'Fear-enr. occ.',
    trans:'Trans. rate', dev:'Pop. deviation'
  };
  function cellColor(col, val) {
    const v = parseFloat(val);
    if (col === 'discr')    return v > 0 ? `rgba(192,57,43,${Math.min(1, Math.abs(v))*0.18})` : `rgba(41,128,185,${Math.min(1, Math.abs(v))*0.18})`;
    if (col === 'freezing') return `rgba(192,57,43,${Math.min(1, v/60)*0.16})`;
    if (col === 'occ')      return `rgba(192,57,43,${Math.min(1, v*8)*0.16})`;
    if (col === 'dev')      return v > 0 ? `rgba(192,57,43,${Math.min(1, Math.abs(v))*0.18})` : `rgba(41,128,185,${Math.min(1, Math.abs(v))*0.18})`;
    return 'transparent';
  }
  return (
    <div className="master-table-wrap">
      <div style={{overflowX:'auto'}}>
        <table className="master-table">
          <thead>
            <tr>
              {groups.map((g, i) => (
                <th key={i} colSpan={g.span} className="col-group"
                    style={{borderLeft: i>0 ? '1px solid var(--border)' : 'none'}}>{g.label}</th>
              ))}
            </tr>
            <tr>
              {groups.flatMap((g, gi) => g.cols.map((c, ci) => (
                <th key={`${gi}-${ci}`} style={{borderLeft: gi>0 && ci===0 ? '1px solid var(--border)' : 'none'}}>
                  {colHeaders[c]}
                </th>
              )))}
            </tr>
          </thead>
          <tbody>
            {D2.masterRows.map((r, i) => (
              <tr key={i} style={{background: `${r.cohort.color}08`}}>
                {groups.flatMap((g, gi) => g.cols.map((c, ci) => {
                  let val;
                  if (c === 'animal') val = r.animal;
                  else if (c === 'cohort') val = r.cohort.name;
                  else val = r[c];
                  return (
                    <td key={`${gi}-${ci}`}
                        style={{
                          borderLeft: gi>0 && ci===0 ? '1px solid var(--border)' : 'none',
                          background: cellColor(c, val)
                        }}>
                      {c === 'cohort' ? (
                        <span style={{display:'inline-flex', alignItems:'center', gap:6}}>
                          <span style={{width:8, height:8, borderRadius:'50%', background: r.cohort.color}}></span>
                          <span style={{fontFamily:'Inter'}}>{r.cohort.name}</span>
                        </span>
                      ) : val}
                    </td>
                  );
                }))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{padding:'10px 14px', fontSize:11, color:'var(--text-tertiary)', fontFamily:"'IBM Plex Mono', monospace", borderTop: '1px solid var(--border)'}}>
        Showing 14 of 22 animals · Updated 16:08:42 today
      </div>
    </div>
  );
}

function LearningTab() {
  const [groupBy, setGroupBy] = useState('Genotype × Drug');
  return (
    <div>
      <div className="chart-card">
        <div className="chart-head-row">
          <div>
            <h2 className="chart-title">Learning curves with sigmoid fit</h2>
            <p className="chart-subtitle">Faded lines: individual animals. Dashed: fitted sigmoid. Bold: cohort mean. Above zero = preference for safe.</p>
          </div>
          <select className="dropdown" value={groupBy} onChange={e => setGroupBy(e.target.value)}>
            <option>Genotype × Drug</option>
            <option>Sex</option>
            <option>Cage</option>
          </select>
        </div>
        <LearningCurvesSigmoid data={D2.learning}/>
        <div className="legend" style={{marginTop:14, justifyContent:'space-between'}}>
          <div className="legend">
            {D2.learning.map((d, i) => (
              <span key={i} className="legend-item" style={{color: d.cohort.color}}>
                <span className="legend-swatch"></span>
                <span style={{color:'var(--text-secondary)'}}>{d.cohort.name}</span>
              </span>
            ))}
          </div>
          <div className="legend">
            <span className="legend-item">
              <span style={{display:'inline-block', width:14, height:2, background:'var(--text-tertiary)'}}></span>
              <span style={{color:'var(--text-secondary)'}}>cohort mean</span>
            </span>
            <span className="legend-item">
              <span className="dash" style={{color:'var(--text-tertiary)'}}></span>
              <span style={{color:'var(--text-secondary)'}}>per-animal sigmoid</span>
            </span>
          </div>
        </div>
      </div>

      <div style={{marginTop: 20}}>
        <div className="card">
          <div style={{padding:'14px 22px', borderBottom:'1px solid var(--border)', display:'flex', justifyContent:'space-between'}}>
            <div>
              <div style={{fontWeight:500, fontSize:13}}>Fit parameters</div>
              <div style={{fontSize:12, color:'var(--text-secondary)', marginTop:2}}>Three-parameter sigmoid: asymptote, slope, half-max day.</div>
            </div>
            <button className="btn">Copy as TSV</button>
          </div>
          <div style={{padding:'0 22px 18px'}}>
            <table className="data-table">
              <thead><tr>
                <th>Cohort</th><th>n</th><th>Asymptote (95% CI)</th><th>Slope</th>
                <th>Half-max day</th><th>R²</th>
              </tr></thead>
              <tbody>
                {[
                  {c:'WT-Saline', n:6, a:'0.81 (0.74, 0.88)', s:0.92, h:4.8, r2:0.87},
                  {c:'WT-CNO',    n:6, a:'0.42 (0.31, 0.53)', s:0.61, h:6.2, r2:0.71},
                  {c:'KO-Saline', n:5, a:'0.74 (0.65, 0.82)', s:0.84, h:5.1, r2:0.83},
                  {c:'KO-CNO',    n:5, a:'0.18 (0.06, 0.30)', s:0.38, h:8.4, r2:0.54}
                ].map((r, i) => (
                  <tr key={i}>
                    <td style={{fontFamily:'Inter'}}>
                      <span style={{display:'inline-flex', gap:6, alignItems:'center'}}>
                        <span style={{width:8, height:8, borderRadius:'50%', background: D2.COHORTS[i].color}}></span>
                        {r.c}
                      </span>
                    </td>
                    <td>{r.n}</td><td>{r.a}</td><td>{r.s.toFixed(2)}</td><td>{r.h.toFixed(1)}</td><td>{r.r2.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function JessTab() {
  const [imported, setImported] = useState(true);
  const exampleCsv = "animal,protein,family,area,height,sn\nM-2400,BDNF,Neurotrophin,1.21e5,3142,84.2\nM-2400,TrkB,Receptor,8.83e4,2210,71.4\nM-2400,PSD-95,Scaffold,9.41e4,2418,76.8\nM-2407,BDNF,Neurotrophin,...";
  return (
    <div style={{display:'flex', flexDirection:'column', gap:22}}>
      <div className="card card-pad">
        <div style={{display:'grid', gridTemplateColumns:'1fr 380px', gap: 28}}>
          <div>
            <div className="section-label" style={{marginBottom:8}}>Import file</div>
            <div className="dropzone" onClick={() => setImported(true)}>
              <div className="dropzone-icon">⇣</div>
              <div style={{fontSize:13, color:'var(--text-primary)', fontWeight:500}}>Drop Jess output here</div>
              <div style={{fontSize:12, marginTop:6}}>or <span style={{color:'var(--c1)', cursor:'pointer'}}>browse files…</span></div>
              <div style={{fontSize:11, marginTop:14, color:'var(--text-tertiary)', fontFamily:"'IBM Plex Mono', monospace"}}>
                .xlsx, .csv · animal ID column required
              </div>
            </div>
          </div>
          <div>
            <div className="section-label" style={{marginBottom:8}}>Expected format</div>
            <pre className="cli-block" style={{lineHeight:1.7, fontSize:10.5, margin:0, whiteSpace:'pre-wrap'}}>{exampleCsv}</pre>
          </div>
        </div>
        {imported && (
          <div className="match-report" style={{marginTop: 18}}>
            <strong style={{color:'var(--success)', fontFamily:'Inter'}}>✓ 18 of 22 animals matched.</strong>
            <span style={{marginLeft:8, color:'var(--text-secondary)'}}>
              Unmatched: M-2428, M-2435, M-2484, M-2519. Behavioral data for these animals is intact.
            </span>
          </div>
        )}
      </div>

      {imported && (
        <>
          <div className="chart-card">
            <div className="chart-head-row">
              <div>
                <h2 className="chart-title">Behavior × Protein correlations</h2>
                <p className="chart-subtitle">Pearson r between each behavioral variable and each protein measurement. Asterisks mark p &lt; 0.05 (FDR-corrected).</p>
              </div>
              <div style={{display:'flex', gap:8, alignItems:'center'}}>
                <span style={{fontSize:11, color:'var(--text-tertiary)', fontFamily:"'IBM Plex Mono', monospace"}}>n = 18</span>
                <button className="btn-primary btn">Run Correlation</button>
              </div>
            </div>
            <div style={{overflowX:'auto', paddingBottom:8}}>
              <CorrHeatmap corrs={D2.correlations} proteins={D2.proteins} behVars={D2.behVars}/>
            </div>
            <div className="legend" style={{marginTop:12}}>
              <span className="legend-item">
                <span style={{width:14, height:10, background: 'rgb(255,245,238)'}}></span>
                <span style={{color:'var(--text-secondary)'}}>r = 0</span>
              </span>
              <span className="legend-item">
                <span style={{width:14, height:10, background: 'rgb(195,65,43)'}}></span>
                <span style={{color:'var(--text-secondary)'}}>r = +1 (fear-correlated)</span>
              </span>
              <span className="legend-item">
                <span style={{width:14, height:10, background: 'rgb(40,115,188)'}}></span>
                <span style={{color:'var(--text-secondary)'}}>r = -1 (safe-correlated)</span>
              </span>
            </div>
          </div>

          <div className="card">
            <div style={{padding:'14px 22px', borderBottom:'1px solid var(--border)'}}>
              <div style={{fontWeight:500, fontSize:13}}>Top 20 correlations</div>
              <div style={{fontSize:12, color:'var(--text-secondary)', marginTop:2}}>Ranked by absolute Pearson r.</div>
            </div>
            <div style={{padding:'0 22px 18px'}}>
              <table className="data-table">
                <thead><tr><th>#</th><th>Behavioral variable</th><th>Protein</th><th>r</th><th>p</th><th>FDR-q</th></tr></thead>
                <tbody>
                  {D2.correlations.slice(0, 12).map((c, i) => (
                    <tr key={i}>
                      <td>{i+1}</td>
                      <td style={{fontFamily:'Inter'}}>{c.behavior}</td>
                      <td>{c.protein}</td>
                      <td className={c.r > 0 ? 'pos' : 'neg'}>{c.r > 0 ? '+' : ''}{c.r.toFixed(3)}</td>
                      <td>{c.p.toFixed(4)}</td>
                      <td>{(c.p * 1.4).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="two-col">
            <div className="card">
              <div style={{padding:'14px 22px', borderBottom:'1px solid var(--border)'}}>
                <div style={{fontWeight:500, fontSize:13}}>Method Comparison · family × protein</div>
                <div style={{fontSize:12, color:'var(--text-secondary)', marginTop:2}}>Summary R² when behavior is predicted by each protein family.</div>
              </div>
              <div style={{padding:'0 22px 18px'}}>
                <table className="data-table">
                  <thead><tr><th>Family</th><th>n proteins</th><th>Best R²</th><th>Mean R²</th></tr></thead>
                  <tbody>
                    {[
                      ['Neurotrophin', 2, 0.48, 0.36],
                      ['Receptor',     3, 0.61, 0.41],
                      ['Scaffold',     2, 0.42, 0.31],
                      ['IEG',          3, 0.55, 0.39],
                      ['Inhibitory',   2, 0.38, 0.27],
                      ['Transcription',2, 0.46, 0.34]
                    ].map((r, i) => (
                      <tr key={i}>
                        <td style={{fontFamily:'Inter'}}>{r[0]}</td>
                        <td>{r[1]}</td>
                        <td>{r[2].toFixed(2)}</td>
                        <td>{r[3].toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <div style={{padding:'14px 22px', borderBottom:'1px solid var(--border)'}}>
                <div style={{fontWeight:500, fontSize:13}}>Jess to Behavior · ranked R²</div>
                <div style={{fontSize:12, color:'var(--text-secondary)', marginTop:2}}>Single-protein linear model predicting top behavioral variable.</div>
              </div>
              <div style={{padding:'0 22px 18px'}}>
                <table className="data-table">
                  <thead><tr><th>Protein</th><th>Predicts</th><th>R²</th><th>p</th></tr></thead>
                  <tbody>
                    {[
                      ['c-Fos',  'Freezing % (Ctx A)', 0.61, '0.0008'],
                      ['BDNF',   'Disc. ratio (d10)',  0.54, '0.0021'],
                      ['Arc',    'Bout dur. (S37)',    0.49, '0.0044'],
                      ['GluA1',  'Disc. slope',        0.41, '0.0091'],
                      ['PSD-95', 'Asymptote',          0.38, '0.0163'],
                      ['pCREB',  'Threshold day',      0.31, '0.0285']
                    ].map((r, i) => (
                      <tr key={i}>
                        <td>{r[0]}</td>
                        <td style={{fontFamily:'Inter'}}>{r[1]}</td>
                        <td>{r[2].toFixed(2)}</td>
                        <td>{r[3]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// =============== ADVANCED ===============
function AdvancedView() {
  const [tab, setTab] = useState('cohort');
  return (
    <div className="view-pad">
      <div className="view-header">
        <div>
          <h1 className="view-title">Advanced</h1>
          <p className="view-subtitle">Analyst-level diagnostics: transitions, motifs, per-animal exploration, fingerprints.</p>
        </div>
      </div>
      <div className="tabs">
        {[
          ['cohort','Cohort Analysis'],
          ['trans','Transition Matrix'],
          ['motifs','Motifs'],
          ['animal','Animal Explorer'],
          ['fp','Fingerprints']
        ].map(([k,v]) => (
          <div key={k} className={`tab ${tab===k?'active':''}`} onClick={() => setTab(k)}>{v}</div>
        ))}
      </div>

      {tab === 'cohort' && <CohortAnalysisTab/>}
      {tab === 'trans' && <TransitionTab/>}
      {tab === 'motifs' && <MotifsTab/>}
      {tab === 'animal' && <AnimalExplorerTab/>}
      {tab === 'fp' && <FingerprintsTab/>}
    </div>
  );
}

function DescCard({ id, children }) {
  const key = `vieb_desc_${id}`;
  const [shown, setShown] = useState(() => {
    try { return localStorage.getItem(key) !== 'dismissed'; } catch(e) { return true; }
  });
  if (!shown) return null;
  return (
    <div className="desc-card">
      <div style={{flex:1}}>{children}</div>
      <span className="close-x" onClick={() => { try { localStorage.setItem(key, 'dismissed'); } catch(e){} setShown(false); }}>×</span>
    </div>
  );
}

function CohortAnalysisTab() {
  return (
    <>
      <DescCard id="cohort">
        Each bar group is one state. Bars within a group show the cohort means with ±SE.
        Use this to spot states whose occupancy differs by group. The dominant state is excluded.
      </DescCard>
      <div className="two-col">
        <div className="chart-card">
          <h2 className="chart-title">Occupancy distribution</h2>
          <p className="chart-subtitle">Per-cohort mean occupancy across the 12 most prevalent non-dominant states.</p>
          <GroupedBars data={D2.stateOccupancy} cohorts={D2.COHORTS}/>
        </div>
        <div className="chart-card">
          <h2 className="chart-title">Fear vs Safe context</h2>
          <p className="chart-subtitle">State-level delta occupancy (A − B). Red bars rise in fear, blue bars rise in safe.</p>
          <DivergingBars data={D2.fearStates}/>
        </div>
      </div>

      <div className="card" style={{marginTop:20}}>
        <div style={{padding:'14px 22px', borderBottom:'1px solid var(--border)'}}>
          <div style={{fontWeight:500, fontSize:13}}>Effect size matrix · Cohen's d</div>
          <div style={{fontSize:12, color:'var(--text-secondary)', marginTop:2}}>Pairwise effect sizes between cohorts for each behavioral variable.</div>
        </div>
        <div style={{padding:'14px 22px 22px', overflowX:'auto'}}>
          <table className="data-table" style={{minWidth: 760}}>
            <thead><tr>
              <th>Variable</th>
              <th>WT-S × WT-C</th><th>WT-S × KO-S</th><th>WT-S × KO-C</th>
              <th>WT-C × KO-S</th><th>WT-C × KO-C</th><th>KO-S × KO-C</th>
            </tr></thead>
            <tbody>
              {[
                ['Disc. ratio (d10)',  0.42, 0.18, 1.84, -0.21, 1.41, 1.62],
                ['Freezing % (Ctx A)', 0.96, 0.31, 1.42, -0.62, 0.48, 1.04],
                ['Bout dur. (S37)',    0.61, 0.14, 1.18, -0.45, 0.57, 0.99],
                ['Trans. entropy',    -0.34, 0.08, -0.81, 0.42, -0.51, -0.88],
                ['Asymptote',          0.49, -0.06, 1.36, -0.55, 0.84, 1.38]
              ].map((r, i) => (
                <tr key={i}>
                  <td style={{fontFamily:'Inter'}}>{r[0]}</td>
                  {r.slice(1).map((v, vi) => (
                    <td key={vi} style={{
                      background: Math.abs(v) > 0.8 ? `rgba(192,57,43,${Math.min(1,Math.abs(v)/2)*0.18})` :
                                  Math.abs(v) > 0.5 ? `rgba(192,57,43,${Math.min(1,Math.abs(v)/2)*0.1})` : 'transparent',
                      color: v > 0 ? 'var(--fear)' : v < 0 ? 'var(--safe)' : 'var(--text-primary)'
                    }}>{v > 0 ? '+' : ''}{v.toFixed(2)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function TransitionTab() {
  return (
    <>
      <DescCard id="trans">
        Rows are the current state, columns are the next state. Darker cells mean the transition is more
        common. The diagonal is naturally bright because most frames stay in their current state.
      </DescCard>
      <div className="chart-card" style={{overflowX:'auto'}}>
        <h2 className="chart-title">State transition matrix · all animals</h2>
        <p className="chart-subtitle">P(next state | current state) computed over all bout boundaries. Self-loops on the diagonal.</p>
        <TransitionMatrix/>
      </div>
    </>
  );
}

function MotifsTab() {
  const motifs = [
    {seq:['S37','S14','S37'], label:'freeze → stretch → freeze',  count:842, ctx:'A'},
    {seq:['S07','S23','S07'], label:'sniff → wall → sniff',        count:611, ctx:'B'},
    {seq:['S14','S51','S37'], label:'stretch → dart → freeze',     count:418, ctx:'A'},
    {seq:['S09','S05','S09'], label:'groom-rear → rear → groom-rear', count:387, ctx:'B'},
    {seq:['S23','S11','S37'], label:'wall → micro → freeze', count:301, ctx:'A'}
  ];
  return (
    <>
      <DescCard id="motifs">
        A motif is a recurring sequence of states. We surface the most frequent length-3 and length-4 motifs
        and label them by context where they're enriched.
      </DescCard>
      <div className="card">
        <div style={{padding:'14px 22px', borderBottom:'1px solid var(--border)'}}>
          <div style={{fontWeight:500, fontSize:13}}>Top motifs</div>
          <div style={{fontSize:12, color:'var(--text-secondary)', marginTop:2}}>Length-3 sequences, minimum 100 occurrences across cohort.</div>
        </div>
        <div style={{padding:'0 22px 18px'}}>
          <table className="data-table">
            <thead><tr><th>Sequence</th><th>Interpretation</th><th>Count</th><th>Enriched in</th></tr></thead>
            <tbody>
              {motifs.map((m, i) => (
                <tr key={i}>
                  <td>
                    <span style={{display:'inline-flex', gap:4, alignItems:'center'}}>
                      {m.seq.map((s, si) => (
                        <React.Fragment key={si}>
                          <span style={{
                            padding:'2px 8px', border:'1px solid var(--border)', borderRadius:3,
                            background: '#fbfbfb', fontWeight:500
                          }}>{s}</span>
                          {si < m.seq.length - 1 && <span style={{color:'var(--text-tertiary)'}}>→</span>}
                        </React.Fragment>
                      ))}
                    </span>
                  </td>
                  <td style={{fontFamily:'Inter', fontStyle:'italic', color:'var(--text-secondary)'}}>{m.label}</td>
                  <td>{m.count}</td>
                  <td><span className={`ctx-badge ctx-${m.ctx}`}>Context {m.ctx}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function AnimalExplorerTab() {
  const [animal, setAnimal] = useState('M-2407');
  return (
    <>
      <DescCard id="animal">
        Pick an animal to see its individual trajectory: discrimination ratio per day, state occupancy histogram,
        and how it compares to its cohort mean.
      </DescCard>
      <div style={{display:'grid', gridTemplateColumns:'260px 1fr', gap: 16}}>
        <div className="card card-pad">
          <div className="section-label">22 animals</div>
          <div className="animal-list" style={{gridTemplateColumns:'repeat(2, 1fr)'}}>
            {Array.from({length: 22}, (_, i) => {
              const id = `M-${(2400 + i*7).toString().padStart(4,'0')}`;
              const c = D2.COHORTS[i % 4];
              return (
                <div key={id} className="animal-tile" style={{
                  borderColor: animal === id ? c.color : 'var(--border)',
                  background: animal === id ? `${c.color}10` : 'white'
                }} onClick={() => setAnimal(id)}>
                  <div className="animal-id">{id}</div>
                  <div className="animal-cohort" style={{color: c.color}}>{c.name}</div>
                </div>
              );
            })}
          </div>
        </div>
        <div style={{display:'flex', flexDirection:'column', gap:16}}>
          <div className="card card-pad">
            <div style={{display:'flex', justifyContent:'space-between'}}>
              <div>
                <h2 style={{margin:0, fontSize:18, fontWeight:600}}>{animal}</h2>
                <p style={{margin:'4px 0 0', fontSize:12, color:'var(--text-secondary)', fontFamily:"'IBM Plex Mono', monospace"}}>
                  WT-Saline · Male · 12 sessions · 219,148 frames
                </p>
              </div>
              <button className="btn">Open in Browser</button>
            </div>
          </div>
          <div className="chart-card">
            <h2 className="chart-title">Discrimination ratio · this animal vs cohort</h2>
            <p className="chart-subtitle">Bold: this animal. Faded: same-cohort siblings.</p>
            <LearningCurves data={[D2.learning[0]]} showIndividual={true}/>
          </div>
          <div className="two-col">
            <div className="chart-card">
              <h2 className="chart-title">Occupancy histogram</h2>
              <p className="chart-subtitle">Fraction of frames in each state. Compare to cohort mean.</p>
              <GroupedBars data={D2.stateOccupancy.slice(0,8)} cohorts={D2.COHORTS.slice(0,2)}/>
            </div>
            <div className="card card-pad">
              <div className="section-label">Daily snapshot</div>
              <table className="data-table">
                <thead><tr><th>Day</th><th>Frames</th><th>Disc.</th><th>Freeze %</th></tr></thead>
                <tbody>
                  {Array.from({length:12}, (_,i) => (
                    <tr key={i}>
                      <td>D{i+1}</td>
                      <td>{(17000 + i*340).toLocaleString()}</td>
                      <td className={i > 3 ? 'pos' : ''}>{(i/12*0.9 - 0.2).toFixed(2)}</td>
                      <td>{(2 + i*3.2).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function FingerprintsTab() {
  return (
    <>
      <DescCard id="fp">
        Each row is one animal. Each column is one state. Cell shade = occupancy. Use the dropdown to sort animals
        by cohort, by similarity, or by total fear-state engagement.
      </DescCard>
      <div className="chart-card">
        <div className="chart-head-row">
          <div>
            <h2 className="chart-title">Behavioral fingerprints · 18 × 22</h2>
            <p className="chart-subtitle">Per-animal occupancy across non-dominant states. Sorted by cohort.</p>
          </div>
          <div style={{display:'flex', gap:8}}>
            <select className="dropdown" defaultValue="By cohort">
              <option>By cohort</option>
              <option>By similarity</option>
              <option>By fear engagement</option>
            </select>
            <button className="btn">Cluster rows</button>
          </div>
        </div>
        <div style={{overflowX:'auto'}}>
          <Fingerprint data={D2.fingerprint}/>
        </div>
        <div className="legend" style={{marginTop:12}}>
          <span className="legend-item">
            <span style={{width:14, height:10, background:'rgb(255,245,230)'}}></span>
            <span style={{color:'var(--text-secondary)'}}>0% occupancy</span>
          </span>
          <span className="legend-item">
            <span style={{width:14, height:10, background:'rgb(195,75,30)'}}></span>
            <span style={{color:'var(--text-secondary)'}}>≥ 12% occupancy</span>
          </span>
        </div>
      </div>
    </>
  );
}

// =============== SETTINGS ===============
function SettingsView() {
  return (
    <div className="view-pad">
      <div className="view-header">
        <div>
          <h1 className="view-title">Settings</h1>
          <p className="view-subtitle">Project paths, arena geometry, and analysis parameters.</p>
        </div>
        <button className="btn">Save changes</button>
      </div>

      <div className="card" style={{padding: '0 28px'}}>
        <div className="settings-section">
          <div className="settings-grid">
            <div>
              <h3>Project paths</h3>
              <p className="sec-desc">Where VIEB reads videos from and writes analysis outputs.</p>
            </div>
            <div>
              <Field label="Results directory" defaultValue="/home/lab/vieb/projects/FearCond_2026_Q2/results"/>
              <Field label="Raw videos directory" defaultValue="/mnt/storage/raw/FearCond_2026_Q2"/>
              <Field label="Poses (DLC h5)"        defaultValue="/home/lab/vieb/projects/FearCond_2026_Q2/poses_h5"/>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <div className="settings-grid">
            <div>
              <h3>Arena bounds</h3>
              <p className="sec-desc">Pixel coordinates of the four arena walls. Used to compute wall distance and exclude out-of-arena tracking errors.</p>
              <div style={{marginTop:12}}><ArenaDiagram/></div>
            </div>
            <div>
              <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'12px 28px'}}>
                <Field label="x_min" defaultValue="142" type="number" small/>
                <Field label="x_max" defaultValue="1318" type="number" small/>
                <Field label="y_min" defaultValue="98"  type="number" small/>
                <Field label="y_max" defaultValue="822" type="number" small/>
              </div>
              <div style={{
                marginTop:14, padding:'10px 12px', background:'#FBFBFB',
                border:'1px solid var(--border)', borderRadius:4,
                fontSize:11, color:'var(--text-secondary)', lineHeight:1.5
              }}>
                Arena spans 1176 × 724 px at 30 fps. Tracks outside these bounds are flagged as noise during
                Pose Estimation (stage 1).
              </div>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <div className="settings-grid">
            <div>
              <h3>Analysis parameters</h3>
              <p className="sec-desc">Defaults shared across all pipeline stages.</p>
            </div>
            <div>
              <Field label="Frame rate (fps)"      defaultValue="30"  type="number"/>
              <Field label="Min cluster size"      defaultValue="80"  type="number"/>
              <Field label="Confidence cutoff"     defaultValue="0.6" type="number"/>
              <Field label="Smoothing window (fr)" defaultValue="15"  type="number"/>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <div className="settings-grid">
            <div>
              <h3>External data</h3>
              <p className="sec-desc">Where VIEB reads cohort assignments and molecular protein measurements from.</p>
            </div>
            <div>
              <Field label="Cohort file" defaultValue="/home/lab/vieb/projects/FearCond_2026_Q2/cohorts.csv"/>
              <Field label="Jess file"   defaultValue="/home/lab/vieb/projects/FearCond_2026_Q2/jess_2026_05.xlsx"/>
            </div>
          </div>
        </div>
      </div>

      <div style={{
        marginTop: 16,
        padding: '12px 16px',
        background: 'rgba(39,174,96,0.06)',
        borderLeft: '3px solid var(--success)',
        borderRadius: 4,
        fontSize: 12,
        color: 'var(--text-secondary)',
        fontFamily: "'IBM Plex Mono', monospace"
      }}>
        ✓ Saved 14s ago · changes auto-save on field blur
      </div>
    </div>
  );
}

function Field({ label, defaultValue, type='text', small }) {
  return (
    <div className="field">
      <span className="field-label">
        {label}
        <span className="help-icon" title={`Help for ${label}`}>?</span>
      </span>
      <input
        className="input"
        type={type}
        defaultValue={defaultValue}
        style={{width: small ? 120 : '100%', fontFamily: type === 'number' ? "'IBM Plex Mono', monospace" : 'inherit'}}
      />
    </div>
  );
}

Object.assign(window, {
  QuantificationView, AdvancedView, SettingsView
});
