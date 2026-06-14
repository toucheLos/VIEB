// Chart components — hand-rolled SVG, Nature Methods aesthetic
// Common helpers
const CHART_COLORS = {
  c1: '#4E79A7', c2: '#E07B39', c3: '#59A14F', c4: '#B07AA1', c5: '#76B7B2', c6: '#EDC948',
  fear: '#C0392B', safe: '#2980B9',
  text: '#1A1A1A', sec: '#6B6B6B', ter: '#9B9B9B',
  grid: '#ECECEC', border: '#E5E5E5'
};

function ChartFrame({ width, height, padding, children }) {
  const p = padding || { t: 14, r: 14, b: 32, l: 44 };
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{display:'block', overflow:'visible'}}>
      {children(p)}
    </svg>
  );
}

function YAxis({ scale, ticks, x0, fmt = v => v }) {
  return (
    <g>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={x0} x2={x0 + scale.width} y1={scale.y(t)} y2={scale.y(t)} stroke={CHART_COLORS.grid} strokeWidth="1"/>
          <text x={x0 - 8} y={scale.y(t)} dy="0.32em" textAnchor="end"
                fontSize="10" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter}>
            {fmt(t)}
          </text>
        </g>
      ))}
      <line x1={x0} x2={x0} y1={scale.y(scale.min)} y2={scale.y(scale.max)} stroke={CHART_COLORS.border}/>
    </g>
  );
}

function XLabels({ labels, scale, y, rotate = 0 }) {
  return (
    <g>
      {labels.map((lbl, i) => (
        <text key={i} x={scale.x(i)} y={y}
              fontSize="10" fontFamily="'IBM Plex Mono', monospace"
              fill={CHART_COLORS.ter} textAnchor={rotate ? 'end' : 'middle'}
              transform={rotate ? `rotate(${rotate}, ${scale.x(i)}, ${y})` : ''}>
          {lbl}
        </text>
      ))}
    </g>
  );
}

// Learning curves: discrimination ratio per day, per cohort
function LearningCurves({ data, showIndividual }) {
  const W = 760, H = 320;
  const p = { t: 16, r: 24, b: 36, l: 48 };
  const innerW = W - p.l - p.r;
  const innerH = H - p.t - p.b;
  const days = data[0].mean.length;
  const xs = i => p.l + (i / (days - 1)) * innerW;
  const ys = v => p.t + (1 - (v + 1) / 2) * innerH;
  const yTicks = [-1, -0.5, 0, 0.5, 1];
  const xLabels = Array.from({length: days}, (_, i) => `D${i+1}`);

  const linePath = arr => arr.map((v, i) => `${i === 0 ? 'M' : 'L'}${xs(i)},${ys(v)}`).join(' ');

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block', overflow:'visible'}}>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={p.l} x2={W-p.r} y1={ys(t)} y2={ys(t)} stroke={t === 0 ? '#bbb' : CHART_COLORS.grid} strokeDasharray={t===0?'3,3':''}/>
          <text x={p.l - 8} y={ys(t)} dy="0.32em" textAnchor="end"
                fontSize="10" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter}>{t.toFixed(1)}</text>
        </g>
      ))}
      <text x={W - p.r - 4} y={ys(0) - 4} fontSize="10" fontStyle="italic" fill={CHART_COLORS.ter} textAnchor="end">chance</text>

      <line x1={p.l} x2={p.l} y1={p.t} y2={H-p.b} stroke={CHART_COLORS.border}/>
      <line x1={p.l} x2={W-p.r} y1={H-p.b} y2={H-p.b} stroke={CHART_COLORS.border}/>

      {xLabels.map((lbl, i) => (
        <text key={i} x={xs(i)} y={H - p.b + 14} fontSize="10"
              fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter} textAnchor="middle">{lbl}</text>
      ))}

      {showIndividual && data.flatMap((d, di) =>
        d.animals.map((a, ai) => (
          <path key={`${di}-${ai}`} d={linePath(a)} fill="none" stroke={d.cohort.color} strokeOpacity="0.18" strokeWidth="1"/>
        ))
      )}

      {data.map((d, i) => (
        <g key={i}>
          <path d={linePath(d.mean)} fill="none" stroke={d.cohort.color} strokeWidth="2"/>
          {d.mean.map((v, j) => (
            <circle key={j} cx={xs(j)} cy={ys(v)} r="2.5" fill={d.cohort.color}/>
          ))}
        </g>
      ))}

      <text x={p.l - 36} y={p.t + innerH/2} fontSize="10" fill={CHART_COLORS.sec}
            transform={`rotate(-90, ${p.l - 36}, ${p.t + innerH/2})`} textAnchor="middle">
        discrimination ratio
      </text>
      <text x={p.l + innerW/2} y={H - 4} fontSize="10" fill={CHART_COLORS.sec} textAnchor="middle">training day</text>
    </svg>
  );
}

// Grouped bar: state occupancy by cohort
function GroupedBars({ data, cohorts }) {
  const W = 560, H = 280;
  const p = { t: 14, r: 12, b: 40, l: 46 };
  const innerW = W - p.l - p.r;
  const innerH = H - p.t - p.b;
  const nStates = data.length;
  const groupW = innerW / nStates;
  const barW = groupW * 0.7 / cohorts.length;
  const maxY = 0.22;
  const ys = v => p.t + (1 - v / maxY) * innerH;
  const yTicks = [0, 0.05, 0.1, 0.15, 0.2];

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block'}}>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={p.l} x2={W-p.r} y1={ys(t)} y2={ys(t)} stroke={CHART_COLORS.grid}/>
          <text x={p.l-8} y={ys(t)} dy="0.32em" textAnchor="end"
                fontSize="10" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter}>{(t*100).toFixed(0)}%</text>
        </g>
      ))}
      <line x1={p.l} x2={p.l} y1={p.t} y2={H-p.b} stroke={CHART_COLORS.border}/>
      <line x1={p.l} x2={W-p.r} y1={H-p.b} y2={H-p.b} stroke={CHART_COLORS.border}/>

      {data.map((s, gi) => {
        const gx0 = p.l + gi * groupW + groupW * 0.15;
        return (
          <g key={gi}>
            {s.cohorts.map((c, ci) => {
              const x = gx0 + ci * barW;
              const y = ys(c.mean);
              const h = (H - p.b) - y;
              return (
                <g key={ci}>
                  <rect x={x} y={y} width={barW - 1} height={h} fill={c.cohort.color}/>
                  <line
                    x1={x + barW/2} x2={x + barW/2}
                    y1={ys(c.mean + c.se)} y2={ys(Math.max(0, c.mean - c.se))}
                    stroke="#333" strokeWidth="1"/>
                  <line x1={x + barW/2 - 2} x2={x + barW/2 + 2} y1={ys(c.mean+c.se)} y2={ys(c.mean+c.se)} stroke="#333" strokeWidth="1"/>
                  <line x1={x + barW/2 - 2} x2={x + barW/2 + 2} y1={ys(Math.max(0,c.mean-c.se))} y2={ys(Math.max(0,c.mean-c.se))} stroke="#333" strokeWidth="1"/>
                </g>
              );
            })}
            <text x={p.l + gi * groupW + groupW/2} y={H - p.b + 14}
                  fontSize="10" fontFamily="'IBM Plex Mono', monospace"
                  fill={CHART_COLORS.ter} textAnchor="middle">S{s.state}</text>
          </g>
        );
      })}
      <text x={p.l - 36} y={p.t + innerH/2} fontSize="10" fill={CHART_COLORS.sec}
            transform={`rotate(-90, ${p.l - 36}, ${p.t + innerH/2})`} textAnchor="middle">
        occupancy
      </text>
      <text x={p.l + innerW/2} y={H - 4} fontSize="10" fill={CHART_COLORS.sec} textAnchor="middle">state ID</text>
    </svg>
  );
}

// Diverging bars: fear-enriched
function DivergingBars({ data }) {
  const W = 560, H = 280;
  const p = { t: 14, r: 24, b: 36, l: 46 };
  const innerW = W - p.l - p.r;
  const innerH = H - p.t - p.b;
  const max = Math.max(...data.map(d => Math.abs(d.diff))) * 1.05;
  const barW = innerW / data.length;
  const ys = v => p.t + innerH/2 - (v / max) * (innerH/2);
  const yTicks = [-max*0.8, -max*0.4, 0, max*0.4, max*0.8];

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block'}}>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={p.l} x2={W-p.r} y1={ys(t)} y2={ys(t)} stroke={CHART_COLORS.grid}/>
          <text x={p.l-8} y={ys(t)} dy="0.32em" textAnchor="end"
                fontSize="10" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter}>{(t*100).toFixed(0)}</text>
        </g>
      ))}
      <line x1={p.l} x2={W-p.r} y1={ys(0)} y2={ys(0)} stroke="#888" strokeWidth="1"/>
      <line x1={p.l} x2={p.l} y1={p.t} y2={H-p.b} stroke={CHART_COLORS.border}/>

      {data.map((d, i) => {
        const x = p.l + i * barW + barW * 0.18;
        const w = barW * 0.64;
        const color = d.diff > 0 ? CHART_COLORS.fear : CHART_COLORS.safe;
        const y0 = ys(0);
        const y1 = ys(d.diff);
        return (
          <rect key={i} x={x} y={Math.min(y0,y1)} width={w} height={Math.abs(y1-y0)} fill={color}/>
        );
      })}
      <text x={W - p.r - 6} y={ys(0) - 6} fontSize="10" fill={CHART_COLORS.fear} textAnchor="end">↑ Context A (fear)</text>
      <text x={W - p.r - 6} y={ys(0) + 14} fontSize="10" fill={CHART_COLORS.safe} textAnchor="end">↓ Context B (safe)</text>
      <text x={p.l - 36} y={p.t + innerH/2} fontSize="10" fill={CHART_COLORS.sec}
            transform={`rotate(-90, ${p.l - 36}, ${p.t + innerH/2})`} textAnchor="middle">
        Δ occupancy (pp)
      </text>
      <text x={p.l + innerW/2} y={H - 4} fontSize="10" fill={CHART_COLORS.sec} textAnchor="middle">state (sorted by Δ)</text>
    </svg>
  );
}

// Horizontal kinematic bars
function KinematicBars({ kin }) {
  const items = [
    { key: 'speed', label: 'Speed (cm/s)' },
    { key: 'accel', label: 'Acceleration' },
    { key: 'bodyArea', label: 'Body area' },
    { key: 'wallDist', label: 'Wall dist.' }
  ];
  return (
    <div className="kinematic-bars">
      {items.map(it => (
        <React.Fragment key={it.key}>
          <div className="kbar-label">{it.label}</div>
          <div className="kbar-track">
            <div className="kbar-fill" style={{width: `${kin[it.key] * 100}%`}}></div>
          </div>
          <div className="kbar-value">{(kin[it.key] * 100).toFixed(1)}</div>
        </React.Fragment>
      ))}
    </div>
  );
}

// Simple keypoint overlay for validation
function KeypointOverlay() {
  // 12 keypoints around an oval body
  const pts = [
    [50, 40, 'nose'], [50, 47, 'head'], [44, 47, 'L ear'], [56, 47, 'R ear'],
    [50, 56, 'neck'], [44, 60, 'L shoulder'], [56, 60, 'R shoulder'],
    [50, 70, 'mid'], [44, 80, 'L hip'], [56, 80, 'R hip'],
    [50, 86, 'tail base'], [50, 95, 'tail tip']
  ];
  const links = [
    [0,1],[1,2],[1,3],[1,4],[4,5],[4,6],[4,7],[7,8],[7,9],[7,10],[10,11]
  ];
  const colors = ['#E07B39','#E07B39','#EDC948','#EDC948','#4E79A7','#59A14F','#59A14F','#4E79A7','#B07AA1','#B07AA1','#76B7B2','#C0392B'];

  return (
    <svg viewBox="0 0 100 110" style={{position:'absolute',inset:0,width:'100%',height:'100%'}}>
      {links.map(([a,b], i) => (
        <line key={i} x1={pts[a][0]} y1={pts[a][1]} x2={pts[b][0]} y2={pts[b][1]} stroke="#ffffff" strokeOpacity="0.55" strokeWidth="0.4"/>
      ))}
      {pts.map((pt, i) => (
        <circle key={i} cx={pt[0]} cy={pt[1]} r="1.1" fill={colors[i]} stroke="#fff" strokeWidth="0.25"/>
      ))}
    </svg>
  );
}

// Confusion matrix mini
function ConfusionMatrix() {
  const labels = ['Freeze','Walk','Groom','Rear','Other'];
  const mat = [
    [42,  1,  0,  0,  1],
    [ 2, 38,  1,  1,  0],
    [ 0,  1, 22,  0,  1],
    [ 0,  1,  0, 18,  1],
    [ 1,  1,  2,  1, 14]
  ];
  const max = 42;
  return (
    <div style={{display:'inline-block'}}>
      <table style={{borderCollapse:'collapse', fontFamily:"'IBM Plex Mono', monospace", fontSize:10}}>
        <thead>
          <tr>
            <td></td>
            {labels.map(l => <th key={l} style={{padding:'4px 6px', fontWeight:500, color:'#9b9b9b'}}>{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {mat.map((row, ri) => (
            <tr key={ri}>
              <th style={{padding:'4px 6px', fontWeight:500, color:'#9b9b9b', textAlign:'right'}}>{labels[ri]}</th>
              {row.map((v, ci) => {
                const t = v/max;
                const bg = ri === ci ? `rgba(78,121,167,${0.15 + 0.7*t})` : `rgba(192,57,43,${0.05 + 0.4*t})`;
                return (
                  <td key={ci} style={{
                    width: 36, height: 28, textAlign: 'center',
                    background: bg, color: t > 0.4 ? 'white' : '#1a1a1a',
                    border: '1px solid #fff'
                  }}>{v}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Correlation heatmap
function CorrHeatmap({ corrs, proteins, behVars }) {
  const cellW = 36, cellH = 22;
  const labelW = 150, labelH = 60;
  const W = labelW + proteins.length * cellW + 8;
  const H = labelH + behVars.length * cellH + 8;
  const get = (b, p) => corrs.find(c => c.behavior === b && c.protein === p);
  function fill(r) {
    const v = Math.max(-1, Math.min(1, r));
    if (v >= 0) {
      const a = v;
      return `rgb(${Math.round(255 - 60*a)}, ${Math.round(245 - 180*a)}, ${Math.round(238 - 195*a)})`;
    } else {
      const a = -v;
      return `rgb(${Math.round(255 - 215*a)}, ${Math.round(245 - 130*a)}, ${Math.round(238 - 50*a)})`;
    }
  }
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block', maxWidth: W}}>
      {proteins.map((p, pi) => (
        <text key={p}
              x={labelW + pi*cellW + cellW/2}
              y={labelH - 6}
              fontSize="10"
              fontFamily="'IBM Plex Mono', monospace"
              fill={CHART_COLORS.sec}
              textAnchor="end"
              transform={`rotate(-50, ${labelW + pi*cellW + cellW/2}, ${labelH - 6})`}>
          {p}
        </text>
      ))}
      {behVars.map((b, bi) => (
        <g key={b}>
          <text x={labelW - 6} y={labelH + bi*cellH + cellH/2} dy="0.32em"
                fontSize="10" fill={CHART_COLORS.sec} textAnchor="end">{b}</text>
          {proteins.map((p, pi) => {
            const c = get(b, p);
            const r = c ? c.r : 0;
            const sig = c && c.p < 0.05;
            return (
              <g key={p}>
                <rect x={labelW + pi*cellW + 1} y={labelH + bi*cellH + 1}
                      width={cellW - 2} height={cellH - 2}
                      fill={fill(r)} stroke="#fff" strokeWidth="0.5"/>
                {sig && (
                  <text x={labelW + pi*cellW + cellW/2}
                        y={labelH + bi*cellH + cellH/2 + 3}
                        fontSize="10" textAnchor="middle"
                        fill="#1a1a1a" fontWeight="700">*</text>
                )}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

// Fingerprint heatmap (animal × state)
function Fingerprint({ data }) {
  const cellW = 22, cellH = 16;
  const labelW = 70, topH = 22;
  const W = labelW + data[0].length * cellW + 8;
  const H = topH + data.length * cellH + 8;
  function fill(v) {
    const x = Math.max(0, Math.min(1, v));
    // perceptual single-hue scale (warm)
    return `rgb(${Math.round(255 - 60*x)}, ${Math.round(245 - 170*x)}, ${Math.round(230 - 200*x)})`;
  }
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block', maxWidth: W}}>
      {data[0].map((_, si) => (
        <text key={si} x={labelW + si*cellW + cellW/2} y={topH - 6}
              fontSize="9" fontFamily="'IBM Plex Mono', monospace"
              fill={CHART_COLORS.ter} textAnchor="middle">
          {si+5}
        </text>
      ))}
      {data.map((row, ai) => (
        <g key={ai}>
          <text x={labelW - 6} y={topH + ai*cellH + cellH/2} dy="0.32em"
                fontSize="10" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.sec} textAnchor="end">
            M-{(2400 + ai*7).toString().padStart(4,'0')}
          </text>
          {row.map((v, si) => (
            <rect key={si} x={labelW + si*cellW + 1} y={topH + ai*cellH + 1}
                  width={cellW - 2} height={cellH - 2}
                  fill={fill(v)}/>
          ))}
        </g>
      ))}
    </svg>
  );
}

// Transition matrix
function TransitionMatrix() {
  const N = 16;
  const states = Array.from({length: N}, (_, i) => i + 1);
  const cellW = 22, cellH = 18;
  const labelW = 30, topH = 22;
  const data = [];
  let seed = 17;
  for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < N; j++) {
      seed = (seed * 9301 + 49297) % 233280;
      const v = i === j ? 0.6 + (seed/233280)*0.35 : (seed/233280) * 0.15;
      row.push(v);
    }
    data.push(row);
  }
  function fill(v) {
    const x = Math.max(0, Math.min(1, v));
    return `rgb(${Math.round(255 - 100*x)}, ${Math.round(248 - 150*x)}, ${Math.round(244 - 60*x)})`;
  }
  const W = labelW + N*cellW + 8;
  const H = topH + N*cellH + 24;
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block', maxWidth: W}}>
      {states.map((s, i) => (
        <text key={`top-${i}`} x={labelW + i*cellW + cellW/2} y={topH - 6}
              fontSize="9" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter} textAnchor="middle">{s}</text>
      ))}
      {data.map((row, ri) => (
        <g key={ri}>
          <text x={labelW - 4} y={topH + ri*cellH + cellH/2} dy="0.32em"
                fontSize="9" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter} textAnchor="end">{states[ri]}</text>
          {row.map((v, ci) => (
            <rect key={ci} x={labelW + ci*cellW + 0.5} y={topH + ri*cellH + 0.5}
                  width={cellW - 1} height={cellH - 1}
                  fill={fill(v)} stroke="#fff" strokeWidth="0.5"/>
          ))}
        </g>
      ))}
      <text x={labelW + (N*cellW)/2} y={H - 4} fontSize="10" fill={CHART_COLORS.sec} textAnchor="middle">→ to state</text>
    </svg>
  );
}

// Arena diagram for settings
function ArenaDiagram() {
  return (
    <svg viewBox="0 0 220 140" width="220" height="140" style={{display:'block'}}>
      <rect x="40" y="20" width="140" height="100" fill="none" stroke="#1a1a1a" strokeWidth="1"/>
      <line x1="40" y1="135" x2="180" y2="135" stroke="#9b9b9b" strokeWidth="0.5"/>
      <line x1="40" y1="20" x2="40" y2="120" stroke="#4E79A7" strokeWidth="1" strokeDasharray="3,2"/>
      <line x1="180" y1="20" x2="180" y2="120" stroke="#4E79A7" strokeWidth="1" strokeDasharray="3,2"/>
      <line x1="40" y1="20" x2="180" y2="20" stroke="#E07B39" strokeWidth="1" strokeDasharray="3,2"/>
      <line x1="40" y1="120" x2="180" y2="120" stroke="#E07B39" strokeWidth="1" strokeDasharray="3,2"/>
      <text x="34" y="22" fontSize="8" fontFamily="'IBM Plex Mono', monospace" fill="#E07B39" textAnchor="end">y_min</text>
      <text x="34" y="122" fontSize="8" fontFamily="'IBM Plex Mono', monospace" fill="#E07B39" textAnchor="end">y_max</text>
      <text x="40" y="14" fontSize="8" fontFamily="'IBM Plex Mono', monospace" fill="#4E79A7" textAnchor="middle">x_min</text>
      <text x="180" y="14" fontSize="8" fontFamily="'IBM Plex Mono', monospace" fill="#4E79A7" textAnchor="middle">x_max</text>
      <circle cx="110" cy="70" r="4" fill="#1a1a1a"/>
      <text x="118" y="73" fontSize="9" fontFamily="'IBM Plex Mono', monospace" fill="#6b6b6b">animal</text>
    </svg>
  );
}

// Sigmoid fit overlay
function LearningCurvesSigmoid({ data }) {
  const W = 760, H = 360;
  const p = { t: 14, r: 24, b: 36, l: 48 };
  const innerW = W - p.l - p.r;
  const innerH = H - p.t - p.b;
  const days = data[0].mean.length;
  const xs = i => p.l + (i / (days - 1)) * innerW;
  const ys = v => p.t + (1 - (v + 1) / 2) * innerH;
  const yTicks = [-1, -0.5, 0, 0.5, 1];

  function sigFit(arr) {
    // approximation: smoothed curve
    return arr.map((v, i, a) => {
      const w = a.slice(Math.max(0, i-2), Math.min(a.length, i+3));
      return w.reduce((s,x) => s+x, 0) / w.length;
    });
  }

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block'}}>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={p.l} x2={W-p.r} y1={ys(t)} y2={ys(t)} stroke={t === 0 ? '#bbb' : CHART_COLORS.grid} strokeDasharray={t===0?'3,3':''}/>
          <text x={p.l - 8} y={ys(t)} dy="0.32em" textAnchor="end"
                fontSize="10" fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter}>{t.toFixed(1)}</text>
        </g>
      ))}
      {Array.from({length: days}, (_, i) => i).map(i => (
        <text key={i} x={xs(i)} y={H - p.b + 14} fontSize="10"
              fontFamily="'IBM Plex Mono', monospace" fill={CHART_COLORS.ter} textAnchor="middle">D{i+1}</text>
      ))}
      <line x1={p.l} x2={p.l} y1={p.t} y2={H-p.b} stroke={CHART_COLORS.border}/>
      <line x1={p.l} x2={W-p.r} y1={H-p.b} y2={H-p.b} stroke={CHART_COLORS.border}/>

      {data.flatMap((d, di) => d.animals.map((a, ai) => (
        <path key={`a-${di}-${ai}`}
              d={a.map((v, i) => `${i===0?'M':'L'}${xs(i)},${ys(v)}`).join(' ')}
              fill="none" stroke={d.cohort.color} strokeOpacity="0.16" strokeWidth="1"/>
      )))}

      {data.flatMap((d, di) => d.animals.map((a, ai) => {
        const s = sigFit(a);
        return (
          <path key={`s-${di}-${ai}`}
                d={s.map((v, i) => `${i===0?'M':'L'}${xs(i)},${ys(v)}`).join(' ')}
                fill="none" stroke={d.cohort.color} strokeOpacity="0.55" strokeWidth="1" strokeDasharray="3,2"/>
        );
      }))}

      {data.map((d, di) => (
        <path key={di}
              d={d.mean.map((v, i) => `${i===0?'M':'L'}${xs(i)},${ys(v)}`).join(' ')}
              fill="none" stroke={d.cohort.color} strokeWidth="2"/>
      ))}
    </svg>
  );
}

Object.assign(window, {
  LearningCurves, GroupedBars, DivergingBars, KinematicBars,
  KeypointOverlay, ConfusionMatrix, CorrHeatmap, Fingerprint,
  TransitionMatrix, ArenaDiagram, LearningCurvesSigmoid,
  CHART_COLORS
});
