// Mock data for VIEB prototype
window.VIEB_DATA = (() => {

  // Cohort colors
  const COHORTS = [
    { id: 'C1', name: 'WT–Saline', color: '#4E79A7', n: 6 },
    { id: 'C2', name: 'WT–CNO',    color: '#E07B39', n: 6 },
    { id: 'C3', name: 'KO–Saline', color: '#59A14F', n: 5 },
    { id: 'C4', name: 'KO–CNO',    color: '#B07AA1', n: 5 }
  ];

  // Discrimination ratio per day per cohort (12 days, range -1..1)
  // Each cohort learns at slightly different rate
  function smoothCurve(rng, slope, plateau, days = 12) {
    const out = [];
    for (let d = 0; d < days; d++) {
      const x = (d / (days - 1)) * 8 - 4;
      const sig = plateau / (1 + Math.exp(-slope * x));
      out.push(sig - plateau/2 + (rng() - 0.5) * 0.06);
    }
    return out;
  }
  function rng(seed) {
    return () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };
  }
  const learning = COHORTS.map((c, i) => ({
    cohort: c,
    mean: smoothCurve(rng(7 + i*13), 0.9 + i*0.1, 1.0),
    animals: Array.from({length: c.n}, (_, k) =>
      smoothCurve(rng(100 + i*30 + k*5), 0.8 + i*0.12 + (rng(k+i)()-0.5)*0.3, 0.95)
    )
  }));

  // State occupancy by cohort: 12 states (state 0 = dominant, excluded)
  const STATE_IDS = Array.from({length: 12}, (_, i) => i + 1);
  const stateOccupancy = STATE_IDS.map((s, i) => {
    const r = rng(s * 17);
    return {
      state: s,
      cohorts: COHORTS.map((c, ci) => {
        const base = 0.04 + r() * 0.12;
        const cohortMod = (ci - 1.5) * 0.012 * (i % 3 - 1);
        return { mean: Math.max(0.005, base + cohortMod), se: 0.008 + r() * 0.012, cohort: c };
      })
    };
  });

  // Fear-enriched states: sorted by A - B
  const fearStates = Array.from({length: 22}, (_, i) => {
    const r = rng(i * 23 + 5);
    const diff = (Math.random ? 0 : 0) + (r() * 2 - 1) * 0.16;
    return { state: i + 5, diff: diff - 0.005 * (i - 11) };
  }).sort((a,b) => b.diff - a.diff);

  // States catalog for Browse States
  const labels = [
    'locomotion-like', 'freezing', 'grooming-front', 'grooming-rear',
    'rearing-supported', 'rearing-unsupported', 'sniffing-low', 'sniffing-high',
    'turning-left', 'turning-right', 'darting', 'startle-recoil',
    'still-alert', 'still-relaxed', 'wall-following', 'corner-tuck',
    'head-scan', 'stretch-attend', 'jumping', 'paw-flick',
    'tail-rattle', 'circling', 'climbing', 'investigating-novel',
    'micro-movement', 'transition'
  ];
  const ctxs = ['A','A','B','A','C','B','A','B','C','A','A','B','A','B','C','A','A','B','A','B','C','A','A','B'];
  const stateCatalog = Array.from({length: labels.length}, (_, i) => {
    const r = rng(i * 7 + 1);
    return {
      id: i + 1,
      label: labels[i],
      speed: 0.05 + r() * 0.9,
      duration: 8 + r() * 80,
      occupancy: 0.005 + r() * 0.08,
      animals: 4 + Math.floor(r() * 18),
      sessions: 20 + Math.floor(r() * 900),
      ctx: ctxs[i % ctxs.length],
      kinematics: {
        speed:    0.1 + r() * 0.9,
        accel:    0.1 + r() * 0.9,
        bodyArea: 0.1 + r() * 0.9,
        wallDist: 0.1 + r() * 0.9
      }
    };
  });

  // Pipeline stages
  const pipeline = [
    { num: 0, name: 'DLC Setup',            desc: 'Initialize DeepLabCut project and load trained model.', status: 'complete', ts: '14:02:11 · today' },
    { num: 1, name: 'Pose Estimation',      desc: 'Track 12 body keypoints per frame across all videos.', status: 'complete', ts: '14:18:44 · today' },
    { num: 2, name: 'Kinematics',           desc: 'Compute speed, acceleration, body angle and area.', status: 'complete', ts: '14:21:02 · today' },
    { num: 3, name: 'Feature Engineering',  desc: 'Roll, smooth, and derive behavioral features.', status: 'complete', ts: '14:23:30 · today' },
    { num: 4, name: 'State Discovery',      desc: 'Cluster behavioral features into discrete states.', status: 'complete', ts: '15:01:18 · today' },
    { num: 5, name: 'State Labeling',       desc: 'Heuristic labels assigned to discovered states.', status: 'complete', ts: '15:03:55 · today' },
    { num: 6, name: 'Bout Detection',       desc: 'Group consecutive frames into behavioral bouts.', status: 'complete', ts: '15:05:21 · today' },
    { num: 7, name: 'Cohort Aggregation',   desc: 'Compute per-cohort state occupancy and statistics.', status: 'running', ts: 'started 16:11:08' },
    { num: 8, name: 'Discrimination',       desc: 'Compute fear discrimination ratios per day.', status: 'not-run', ts: '—' },
    { num: 9, name: 'Quantification',       desc: 'Build the master table of behavioral variables.', status: 'not-run', ts: '—' },
    { num:10, name: 'Jess Correlation',     desc: 'Correlate behavior with protein measurements.', status: 'not-run', ts: '—' }
  ];

  // Master table rows
  const masterRows = Array.from({length: 14}, (_, i) => {
    const c = COHORTS[i % COHORTS.length];
    const r = rng(i * 11 + 3);
    return {
      animal: `M-${(2400 + i * 7).toString().padStart(4,'0')}`,
      cohort: c,
      sex: i % 2 ? 'F' : 'M',
      day: 1 + (i % 12),
      discr: (r() * 1.4 - 0.4).toFixed(3),
      thr: (r()).toFixed(3),
      asymp: (0.4 + r()*0.5).toFixed(3),
      freezing: (r() * 60).toFixed(1),
      bouts: Math.floor(8 + r() * 40),
      occ: (r() * 0.12).toFixed(4),
      trans: (40 + r() * 80).toFixed(1),
      dev: (r()*2 - 1).toFixed(3)
    };
  });

  // Top correlations
  const proteins = ['BDNF','TrkB','PSD-95','GluA1','GluA2','NR2B','PV','SOM','c-Fos','Arc','GAD67','VGAT','CREB','pCREB'];
  const behVars = ['Disc. ratio (d10)','Freezing % (Ctx A)','Bout dur. (S37)','Trans. entropy','Occ. fear-enriched','Disc. slope','Asymptote','Threshold day'];
  const correlations = [];
  proteins.forEach((p, pi) => {
    behVars.forEach((b, bi) => {
      const r = rng(pi * 23 + bi * 7);
      correlations.push({
        protein: p, behavior: b,
        r: (r() * 1.4 - 0.7),
        p: r() * r() * 0.2
      });
    });
  });
  correlations.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));

  // Fingerprint heatmap (animals × states)
  const fingerprint = Array.from({length: 18}, (_, ai) =>
    Array.from({length: 22}, (_, si) => {
      const r = rng(ai * 31 + si * 13);
      return r();
    })
  );

  return {
    COHORTS,
    learning,
    stateOccupancy,
    fearStates,
    stateCatalog,
    pipeline,
    masterRows,
    correlations,
    proteins,
    behVars,
    fingerprint,
    summary: {
      videos: 222,
      frames: '4.31M',
      states: 61,
      noise: 57.8
    }
  };
})();
