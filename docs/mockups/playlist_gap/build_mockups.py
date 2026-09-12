#!/usr/bin/env python3
"""Generate the Playlist Gap interface mockups.

Source of truth for docs/mockups/playlist_gap/*.html. Emits five standalone
mockups plus index.html, a switcher that holds all five for side-by-side
comparison. Run from anywhere:

    python docs/mockups/playlist_gap/build_mockups.py

Palette is the app's real "Midnight" theme from gui/themes/tokens.py, so the
mockups read as actual AlphaDEX workspaces rather than web pages.
"""
from __future__ import annotations

import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MOCKUPS = [
    ("01-flow-tiles",   "mk1", "Flow Tiles",      "Every step is a tile on one board, wired in order. You always see the whole pipeline and where you stopped."),
    ("02-tabbed-steps", "mk2", "Tabbed Steps",    "One step at a time behind a tab bar that tracks completion. Closest to the workspaces the app already has."),
    ("03-workbench",    "mk3", "Workbench",       "No steps. Setup rail, a three-column bucket board you drag between, evidence docked on the right."),
    ("04-triage-queue", "mk4", "Triage Queue",    "An inbox. List on the left, evidence in a reading pane, keyboard-driven. Built to clear the unsure pile fast."),
    ("05-answer-first", "mk5", "Answer First",    "Opens on the download list. The machinery is collapsed behind a summary strip you expand only if you doubt it."),
]

# ── shared chrome ────────────────────────────────────────────────────────────

BASE_CSS = """
  :root{
    --sidebar-bg:#0d1117; --sidebar-hover:#1c2128; --sidebar-active:#1c2a3a;
    --sidebar-accent:#58a6ff; --sidebar-text:#c9d1d9; --sidebar-hdr:#484f58;
    --content-bg:#161b22; --card-bg:#1c2128; --card-border:#30363d;
    --text:#e6edf3; --text-2:#8b949e; --text-3:#484f58; --text-inv:#0d1117;
    --accent:#58a6ff; --accent-hover:#79b8ff; --accent-pressed:#388bfd;
    --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444;
    --ok-bg:#10251a; --warn-bg:#2a1f0a; --bad-bg:#2a1212; --accent-bg:#10243d;
    --input-bg:#0d1117; --input-border:#30363d;
    --ui:"Inter","Segoe UI",system-ui,-apple-system,sans-serif;
    --mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{background:#0a0d12; color:var(--text); font-family:var(--ui); font-size:13px; line-height:1.45}

  /* app frame */
  .app{
    display:grid; grid-template-columns:186px 1fr;
    background:var(--content-bg); border:1px solid var(--card-border);
    border-radius:8px; overflow:hidden; min-height:660px;
  }
  .rail{background:var(--sidebar-bg); padding:12px 0; border-right:1px solid var(--card-border)}
  .rail .brand{
    font-weight:700; font-size:14px; letter-spacing:-.01em; color:var(--text);
    padding:4px 16px 14px; display:flex; align-items:center; gap:7px;
  }
  .rail .brand span{color:var(--accent)}
  .rail .hdr{
    font-size:9.5px; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
    color:var(--sidebar-hdr); padding:12px 16px 5px;
  }
  .rail a{
    display:flex; align-items:center; gap:9px; padding:6px 16px;
    color:var(--sidebar-text); text-decoration:none; font-size:12.5px;
    border-left:3px solid transparent;
  }
  .rail a .ic{width:15px; text-align:center; opacity:.75; font-size:12px}
  .rail a:hover{background:var(--sidebar-hover)}
  .rail a.on{background:var(--sidebar-active); border-left-color:var(--sidebar-accent); color:#fff; font-weight:600}
  .rail a.on .ic{opacity:1}
  .rail a.new::after{
    content:"NEW"; margin-left:auto; font-family:var(--mono); font-size:8px; font-weight:700;
    letter-spacing:.08em; color:var(--accent); background:var(--accent-bg);
    border:1px solid var(--accent-pressed); border-radius:3px; padding:1px 4px;
  }
  .main{display:flex; flex-direction:column; min-width:0; overflow:hidden}
  .titlebar{
    padding:14px 20px 13px; border-bottom:1px solid var(--card-border);
    display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; flex:none;
  }
  .titlebar h2{margin:0; font-size:16px; font-weight:700; letter-spacing:-.015em}
  .titlebar .crumb{font-size:11.5px; color:var(--text-2)}
  .titlebar .right{margin-left:auto; display:flex; gap:7px; align-items:center}
  .body{padding:18px 20px; overflow:auto; flex:1; min-width:0}
  .statusbar{
    flex:none; border-top:1px solid var(--card-border); padding:6px 20px;
    font-size:11px; color:var(--text-2); display:flex; gap:14px; align-items:center;
    background:var(--sidebar-bg);
  }
  .statusbar .sp{margin-left:auto}

  /* primitives */
  .card{background:var(--card-bg); border:1px solid var(--card-border); border-radius:6px}
  .btn{
    font-family:var(--ui); font-size:12px; font-weight:600; color:var(--text);
    background:#21262d; border:1px solid var(--card-border); border-radius:5px;
    padding:6px 12px; cursor:pointer;
  }
  .btn:hover{border-color:var(--accent); color:var(--accent-hover)}
  .btn.pri{background:var(--accent); border-color:var(--accent); color:var(--text-inv)}
  .btn.pri:hover{background:var(--accent-hover); color:var(--text-inv)}
  .btn.sm{padding:3px 8px; font-size:11px}
  .btn:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
  .btn[disabled]{opacity:.4; cursor:default}
  .inp{
    background:var(--input-bg); border:1px solid var(--input-border); border-radius:5px;
    padding:6px 9px; color:var(--text); font-size:12px; font-family:var(--ui); width:100%;
  }
  .inp.mono{font-family:var(--mono); font-size:11.5px}
  .pill{
    display:inline-block; font-family:var(--mono); font-size:9px; font-weight:700;
    letter-spacing:.07em; text-transform:uppercase; padding:2px 6px; border-radius:3px;
    white-space:nowrap;
  }
  .pill.ok{background:var(--ok-bg); color:var(--ok); border:1px solid #1d4a30}
  .pill.warn{background:var(--warn-bg); color:var(--warn); border:1px solid #5c4410}
  .pill.bad{background:var(--bad-bg); color:var(--bad); border:1px solid #5e2320}
  .pill.acc{background:var(--accent-bg); color:var(--accent); border:1px solid var(--accent-pressed)}
  .pill.mute{background:#21262d; color:var(--text-2); border:1px solid var(--card-border)}
  .lbl{
    font-size:9.5px; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:var(--text-3);
  }
  .mono{font-family:var(--mono)}
  .t2{color:var(--text-2)}
  .t3{color:var(--text-3)}
  .num{font-variant-numeric:tabular-nums}
  .track{font-family:var(--mono); font-size:11.5px; word-break:break-word}

  /* mockup page shell */
  .shell{max-width:1240px; margin:0 auto; padding:22px 18px 60px}
  .shell > header{margin-bottom:16px}
  .shell h1{
    margin:0 0 5px; font-size:19px; font-weight:700; letter-spacing:-.02em;
  }
  .shell h1 em{font-style:normal; font-family:var(--mono); font-size:12px; color:var(--accent); font-weight:600; margin-right:8px}
  .shell .thesis{margin:0; font-size:13px; color:var(--text-2); max-width:78ch}
  @media (max-width:820px){
    .app{grid-template-columns:1fr}
    .rail{display:none}
  }
"""

RAIL = """      <nav class="rail">
        <div class="brand">◈ Alpha<span>DEX</span></div>
        <div class="hdr">Organize</div>
        <a href="#"><span class="ic">⊞</span>Indexer</a>
        <a href="#"><span class="ic">⇄</span>Library Sync</a>
        <a href="#"><span class="ic">⧉</span>Duplicates</a>
        <a href="#" class="on new"><span class="ic">◎</span>Playlist Gap</a>
        <div class="hdr">Metadata</div>
        <a href="#"><span class="ic">✎</span>Tag Fixer</a>
        <a href="#"><span class="ic">♫</span>Genres</a>
        <div class="hdr">Discover</div>
        <a href="#"><span class="ic">≡</span>Playlists</a>
        <a href="#"><span class="ic">✳</span>Clustered</a>
        <a href="#"><span class="ic">◌</span>Graph</a>
        <div class="hdr">Utility</div>
        <a href="#"><span class="ic">▶</span>Player</a>
        <a href="#"><span class="ic">⚙</span>Tools</a>
      </nav>
"""

STATUS = """        <div class="statusbar">
          <span>Library snapshot: <b class="num">8,412</b> tracks · read 4 min ago</span>
          <span class="sp">Source: YouTube Music · ytmusicapi</span>
        </div>
"""

# ── 1 · Flow Tiles ───────────────────────────────────────────────────────────

MK1_CSS = """
  #mk1 .board{display:flex; flex-direction:column; gap:0}
  #mk1 .row{display:grid; grid-template-columns:1fr 26px 1fr 26px 1fr; gap:0; align-items:stretch}
  #mk1 .tile{
    background:var(--card-bg); border:1px solid var(--card-border); border-radius:6px;
    padding:13px 15px; display:flex; flex-direction:column; gap:8px; min-height:132px;
  }
  #mk1 .tile.done{border-color:#1d4a30}
  #mk1 .tile.active{border-color:var(--accent); box-shadow:0 0 0 1px var(--accent-pressed), 0 6px 20px -12px #000}
  #mk1 .tile.locked{opacity:.45}
  #mk1 .tile .top{display:flex; align-items:center; gap:8px}
  #mk1 .tile .n{
    font-family:var(--mono); font-size:10px; font-weight:700; color:var(--text-inv);
    background:var(--text-3); width:18px; height:18px; border-radius:50%;
    display:grid; place-items:center; flex:none;
  }
  #mk1 .tile.done .n{background:var(--ok)}
  #mk1 .tile.active .n{background:var(--accent)}
  #mk1 .tile h4{margin:0; font-size:12.5px; font-weight:700; letter-spacing:-.01em}
  #mk1 .tile .state{margin-left:auto}
  #mk1 .tile .meat{font-size:11.5px; color:var(--text-2); flex:1}
  #mk1 .tile .meat b{color:var(--text); font-weight:600}
  #mk1 .tile .act{display:flex; gap:6px; flex-wrap:wrap}
  #mk1 .arrow{display:grid; place-items:center; color:var(--text-3); font-size:15px}
  #mk1 .arrow.lit{color:var(--accent)}
  #mk1 .down{display:grid; grid-template-columns:1fr 26px 1fr 26px 1fr; height:30px}
  #mk1 .down .v{grid-column:5; display:grid; place-items:center; color:var(--accent); font-size:15px}
  #mk1 .down.rev .v{grid-column:1}
  #mk1 .buckets{display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:2px}
  #mk1 .bk{border:1px solid var(--card-border); border-radius:5px; padding:8px 10px; background:var(--input-bg)}
  #mk1 .bk .v{font-size:19px; font-weight:700; letter-spacing:-.02em}
  #mk1 .bk.g{border-color:#1d4a30} #mk1 .bk.g .v{color:var(--ok)}
  #mk1 .bk.a{border-color:#5c4410} #mk1 .bk.a .v{color:var(--warn)}
  #mk1 .bk.r{border-color:#5e2320} #mk1 .bk.r .v{color:var(--bad)}
  @media (max-width:900px){
    #mk1 .row,#mk1 .down{grid-template-columns:1fr}
    #mk1 .row{gap:8px; margin-bottom:8px}
    #mk1 .arrow{display:none}
    #mk1 .down{height:auto}
    #mk1 .down .v{grid-column:1 !important}
  }
"""

MK1 = """
      <div class="titlebar">
        <h2>Playlist Gap</h2>
        <span class="crumb">Liked videos · YouTube Music · step 4 of 6</span>
        <div class="right">
          <button class="btn sm">Saved lists (4)</button>
          <button class="btn sm">Reset</button>
        </div>
      </div>
      <div class="body">
        <div class="board">
          <div class="row">
            <div class="tile done">
              <div class="top"><span class="n">1</span><h4>Source</h4><span class="state pill ok">done</span></div>
              <div class="meat">YouTube Music — <b>Liked videos</b><br><b class="num">1,204</b> wanted tracks<br><span class="t3">1 unavailable (deleted video)</span></div>
              <div class="act"><button class="btn sm">Change source</button></div>
            </div>
            <div class="arrow lit">→</div>
            <div class="tile done">
              <div class="top"><span class="n">2</span><h4>Library snapshot</h4><span class="state pill ok">done</span></div>
              <div class="meat"><b class="num">8,412</b> tracks indexed<br><span class="t3">incl. Not Sorted, Quarantine, Manual Review</span></div>
              <div class="act"><button class="btn sm">Refresh</button><button class="btn sm">Folders…</button></div>
            </div>
            <div class="arrow lit">→</div>
            <div class="tile done">
              <div class="top"><span class="n">3</span><h4>Read &amp; split</h4><span class="state pill ok">done</span></div>
              <div class="meat"><b class="num">1,189</b> parsed cleanly<br><b class="num">15</b> needed candidate splits<br><span class="t3">e.g. “Self Aware x Babydoll”</span></div>
              <div class="act"><button class="btn sm">Inspect splits</button></div>
            </div>
          </div>

          <div class="down rev"><div class="v">↓</div></div>

          <div class="row">
            <div class="tile active">
              <div class="top"><span class="n">4</span><h4>Compare</h4><span class="state pill acc">ready</span></div>
              <div class="meat">Ladder rungs available for this source: <b>0, 2, 4, 5, 5b</b><br><span class="t3">video ID · duration · artist all present</span></div>
              <div class="act"><button class="btn pri">Compare now</button></div>
            </div>
            <div class="arrow lit">→</div>
            <div class="tile">
              <div class="top"><span class="n">5</span><h4>Triage</h4><span class="state pill warn">68 to review</span></div>
              <div class="buckets">
                <div class="bk g"><div class="lbl">Owned</div><div class="v num">1,031</div></div>
                <div class="bk a"><div class="lbl">Unsure</div><div class="v num">68</div></div>
                <div class="bk r"><div class="lbl">Missing</div><div class="v num">105</div></div>
              </div>
              <div class="act"><button class="btn pri">Review unsure</button><button class="btn sm">Browse all</button></div>
            </div>
            <div class="arrow">→</div>
            <div class="tile locked">
              <div class="top"><span class="n">6</span><h4>Download list</h4><span class="state pill mute">waiting</span></div>
              <div class="meat">Unlocks once the unsure pile is cleared — or export the <b class="num">105</b> certain ones now.</div>
              <div class="act"><button class="btn sm">Export 105 anyway</button></div>
            </div>
          </div>

          <div class="down"><div class="v">↓</div></div>

          <div class="row">
            <div class="tile" style="grid-column:1/-1; min-height:0">
              <div class="top"><span class="n">✓</span><h4>After you download</h4><span class="state pill mute">hand-off</span></div>
              <div class="meat"><b class="num">42</b> tracks exported 6 days ago and still marked pending. Verify them against the library with real fingerprints.</div>
              <div class="act"><button class="btn sm">Open Library Sync →</button><button class="btn sm">Mark 42 as downloaded</button></div>
            </div>
          </div>
        </div>
      </div>
"""

# ── 2 · Tabbed Steps ─────────────────────────────────────────────────────────

MK2_CSS = """
  #mk2 .tabs{display:flex; gap:2px; border-bottom:1px solid var(--card-border); padding:0 20px; flex:none; background:var(--sidebar-bg)}
  #mk2 .tab{
    display:flex; align-items:center; gap:7px; padding:9px 15px 8px;
    font-size:12px; font-weight:600; color:var(--text-2);
    border:1px solid transparent; border-bottom:none; border-radius:5px 5px 0 0;
    cursor:pointer; background:none; font-family:var(--ui); position:relative; top:1px;
  }
  #mk2 .tab .mk{
    font-family:var(--mono); font-size:9px; font-weight:700; width:15px; height:15px;
    border-radius:50%; display:grid; place-items:center; flex:none;
    background:#21262d; color:var(--text-3);
  }
  #mk2 .tab.done .mk{background:var(--ok); color:var(--text-inv)}
  #mk2 .tab.warn .mk{background:var(--warn); color:var(--text-inv)}
  #mk2 .tab:hover{color:var(--text)}
  #mk2 .tab.on{
    background:var(--content-bg); border-color:var(--card-border);
    color:var(--text); border-bottom:1px solid var(--content-bg);
  }
  #mk2 .tab.on .mk{background:var(--accent); color:var(--text-inv)}
  #mk2 .pane{padding:18px 20px; overflow:auto; flex:1}
  #mk2 .grid2{display:grid; grid-template-columns:1.35fr 1fr; gap:14px; align-items:start}
  #mk2 .sect{padding:14px 16px}
  #mk2 .sect h4{margin:0 0 3px; font-size:12.5px; font-weight:700; letter-spacing:-.01em}
  #mk2 .sect .sub{margin:0 0 12px; font-size:11.5px; color:var(--text-2)}
  #mk2 .frow{display:flex; gap:7px; align-items:center; margin-bottom:9px}
  #mk2 .frow > .lbl{flex:0 0 78px}
  #mk2 table{width:100%; border-collapse:collapse; font-size:11.5px}
  #mk2 th{text-align:left; padding:5px 8px; color:var(--text-3); font-size:9.5px; letter-spacing:.11em; text-transform:uppercase; border-bottom:1px solid var(--card-border)}
  #mk2 td{padding:6px 8px; border-bottom:1px solid #23282f; vertical-align:middle}
  #mk2 tr:last-child td{border-bottom:none}
  #mk2 .mapped{color:var(--ok); font-family:var(--mono); font-size:11px}
  #mk2 .empty{color:var(--bad); font-family:var(--mono); font-size:11px}
  #mk2 .navbtns{display:flex; gap:8px; margin-top:14px; padding-top:13px; border-top:1px solid var(--card-border)}
  #mk2 .navbtns .sp{margin-left:auto}
  #mk2 .kv{display:flex; justify-content:space-between; gap:10px; padding:5px 0; font-size:11.5px; border-bottom:1px solid #23282f}
  #mk2 .kv:last-child{border-bottom:none}
  #mk2 .kv b{font-variant-numeric:tabular-nums}
  @media (max-width:900px){#mk2 .grid2{grid-template-columns:1fr} #mk2 .tabs{overflow-x:auto}}
"""

MK2 = """
      <div class="titlebar">
        <h2>Playlist Gap</h2>
        <span class="crumb">Liked videos · YouTube Music</span>
        <div class="right"><button class="btn sm">Saved lists (4)</button></div>
      </div>
      <div class="tabs">
        <button class="tab done"><span class="mk">✓</span>Source</button>
        <button class="tab done"><span class="mk">✓</span>Library</button>
        <button class="tab on"><span class="mk">3</span>Read &amp; split</button>
        <button class="tab warn"><span class="mk">!</span>Triage <span class="pill warn">68</span></button>
        <button class="tab"><span class="mk">5</span>Download list</button>
      </div>
      <div class="pane">
        <div class="grid2">
          <div class="card sect">
            <h4>How each row was read</h4>
            <p class="sub">Rows the app could not split confidently are probed against the library with several candidate splits. Nothing is committed until a lookup agrees.</p>
            <table>
              <thead><tr><th>Wanted row</th><th>Artist</th><th>Title</th><th>Read as</th></tr></thead>
              <tbody>
                <tr><td class="track">Kate Bush - Running Up That Hill (A Deal With God)</td><td class="mapped">Kate Bush</td><td class="mapped">Running Up That Hill (A Deal With God)</td><td><span class="pill ok">clean</span></td></tr>
                <tr><td class="track">Wavebeatmaker - Resonance</td><td class="mapped">Wavebeatmaker</td><td class="mapped">Resonance</td><td><span class="pill ok">clean</span></td></tr>
                <tr><td class="track">FIFTY FIFTY - Cupid (Twin Version)</td><td class="mapped">FIFTY FIFTY</td><td class="mapped">Cupid <span class="t3">+ (Twin Version)</span></td><td><span class="pill ok">clean</span></td></tr>
                <tr><td class="track">French 79 · New Constellations - Colors Collide</td><td class="empty">2 candidates</td><td class="empty">2 candidates</td><td><span class="pill warn">ambiguous</span></td></tr>
                <tr><td class="track">(Triple Vibe) HOME - Resonance but it's beats 3,3</td><td class="mapped">HOME</td><td class="mapped">Resonance <span class="t3">+ fan edit</span></td><td><span class="pill warn">derivative</span></td></tr>
                <tr><td class="track">Self Aware x Babydoll</td><td class="empty">none found</td><td class="empty">none found</td><td><span class="pill bad">mashup</span></td></tr>
                <tr><td class="track t3">(blank row)</td><td class="empty">—</td><td class="empty">—</td><td><span class="pill bad">unavailable</span></td></tr>
              </tbody>
            </table>
            <div class="navbtns">
              <button class="btn">← Library</button>
              <button class="btn sm">Edit split rules…</button>
              <button class="btn pri sp">Compare against library →</button>
            </div>
          </div>

          <div style="display:flex; flex-direction:column; gap:14px">
            <div class="card sect">
              <h4>This step</h4>
              <p class="sub" style="margin-bottom:8px">Reading 1,204 rows.</p>
              <div class="kv"><span>Parsed cleanly</span><b class="num">1,189</b></div>
              <div class="kv"><span>Ambiguous delimiter</span><b class="num">9</b></div>
              <div class="kv"><span>Derivative / fan edit</span><b class="num">4</b></div>
              <div class="kv"><span>No artist found</span><b class="num">1</b></div>
              <div class="kv"><span>Unavailable upstream</span><b class="num">1</b></div>
            </div>
            <div class="card sect">
              <h4>Carried from earlier steps</h4>
              <div class="kv"><span>Source</span><b>ytmusicapi</b></div>
              <div class="kv"><span>Fields provided</span><b>id · artist · album · duration</b></div>
              <div class="kv"><span>Library snapshot</span><b class="num">8,412</b></div>
              <div class="kv"><span>Last compared</span><b>6 days ago</b></div>
              <div class="kv"><span>New since then</span><b class="num">14</b></div>
            </div>
          </div>
        </div>
      </div>
"""

# ── 3 · Workbench ────────────────────────────────────────────────────────────

MK3_CSS = """
  #mk3 .wb{display:grid; grid-template-columns:196px 1fr 250px; gap:0; flex:1; min-height:0; overflow:hidden}
  #mk3 .setup{border-right:1px solid var(--card-border); padding:13px 14px; overflow:auto; background:var(--sidebar-bg)}
  #mk3 .setup .grp{margin-bottom:15px}
  #mk3 .setup .grp > .lbl{display:block; margin-bottom:6px}
  #mk3 .setup .val{font-size:11.5px; margin-bottom:5px}
  #mk3 .setup .val b{font-weight:600}
  #mk3 .cols{display:grid; grid-template-columns:repeat(3,1fr); gap:9px; padding:13px 14px; overflow:hidden; min-height:0}
  #mk3 .col{display:flex; flex-direction:column; border:1px solid var(--card-border); border-radius:6px; background:var(--input-bg); min-height:0}
  #mk3 .col > .ch{
    display:flex; align-items:center; gap:7px; padding:8px 10px;
    border-bottom:1px solid var(--card-border); flex:none;
  }
  #mk3 .col > .ch .nm{font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase}
  #mk3 .col > .ch .ct{margin-left:auto; font-family:var(--mono); font-size:14px; font-weight:700}
  #mk3 .col.g{border-top:2px solid var(--ok)} #mk3 .col.g .nm,#mk3 .col.g .ct{color:var(--ok)}
  #mk3 .col.a{border-top:2px solid var(--warn)} #mk3 .col.a .nm,#mk3 .col.a .ct{color:var(--warn)}
  #mk3 .col.r{border-top:2px solid var(--bad)} #mk3 .col.r .nm,#mk3 .col.r .ct{color:var(--bad)}
  #mk3 .items{overflow:auto; padding:7px; display:flex; flex-direction:column; gap:5px; flex:1}
  #mk3 .it{
    background:var(--card-bg); border:1px solid var(--card-border); border-radius:4px;
    padding:7px 9px; cursor:grab;
  }
  #mk3 .it:hover{border-color:var(--accent)}
  #mk3 .it.sel{border-color:var(--accent); box-shadow:0 0 0 1px var(--accent-pressed)}
  #mk3 .it .nmm{font-family:var(--mono); font-size:10.5px; line-height:1.35; word-break:break-word}
  #mk3 .it .why{font-size:10px; color:var(--text-2); margin-top:3px}
  #mk3 .it.ghost{border-style:dashed; opacity:.55; cursor:default}
  #mk3 .drop{
    border:1px dashed var(--accent); border-radius:4px; padding:9px;
    text-align:center; font-size:10.5px; color:var(--accent); background:var(--accent-bg);
  }
  #mk3 .insp{border-left:1px solid var(--card-border); padding:13px 14px; overflow:auto; background:var(--sidebar-bg)}
  #mk3 .insp h4{margin:0 0 9px; font-size:12px; font-weight:700}
  #mk3 .cmp{display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:10.5px; margin-bottom:4px}
  #mk3 .cmp.one{grid-template-columns:1fr}
  #mk3 .cmp .hh{font-size:9px; letter-spacing:.1em; text-transform:uppercase; color:var(--text-3); padding-bottom:3px}
  #mk3 .fld{padding:4px 0; border-top:1px solid #23282f}
  #mk3 .fld .k{font-size:9px; letter-spacing:.09em; text-transform:uppercase; color:var(--text-3); margin-bottom:2px}
  #mk3 .fld .vv{font-family:var(--mono); font-size:10.5px; word-break:break-word}
  #mk3 .fld.hit .vv{color:var(--ok)}
  #mk3 .fld.miss .vv{color:var(--bad)}
  #mk3 .verdict{border:1px solid #5c4410; background:var(--warn-bg); border-radius:5px; padding:9px 10px; margin:11px 0; font-size:11px; line-height:1.5}
  #mk3 .insp .acts{display:flex; flex-direction:column; gap:6px}
  @media (max-width:1020px){#mk3 .wb{grid-template-columns:1fr} #mk3 .setup,#mk3 .insp{border:none} #mk3 .cols{grid-template-columns:1fr}}
"""

MK3 = """
      <div class="titlebar">
        <h2>Playlist Gap</h2>
        <span class="crumb">Liked videos · 1,204 wanted · drag rows between columns</span>
        <div class="right">
          <button class="btn sm">Re-compare</button>
          <button class="btn pri sm">Export 105 →</button>
        </div>
      </div>
      <div class="wb">
        <div class="setup">
          <div class="grp">
            <span class="lbl">Wanted list</span>
            <div class="val"><b>Liked videos</b></div>
            <div class="val t2">YouTube Music · ytmusicapi</div>
            <div class="val t2 num">1,204 rows · 14 new</div>
            <button class="btn sm" style="width:100%; margin-top:5px">Switch list…</button>
          </div>
          <div class="grp">
            <span class="lbl">Library</span>
            <div class="val num"><b>8,412</b> tracks</div>
            <div class="val t2">read 4 min ago</div>
            <div class="val t3" style="font-size:10.5px">Not Sorted ✓ · Quarantine ✓<br>Manual Review ✓ · Trash ✕</div>
            <button class="btn sm" style="width:100%; margin-top:5px">Refresh snapshot</button>
          </div>
          <div class="grp">
            <span class="lbl">Ladder</span>
            <div class="val t2" style="font-size:10.5px; line-height:1.7">
              0 · video ID <span class="pill ok">on</span><br>
              2 · core + artist <span class="pill ok">on</span><br>
              4 · fuzzy in artist <span class="pill ok">on</span><br>
              5b · filename <span class="pill ok">on</span>
            </div>
          </div>
          <div class="grp">
            <span class="lbl">Duration gate</span>
            <div class="val t2">same ≤ 3 s · differs &gt; 15 s</div>
            <input class="inp" value="3 / 15" style="margin-top:4px">
          </div>
        </div>

        <div class="cols">
          <div class="col g">
            <div class="ch"><span class="nm">Owned</span><span class="ct num">1,031</span></div>
            <div class="items">
              <div class="it"><div class="nmm">Kate Bush — Running Up That Hill (A Deal With God)</div><div class="why">video ID match</div></div>
              <div class="it"><div class="nmm">Wavebeatmaker — Resonance</div><div class="why">exact · duration 0 s apart</div></div>
              <div class="it"><div class="nmm">The Sways — Someday We Will Dream About Today</div><div class="why">filename match</div></div>
              <div class="it ghost"><div class="nmm t3">1,028 more…</div></div>
            </div>
          </div>
          <div class="col a">
            <div class="ch"><span class="nm">Unsure</span><span class="ct num">68</span></div>
            <div class="items">
              <div class="it sel"><div class="nmm">FIFTY FIFTY — Cupid (Twin Version)</div><div class="why">you own “Cupid” · 41 s shorter</div></div>
              <div class="it"><div class="nmm">(Triple Vibe) HOME — Resonance but it's beats 3,3</div><div class="why">fan edit of a track you own</div></div>
              <div class="it"><div class="nmm">Self Aware x Babydoll</div><div class="why">mashup · no artist found</div></div>
              <div class="it"><div class="nmm">French 79 · New Constellations — Colors Collide</div><div class="why">2 possible splits, both hit</div></div>
              <div class="drop">drop here to mark unsure</div>
              <div class="it ghost"><div class="nmm t3">64 more…</div></div>
            </div>
          </div>
          <div class="col r">
            <div class="ch"><span class="nm">Missing</span><span class="ct num">105</span></div>
            <div class="items">
              <div class="it"><div class="nmm">Yotto — Radiate</div><div class="why">nothing plausible found</div></div>
              <div class="it"><div class="nmm">Ben Böhmer — Breathing</div><div class="why">nothing plausible found</div></div>
              <div class="it"><div class="nmm">Nora En Pure — Enchantment</div><div class="why">only a remix owned</div></div>
              <div class="it ghost"><div class="nmm t3">102 more…</div></div>
            </div>
          </div>
        </div>

        <div class="insp">
          <h4>Evidence</h4>
          <div class="cmp one"><div class="hh">wanted&nbsp; ·&nbsp; best candidate in library</div></div>
          <div class="fld hit"><div class="k">Core title</div><div class="vv">Cupid  ·  Cupid</div></div>
          <div class="fld hit"><div class="k">Artist</div><div class="vv">FIFTY FIFTY  ·  FIFTY FIFTY</div></div>
          <div class="fld miss"><div class="k">Modifiers</div><div class="vv">(Twin Version)  ·  —</div></div>
          <div class="fld miss"><div class="k">Duration</div><div class="vv">2:54  ·  3:35   (41 s)</div></div>
          <div class="fld"><div class="k">Album</div><div class="vv t2">The Beginning: Cupid  ·  same</div></div>
          <div class="verdict">
            Same song, but “Twin Version” is a distinct official recording and the runtimes are
            41 s apart. Owning <span class="mono">Cupid</span> is not owning this.
          </div>
          <div class="acts">
            <button class="btn pri">Send to Missing</button>
            <button class="btn">Accept as owned</button>
            <button class="btn sm">▶ Play candidate</button>
            <button class="btn sm">Never want this</button>
          </div>
        </div>
      </div>
"""

# ── 4 · Triage Queue ────────────────────────────────────────────────────────

MK4_CSS = """
  #mk4 .q{display:grid; grid-template-columns:300px 1fr; flex:1; min-height:0; overflow:hidden}
  #mk4 .list{border-right:1px solid var(--card-border); display:flex; flex-direction:column; min-height:0; background:var(--sidebar-bg)}
  #mk4 .filters{display:flex; gap:3px; padding:9px 10px; border-bottom:1px solid var(--card-border); flex:none; flex-wrap:wrap}
  #mk4 .f{
    font-size:10.5px; font-weight:600; padding:3px 8px; border-radius:11px; cursor:pointer;
    border:1px solid var(--card-border); background:#21262d; color:var(--text-2); font-family:var(--ui);
  }
  #mk4 .f.on{background:var(--warn-bg); border-color:#5c4410; color:var(--warn)}
  #mk4 .rows{overflow:auto; flex:1}
  #mk4 .r{
    display:grid; grid-template-columns:16px 1fr; gap:8px; padding:8px 11px;
    border-bottom:1px solid #23282f; cursor:pointer; align-items:start;
  }
  #mk4 .r:hover{background:var(--sidebar-hover)}
  #mk4 .r.on{background:var(--sidebar-active); box-shadow:inset 3px 0 0 var(--accent)}
  #mk4 .r .dot{width:7px; height:7px; border-radius:50%; margin-top:4px; background:var(--warn)}
  #mk4 .r.read .dot{background:transparent; border:1px solid var(--text-3)}
  #mk4 .r .tt{font-family:var(--mono); font-size:10.5px; line-height:1.35; word-break:break-word}
  #mk4 .r .mm{font-size:10px; color:var(--text-2); margin-top:2px}
  #mk4 .r.read .tt{color:var(--text-2)}
  #mk4 .reader{display:flex; flex-direction:column; min-height:0; overflow:auto}
  #mk4 .rh{padding:16px 22px 13px; border-bottom:1px solid var(--card-border)}
  #mk4 .rh .grp{font-size:10.5px; color:var(--warn); font-weight:600; margin-bottom:5px}
  #mk4 .rh h3{margin:0 0 7px; font-size:16px; font-weight:700; letter-spacing:-.02em; font-family:var(--mono)}
  #mk4 .rh .meta{font-size:11.5px; color:var(--text-2)}
  #mk4 .rb{padding:16px 22px; display:flex; flex-direction:column; gap:14px}
  #mk4 .side{display:grid; grid-template-columns:1fr 1fr; gap:11px}
  #mk4 .side .bx{border:1px solid var(--card-border); border-radius:6px; padding:11px 13px; background:var(--card-bg)}
  #mk4 .side .bx.cand{border-color:var(--accent-pressed)}
  #mk4 .side .bx > .lbl{display:block; margin-bottom:7px}
  #mk4 .f2{padding:4px 0; border-top:1px solid #23282f; font-size:11px}
  #mk4 .f2:first-of-type{border-top:none}
  #mk4 .f2 .k{color:var(--text-3); font-size:9px; letter-spacing:.1em; text-transform:uppercase}
  #mk4 .f2 .v{font-family:var(--mono); font-size:11px}
  #mk4 .f2.d .v{color:var(--bad)}
  #mk4 .f2.s .v{color:var(--ok)}
  #mk4 .reason{border-left:3px solid var(--warn); background:var(--warn-bg); padding:11px 14px; border-radius:0 5px 5px 0; font-size:12px; line-height:1.55}
  #mk4 .keys{display:flex; gap:8px; flex-wrap:wrap; align-items:center}
  #mk4 kbd{
    font-family:var(--mono); font-size:10px; font-weight:700; background:#21262d;
    border:1px solid var(--card-border); border-bottom-width:2px; border-radius:4px;
    padding:2px 6px; color:var(--text);
  }
  #mk4 .bulk{
    border:1px solid var(--accent-pressed); background:var(--accent-bg); border-radius:6px;
    padding:11px 14px; font-size:11.5px; display:flex; gap:11px; align-items:center; flex-wrap:wrap;
  }
  #mk4 .bulk b{font-variant-numeric:tabular-nums}
  #mk4 .prog{height:3px; background:#21262d; flex:none}
  #mk4 .prog i{display:block; height:100%; width:32%; background:var(--warn)}
  @media (max-width:900px){#mk4 .q{grid-template-columns:1fr} #mk4 .side{grid-template-columns:1fr}}
"""

MK4 = """
      <div class="titlebar">
        <h2>Playlist Gap</h2>
        <span class="crumb">Unsure queue · 22 of 68 cleared</span>
        <div class="right">
          <span class="pill ok">1,031 owned</span>
          <span class="pill bad">105 missing</span>
          <button class="btn pri sm">Finish &amp; export</button>
        </div>
      </div>
      <div class="prog"><i></i></div>
      <div class="q">
        <div class="list">
          <div class="filters">
            <button class="f on">Unsure 46</button>
            <button class="f">Missing 105</button>
            <button class="f">Owned 1,031</button>
            <button class="f">All</button>
          </div>
          <div class="rows">
            <div class="r on"><span class="dot"></span><div><div class="tt">FIFTY FIFTY - Cupid (Twin Version)</div><div class="mm">version differs · 41 s apart</div></div></div>
            <div class="r"><span class="dot"></span><div><div class="tt">(Triple Vibe) HOME - Resonance but it's beats 3,3</div><div class="mm">fan edit of an owned track</div></div></div>
            <div class="r"><span class="dot"></span><div><div class="tt">Self Aware x Babydoll</div><div class="mm">mashup · no artist found</div></div></div>
            <div class="r"><span class="dot"></span><div><div class="tt">French 79 · New Constellations - Colors Collide</div><div class="mm">2 splits, both plausible</div></div></div>
            <div class="r"><span class="dot"></span><div><div class="tt">ODESZA - Bloom (Extended)</div><div class="mm">version differs · 2:11 apart</div></div></div>
            <div class="r read"><span class="dot"></span><div><div class="tt">Lane 8 - Fingerprint</div><div class="mm">→ marked owned</div></div></div>
            <div class="r read"><span class="dot"></span><div><div class="tt">RÜFÜS DU SOL - Innerbloom (Live)</div><div class="mm">→ sent to missing</div></div></div>
            <div class="r read"><span class="dot"></span><div><div class="tt">Bonobo - Kerala</div><div class="mm">→ marked owned</div></div></div>
          </div>
        </div>

        <div class="reader">
          <div class="rh">
            <div class="grp">Version differs · 1 of 17 in this group</div>
            <h3>FIFTY FIFTY - Cupid (Twin Version)</h3>
            <div class="meta">Wanted from <b>Liked videos</b> · added 14 Aug · never compared before</div>
          </div>
          <div class="rb">
            <div class="bulk">
              <span><b>17 rows</b> in this queue differ only by a version modifier with a runtime gap over 15 s.</span>
              <button class="btn pri sm">Send all 17 to missing</button>
              <button class="btn sm">Keep reviewing one by one</button>
            </div>

            <div class="side">
              <div class="bx">
                <span class="lbl">Wanted</span>
                <div class="f2"><div class="k">Core title</div><div class="v">Cupid</div></div>
                <div class="f2"><div class="k">Artist</div><div class="v">FIFTY FIFTY</div></div>
                <div class="f2 d"><div class="k">Modifiers</div><div class="v">(Twin Version)</div></div>
                <div class="f2 d"><div class="k">Duration</div><div class="v">2:54</div></div>
                <div class="f2"><div class="k">Album</div><div class="v">The Beginning: Cupid</div></div>
              </div>
              <div class="bx cand">
                <span class="lbl">Best candidate in library</span>
                <div class="f2 s"><div class="k">Core title</div><div class="v">Cupid</div></div>
                <div class="f2 s"><div class="k">Artist</div><div class="v">FIFTY FIFTY</div></div>
                <div class="f2 d"><div class="k">Modifiers</div><div class="v">— none —</div></div>
                <div class="f2 d"><div class="k">Duration</div><div class="v">3:35</div></div>
                <div class="f2"><div class="k">File</div><div class="v">By Artist/FIFTY FIFTY/Cupid.flac</div></div>
              </div>
            </div>

            <div class="reason">
              Core title and artist agree, but “Twin Version” is a distinct official recording and
              the runtimes are <b>41 s</b> apart — past the 15 s gate. Owning
              <span class="mono">Cupid</span> is not owning this one.
            </div>

            <div class="keys">
              <button class="btn pri">Missing <kbd>M</kbd></button>
              <button class="btn">Owned <kbd>A</kbd></button>
              <button class="btn">Never want <kbd>X</kbd></button>
              <button class="btn sm">▶ Play <kbd>Space</kbd></button>
              <span class="t3" style="margin-left:auto; font-size:10.5px"><kbd>J</kbd><kbd>K</kbd> move · <kbd>U</kbd> undo</span>
            </div>
          </div>
        </div>
      </div>
"""

# ── 5 · Answer First ────────────────────────────────────────────────────────

MK5_CSS = """
  #mk5 .strip{
    display:flex; gap:9px; align-items:stretch; padding:13px 20px;
    border-bottom:1px solid var(--card-border); flex-wrap:wrap; background:var(--sidebar-bg); flex:none;
  }
  #mk5 .st{border:1px solid var(--card-border); border-radius:6px; padding:7px 13px; min-width:108px; background:var(--card-bg)}
  #mk5 .st .lbl{display:block; margin-bottom:1px}
  #mk5 .st .v{font-size:20px; font-weight:700; letter-spacing:-.025em; font-variant-numeric:tabular-nums}
  #mk5 .st.g{border-color:#1d4a30} #mk5 .st.g .v{color:var(--ok)}
  #mk5 .st.a{border-color:#5c4410} #mk5 .st.a .v{color:var(--warn)}
  #mk5 .st.r{border-color:var(--bad); border-width:2px; background:var(--bad-bg)} #mk5 .st.r .v{color:var(--bad)}
  #mk5 .st .sub{font-size:10px; color:var(--text-2)}
  #mk5 .strip .rt{margin-left:auto; display:flex; flex-direction:column; gap:5px; justify-content:center}
  #mk5 .banner{
    display:flex; gap:11px; align-items:center; padding:9px 20px; font-size:12px;
    background:var(--warn-bg); border-bottom:1px solid #5c4410; flex-wrap:wrap; flex:none;
  }
  #mk5 .banner b{color:var(--warn)}
  #mk5 .banner .sp{margin-left:auto}
  #mk5 .out{padding:18px 20px; overflow:auto; flex:1}
  #mk5 .outhd{display:flex; align-items:baseline; gap:11px; margin-bottom:11px; flex-wrap:wrap}
  #mk5 .outhd h3{margin:0; font-size:14.5px; font-weight:700; letter-spacing:-.015em}
  #mk5 .outhd .tools{margin-left:auto; display:flex; gap:6px; align-items:center}
  #mk5 .seg{display:flex; border:1px solid var(--card-border); border-radius:5px; overflow:hidden}
  #mk5 .seg button{
    font-family:var(--ui); font-size:11px; font-weight:600; padding:4px 10px; cursor:pointer;
    background:#21262d; color:var(--text-2); border:none; border-right:1px solid var(--card-border);
  }
  #mk5 .seg button:last-child{border-right:none}
  #mk5 .seg button.on{background:var(--accent); color:var(--text-inv)}
  #mk5 .listbox{
    background:var(--input-bg); border:1px solid var(--card-border); border-radius:6px;
    font-family:var(--mono); font-size:11.5px; line-height:1.85; padding:13px 15px;
    max-height:268px; overflow:auto; white-space:pre; color:var(--text);
  }
  #mk5 .listbox .grp{color:var(--accent); font-weight:600}
  #mk5 .listbox .dim{color:var(--text-3)}
  #mk5 .exp{margin-top:15px; border-top:1px solid var(--card-border); padding-top:13px}
  #mk5 .acc{border:1px solid var(--card-border); border-radius:6px; margin-bottom:7px; background:var(--card-bg)}
  #mk5 .acc > .hd{
    display:flex; align-items:center; gap:9px; padding:9px 13px; cursor:pointer; font-size:12px;
  }
  #mk5 .acc > .hd .ar{color:var(--text-3); font-size:10px}
  #mk5 .acc > .hd .nm{font-weight:600}
  #mk5 .acc > .hd .sm2{color:var(--text-2); font-size:11px}
  #mk5 .acc > .hd .rr{margin-left:auto; display:flex; gap:6px; align-items:center}
  #mk5 .acc > .bd{padding:0 13px 12px; border-top:1px solid var(--card-border); padding-top:11px}
  #mk5 .kv2{display:flex; justify-content:space-between; gap:11px; font-size:11.5px; padding:4px 0; border-bottom:1px solid #23282f}
  #mk5 .kv2:last-child{border-bottom:none}
  #mk5 .kv2 b{font-variant-numeric:tabular-nums}
  #mk5 .grpline{
    display:flex; gap:9px; align-items:center; padding:7px 0; border-bottom:1px solid #23282f;
    font-size:11.5px; flex-wrap:wrap;
  }
  #mk5 .grpline:last-child{border-bottom:none}
  #mk5 .grpline .cnt{font-family:var(--mono); font-weight:700; color:var(--warn); min-width:26px}
  #mk5 .grpline .aa{margin-left:auto; display:flex; gap:5px}
"""

MK5 = """
      <div class="titlebar">
        <h2>Playlist Gap</h2>
        <span class="crumb">Liked videos · compared 2 min ago · 6 days since last run</span>
        <div class="right"><button class="btn sm">Saved lists (4)</button><button class="btn sm">Re-compare</button></div>
      </div>
      <div class="strip">
        <div class="st"><span class="lbl">Wanted</span><div class="v">1,204</div><div class="sub">14 new since last run</div></div>
        <div class="st g"><span class="lbl">Already owned</span><div class="v">1,031</div><div class="sub">86% of the list</div></div>
        <div class="st a"><span class="lbl">Unsure</span><div class="v">68</div><div class="sub">4 patterns</div></div>
        <div class="st r"><span class="lbl">Go download</span><div class="v">105</div><div class="sub">the answer</div></div>
        <div class="rt">
          <button class="btn pri">⧉  Copy all 105</button>
          <button class="btn sm">Save as .txt / .csv / report</button>
        </div>
      </div>
      <div class="banner">
        <b>68 rows are unsure.</b>
        <span class="t2">Clearing them could move some into this list. Four bulk decisions will handle 62 of them.</span>
        <button class="btn sm sp">Review unsure →</button>
      </div>
      <div class="out">
        <div class="outhd">
          <h3>Download list</h3>
          <span class="t2 num">105 tracks · 38 artists · 11 albums</span>
          <div class="tools">
            <div class="seg">
              <button class="on">Group by album</button>
              <button>Flat list</button>
              <button>By artist</button>
            </div>
            <button class="btn sm">Mark exported</button>
          </div>
        </div>
        <div class="listbox"><span class="grp"># Ben Böhmer — Begin Again  (4 of 12 tracks missing)</span>
Ben Böhmer - Breathing
Ben Böhmer - Sailing
Ben Böhmer - After Earth
Ben Böhmer - Lost In Thought
<span class="dim"># → whole album is probably the better grab</span>

<span class="grp"># Yotto — Erased Dreams  (2 of 8 missing)</span>
Yotto - Radiate
Yotto - Nova

<span class="grp"># Singles &amp; one-offs  (99)</span>
Nora En Pure - Enchantment
ODESZA - Bloom
Lane 8 - Shatter
Tinlicker - Because You Move Me
Marsh - Alpine
<span class="dim">… 94 more</span></div>

        <div class="exp">
          <div class="acc">
            <div class="hd"><span class="ar">▾</span><span class="nm">Unsure</span><span class="sm2">— 4 patterns cover 62 of 68 rows</span><div class="rr"><span class="pill warn">68</span></div></div>
            <div class="bd">
              <div class="grpline"><span class="cnt">29</span><span>differ only by a <span class="mono">feat.</span> credit the library file lacks</span><span class="aa"><button class="btn sm">Accept all as owned</button><button class="btn sm">Review</button></span></div>
              <div class="grpline"><span class="cnt">17</span><span>version modifier with a runtime gap over 15 s</span><span class="aa"><button class="btn sm">Send all to missing</button><button class="btn sm">Review</button></span></div>
              <div class="grpline"><span class="cnt">11</span><span>fan edits / mashups of tracks already owned</span><span class="aa"><button class="btn sm">Send all to missing</button><button class="btn sm">Review</button></span></div>
              <div class="grpline"><span class="cnt">5</span><span>ambiguous artist/title split, more than one candidate hit</span><span class="aa"><button class="btn sm">Review</button></span></div>
              <div class="grpline"><span class="cnt">6</span><span>weak match, no corroborating signal</span><span class="aa"><button class="btn sm">Review individually</button></span></div>
            </div>
          </div>
          <div class="acc">
            <div class="hd"><span class="ar">▸</span><span class="nm">How this was worked out</span><span class="sm2">— source, snapshot, ladder rungs, thresholds</span><div class="rr"><span class="pill ok">all green</span></div></div>
          </div>
          <div class="acc">
            <div class="hd"><span class="ar">▸</span><span class="nm">Pending downloads</span><span class="sm2">— 42 exported 6 days ago, not yet verified</span><div class="rr"><span class="pill acc">42</span><button class="btn sm">Verify in Library Sync</button></div></div>
          </div>
          <div class="acc">
            <div class="hd"><span class="ar">▸</span><span class="nm">Never want</span><span class="sm2">— 7 rows you excluded permanently</span><div class="rr"><span class="pill mute">7</span></div></div>
          </div>
        </div>
      </div>
"""

FRAGS = {"mk1": MK1, "mk2": MK2, "mk3": MK3, "mk4": MK4, "mk5": MK5}
CSSES = {"mk1": MK1_CSS, "mk2": MK2_CSS, "mk3": MK3_CSS, "mk4": MK4_CSS, "mk5": MK5_CSS}

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap">')


def app_block(slug: str, title: str, thesis: str) -> str:
    return (
        f'    <section class="app" id="{slug}">\n'
        f'{RAIL}'
        f'      <div class="main">{FRAGS[slug]}{STATUS}      </div>\n'
        f'    </section>\n'
    )


def standalone(name: str, slug: str, title: str, thesis: str) -> str:
    return f"""<title>Playlist Gap — {title}</title>
{FONTS}
<style>{BASE_CSS}{CSSES[slug]}</style>
<div class="shell">
  <header>
    <h1><em>Mockup {name[:2]}</em>{title}</h1>
    <p class="thesis">{thesis}</p>
  </header>
{app_block(slug, title, thesis)}</div>
"""


def index() -> str:
    css = BASE_CSS + "".join(CSSES[s] for _n, s, _t, _d in MOCKUPS) + """
  .picker{display:flex; gap:5px; flex-wrap:wrap; margin:0 0 14px}
  .picker button{
    font-family:var(--ui); font-size:12px; font-weight:600; padding:7px 13px; cursor:pointer;
    background:var(--card-bg); color:var(--text-2); border:1px solid var(--card-border); border-radius:6px;
    display:flex; gap:7px; align-items:center;
  }
  .picker button:hover{color:var(--text); border-color:var(--accent)}
  .picker button[aria-selected="true"]{background:var(--accent); border-color:var(--accent); color:var(--text-inv)}
  .picker button .k{font-family:var(--mono); font-size:10px; opacity:.75}
  .picker button:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
  .pgtitle{margin:0 0 4px; font-size:22px; font-weight:700; letter-spacing:-.025em}
  .pgsub{margin:0 0 16px; font-size:13px; color:var(--text-2); max-width:80ch}
  .mkwrap[hidden]{display:none !important}
  .mkwrap h1{margin:0 0 5px; font-size:17px; font-weight:700; letter-spacing:-.02em}
  .mkwrap h1 em{font-style:normal; font-family:var(--mono); font-size:11.5px; color:var(--accent); font-weight:600; margin-right:8px}
  .mkwrap .thesis{margin:0 0 13px; font-size:12.5px; color:var(--text-2); max-width:82ch}
"""
    picker = "\n".join(
        f'    <button role="tab" aria-selected="{"true" if i == 0 else "false"}" data-t="{slug}">'
        f'<span class="k">{name[:2]}</span>{title}</button>'
        for i, (name, slug, title, _d) in enumerate(MOCKUPS)
    )
    panels = "\n".join(
        f'  <div class="mkwrap" id="w-{slug}"{"" if i == 0 else " hidden"}>\n'
        f'    <h1><em>Mockup {name[:2]}</em>{title}</h1>\n'
        f'    <p class="thesis">{thesis}</p>\n'
        f'{app_block(slug, title, thesis)}  </div>'
        for i, (name, slug, title, thesis) in enumerate(MOCKUPS)
    )
    return f"""<title>Playlist Gap Mockups</title>
{FONTS}
<style>{css}</style>
<div class="shell">
  <h1 class="pgtitle">Playlist Gap — five interfaces</h1>
  <p class="pgsub">Same feature, same data, five interaction models. Palette and sidebar are the
  app's real Midnight theme, so these read at roughly the density the Qt build would have.
  Pick one and we refine it.</p>
  <div class="picker" role="tablist">
{picker}
  </div>
{panels}
</div>
<script>
  const picker = document.querySelector('.picker');
  picker.addEventListener('click', e => {{
    const b = e.target.closest('button[data-t]');
    if (!b) return;
    picker.querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
    document.querySelectorAll('.mkwrap').forEach(w => {{ w.hidden = w.id !== 'w-' + b.dataset.t; }});
    window.scrollTo({{ top: 0, behavior: 'instant' }});
  }});
  document.addEventListener('keydown', e => {{
    if (e.target.matches('input, textarea')) return;
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= {len(MOCKUPS)}) picker.querySelectorAll('button')[n - 1].click();
  }});
</script>
"""


def main() -> None:
    for name, slug, title, thesis in MOCKUPS:
        path = os.path.join(OUT_DIR, f"{name}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(standalone(name, slug, title, thesis))
        print("wrote", os.path.basename(path))
    path = os.path.join(OUT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(index())
    print("wrote index.html")


if __name__ == "__main__":
    main()
