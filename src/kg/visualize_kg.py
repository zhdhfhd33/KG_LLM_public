"""
KG 시각화 — Pyvis 인터랙티브 HTML

[동작]
  1. data/kg_store.json을 읽어 entity 노드와 트리플 기반 엣지를 추출한다.
  2. Pyvis Network로 인터랙티브 그래프를 생성한다.
  3. 슬라이더 UI 패널을 HTML에 후처리로 주입한다.
  4. data/kg_graph.html로 저장하고 브라우저에서 자동으로 연다.

[조작법 (브라우저)]
  - 노드 드래그    : 위치 이동
  - 스크롤         : 확대/축소
  - 우상단 패널    : 노드 간격·연결 강도·반발력·감쇠 슬라이더 실시간 조절
  - 물리엔진 토글  : 노드 위치 고정/해제
  - 화면 맞춤      : 전체 그래프가 뷰포트에 들어오도록 자동 조정
"""

import json
import webbrowser
from pathlib import Path
from pyvis.network import Network

# ── 경로 ──────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent.parent.parent / "data"
KG_STORE_DIR   = DATA_DIR / "kg_store"
KG_VISUALS_DIR = DATA_DIR / "kg_visuals"
KG_STORE_PATH  = KG_STORE_DIR / "kg_store.json"
OUTPUT_PATH    = KG_VISUALS_DIR / "kg_graph_without_ontology.html"

# ── 시각화 옵션 ────────────────────────────────────────────────────────────────
SKIP_RELATIONS = {"SOURCE", "NEXT", "PREVIOUS"}
MAX_NODES = 300

# ── 슬라이더 패널 (HTML 후처리로 </body> 직전에 주입) ────────────────────────
_CONTROL_PANEL_HTML = """
<div id="custom-controls">
  <b>그래프 조절 패널</b>
  <hr>

  <div class="ctrl-row">
    <label>노드 간격 <span id="val-springLength">95</span></label>
    <input type="range" id="sl-springLength" min="30" max="600" step="5" value="95">
  </div>
  <div class="ctrl-row">
    <label>연결 강도 <span id="val-springConstant">0.04</span></label>
    <input type="range" id="sl-springConstant" min="0.01" max="0.30" step="0.01" value="0.04">
  </div>
  <div class="ctrl-row">
    <label>반발력 <span id="val-gravity">2000</span></label>
    <input type="range" id="sl-gravity" min="100" max="15000" step="100" value="2000">
  </div>
  <div class="ctrl-row">
    <label>감쇠 <span id="val-damping">0.09</span></label>
    <input type="range" id="sl-damping" min="0.01" max="1.00" step="0.01" value="0.09">
  </div>

  <hr>
  <div class="btn-row">
    <button id="btn-physics" onclick="togglePhysics()">물리엔진 끄기</button>
    <button onclick="network.fit()">화면 맞춤</button>
  </div>
</div>

<style>
#custom-controls {
  position: fixed;
  top: 12px;
  right: 12px;
  background: rgba(15, 15, 35, 0.93);
  border: 1px solid #4a4a6a;
  border-radius: 10px;
  padding: 14px 16px;
  color: #dde;
  font-family: 'Segoe UI', sans-serif;
  font-size: 13px;
  z-index: 9999;
  min-width: 230px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.6);
}
#custom-controls b  { font-size: 14px; letter-spacing: 0.3px; }
#custom-controls hr { border: none; border-top: 1px solid #3a3a5a; margin: 10px 0; }
.ctrl-row           { margin: 9px 0; }
.ctrl-row label     { display: flex; justify-content: space-between; margin-bottom: 4px; color: #bbc; }
.ctrl-row label span { color: #7bf; font-weight: bold; min-width: 38px; text-align: right; }
.ctrl-row input[type=range] { width: 100%; accent-color: #3498db; cursor: pointer; }
.btn-row            { display: flex; gap: 8px; margin-top: 4px; }
.btn-row button {
  flex: 1; padding: 6px 4px;
  background: #1e2d42; color: #cde;
  border: 1px solid #4a5a7a; border-radius: 6px;
  cursor: pointer; font-size: 12px; transition: background 0.15s;
}
.btn-row button:hover { background: #2a3f5f; }
</style>

<script>
var physicsEnabled = true;

function togglePhysics() {
  physicsEnabled = !physicsEnabled;
  network.setOptions({ physics: { enabled: physicsEnabled } });
  document.getElementById('btn-physics').textContent =
    physicsEnabled ? '물리엔진 끄기' : '물리엔진 켜기';
}

function waitForNetwork(cb) {
  if (typeof network !== 'undefined' && network !== null) { cb(); }
  else { setTimeout(function () { waitForNetwork(cb); }, 100); }
}

waitForNetwork(function () {
  var sliders = [
    { id: 'springLength',   param: 'springLength',         invert: false },
    { id: 'springConstant', param: 'springConstant',       invert: false },
    { id: 'gravity',        param: 'gravitationalConstant', invert: true  },
    { id: 'damping',        param: 'damping',              invert: false  },
  ];

  sliders.forEach(function (s) {
    var el = document.getElementById('sl-' + s.id);
    var vl = document.getElementById('val-' + s.id);
    el.addEventListener('input', function () {
      var v = parseFloat(this.value);
      vl.textContent = v;
      var opts = { physics: { barnesHut: {} } };
      opts.physics.barnesHut[s.param] = s.invert ? -v : v;
      network.setOptions(opts);
    });
  });
});
</script>
"""


def load_kg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_network(data: dict) -> Network:
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
    )

    nodes_data = data["nodes"]
    triplets   = data["triplets"]

    entity_names = {
        v["name"]
        for v in nodes_data.values()
        if v.get("label") != "text_chunk" and v.get("name")
    }

    used_subjects, used_objects, valid_triplets = set(), set(), []
    for t in triplets:
        subj, rel, obj = t[0], t[1], t[2]
        if rel in SKIP_RELATIONS:
            continue
        if subj not in entity_names and obj not in entity_names:
            continue
        valid_triplets.append((subj, rel, obj))
        used_subjects.add(subj)
        used_objects.add(obj)

    if MAX_NODES and len(used_subjects | used_objects) > MAX_NODES:
        from collections import Counter
        freq = Counter()
        for subj, _, obj in valid_triplets:
            freq[subj] += 1
            freq[obj]  += 1
        top_nodes = {n for n, _ in freq.most_common(MAX_NODES)}
        valid_triplets = [(s, r, o) for s, r, o in valid_triplets if s in top_nodes and o in top_nodes]
        used_subjects  = {s for s, _, _ in valid_triplets}
        used_objects   = {o for _, _, o in valid_triplets}

    all_nodes = used_subjects | used_objects
    both      = used_subjects & used_objects

    for name in all_nodes:
        if name in both:
            color, shape = "#f39c12", "dot"
        elif name in used_subjects:
            color, shape = "#3498db", "dot"
        else:
            color, shape = "#2ecc71", "diamond"
        net.add_node(name, label=name, color=color, shape=shape, title=name)

    for subj, rel, obj in valid_triplets:
        net.add_edge(subj, obj, label=rel, title=rel, color="#aaaaaa", arrows="to")

    return net


def inject_controls(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("</body>", _CONTROL_PANEL_HTML + "\n</body>", 1)
    html_path.write_text(html, encoding="utf-8")


def main(
    kg_path: Path = KG_STORE_PATH,
    output_path: Path = OUTPUT_PATH,
):
    if not kg_path.exists():
        raise FileNotFoundError(f"KG 파일이 없습니다: {kg_path}\nbuild_kg.py를 먼저 실행하세요.")

    print(f"[로드] {kg_path}")
    data = load_kg(kg_path)

    total_nodes    = len([v for v in data["nodes"].values() if v.get("label") != "text_chunk"])
    total_triplets = len(data["triplets"])
    print(f"      entity 노드: {total_nodes}개 / 트리플: {total_triplets}개")

    print("[시각화] 그래프 생성 중...")
    net = build_network(data)
    net.write_html(str(output_path))

    inject_controls(output_path)
    print(f"[저장] {output_path}")

    print("[열기] 브라우저에서 그래프를 엽니다...")
    webbrowser.open(output_path.resolve().as_uri())

    print("\n=== 시각화 완료 ===")
    print(f"파일 위치: {output_path}")
    print("브라우저에서 열리지 않으면 위 경로를 직접 브라우저로 드래그하세요.")


if __name__ == "__main__":
    main()
