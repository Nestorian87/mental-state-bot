from __future__ import annotations

import json
from html import escape
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db import repositories as repo


async def build_memory_graph_html(
    session: AsyncSession,
    *,
    user_id,
    node_limit: int = 500,
    edge_limit: int = 1000,
) -> bytes:
    nodes = list(await repo.list_memory_nodes(session, user_id=user_id, limit=node_limit))
    edges = list(await repo.list_memory_edges(session, user_id=user_id, limit=edge_limit))
    node_payload = [
        {
            "id": str(node.id),
            "label": node.label,
            "kind": node.kind,
            "summary": node.summary or "",
            "confidence": float(node.confidence or 0),
            "weight": float(node.weight or 0),
            "status": node.status,
            "aliases": node.aliases or [],
        }
        for node in nodes
    ]
    node_ids = {item["id"] for item in node_payload}
    edge_payload = [
        {
            "id": str(edge.id),
            "from": str(edge.source_node_id),
            "to": str(edge.target_node_id),
            "label": edge.relation_label,
            "summary": edge.summary or "",
            "confidence": float(edge.confidence or 0),
            "weight": float(edge.weight or 0),
            "evidence_count": edge.evidence_count,
        }
        for edge in edges
        if str(edge.source_node_id) in node_ids and str(edge.target_node_id) in node_ids
    ]
    return _render_html(node_payload, edge_payload)


def _render_html(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> bytes:
    payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False).replace("<", "\\u003c")
    fallback_nodes = "".join(
        f"<li><strong>{escape(str(node['label']))}</strong> · {escape(str(node['kind']))} · "
        f"впевненість {float(node['confidence']):.0%} · вага {float(node['weight']):.2f}"
        f"{(' — ' + escape(str(node['summary']))) if node['summary'] else ''}</li>"
        for node in nodes
    ) or "<li>У графі поки немає вузлів.</li>"
    fallback_edges = "".join(
        f"<li>{escape(_label_for_id(nodes, edge['from']))} → "
        f"{escape(_label_for_id(nodes, edge['to']))}: {escape(str(edge['label']))}"
        f"{(' — ' + escape(str(edge['summary']))) if edge['summary'] else ''}</li>"
        for edge in edges
    ) or "<li>У графі поки немає зв’язків.</li>"
    html = f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Граф пам’яті</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{ color-scheme: dark; --bg:#111827; --panel:#1f2937; --muted:#9ca3af; --text:#f3f4f6; --line:#374151; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:15px system-ui,-apple-system,sans-serif; }}
    header {{ padding:18px 20px 12px; border-bottom:1px solid var(--line); }} h1 {{ margin:0 0 6px; font-size:22px; }}
    p {{ margin:4px 0; color:var(--muted); }} main {{ padding:14px; max-width:1400px; margin:auto; }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }} input, button {{ background:var(--panel); border:1px solid var(--line); color:var(--text); border-radius:6px; padding:9px 11px; }}
    input {{ flex:1; min-width:220px; }} button {{ cursor:pointer; }} #network {{ height:70vh; min-height:420px; background:#0b1220; border:1px solid var(--line); border-radius:6px; }}
    details {{ margin-top:14px; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px 12px; }} summary {{ cursor:pointer; }} li {{ margin:7px 0; line-height:1.4; }}
    .empty {{ color:var(--muted); padding:20px; }}
  </style>
</head>
<body>
  <header><h1>Граф пам’яті</h1><p id="stats"></p><p>Це візуалізація збережених AI-кандидатів, а не підтверджена біографія.</p></header>
  <main>
    <div class="toolbar"><input id="filter" placeholder="Фільтр за назвою, типом або зв’язком"><button id="fit">Підігнати граф</button><button id="reset">Скинути фільтр</button></div>
    <div id="network"><div class="empty">Граф завантажується…</div></div>
    <details><summary>Список вузлів</summary><ul>{fallback_nodes}</ul></details>
    <details><summary>Список зв’язків</summary><ul>{fallback_edges}</ul></details>
  </main>
  <script>
    const payload = {payload};
    const stats = document.getElementById('stats');
    stats.textContent = `Вузлів: ${{payload.nodes.length}} · Зв’язків: ${{payload.edges.length}}`;
    const colors = {{person:'#60a5fa', place:'#34d399', project:'#fbbf24', activity:'#f472b6', lexicon:'#fb7185', concept:'#a78bfa'}};
    const htmlEscape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    const makeNodes = (items) => new vis.DataSet(items.map(n => ({{...n, title:`<b>${{htmlEscape(n.label)}}</b><br>Тип: ${{htmlEscape(n.kind)}}<br>Впевненість: ${{(n.confidence*100).toFixed(0)}}%<br>Вага: ${{n.weight.toFixed(2)}}${{n.summary ? `<br>${{htmlEscape(n.summary)}}` : ''}}`, color:{{background:colors[n.kind] || '#94a3b8', border:'#e5e7eb'}}, size:14 + n.weight*14}})));
    const allNodes = payload.nodes, allEdges = payload.edges;
    const container = document.getElementById('network');
    let network = null, nodeData = null, edgeData = null;
    function render(query='') {{
      const q = query.trim().toLowerCase();
      const visible = allNodes.filter(n => !q || [n.label,n.kind,n.summary,...n.aliases].join(' ').toLowerCase().includes(q));
      const ids = new Set(visible.map(n => n.id));
      const visibleEdges = allEdges.filter(e => ids.has(e.from) && ids.has(e.to) && (!q || e.label.toLowerCase().includes(q) || e.summary.toLowerCase().includes(q)));
      nodeData = makeNodes(visible); edgeData = new vis.DataSet(visibleEdges.map(e => ({{...e, arrows:'to', font:{{color:'#d1d5db', size:12, strokeWidth:0}}, color:{{color:'#64748b', highlight:'#e5e7eb'}}, width:1+e.weight*3, title:`${{htmlEscape(e.label)}} · впевненість ${{(e.confidence*100).toFixed(0)}}% · доказів ${{e.evidence_count}}`}})));
      if (network) network.destroy();
      network = new vis.Network(container, {{nodes:nodeData, edges:edgeData}}, {{nodes:{{shape:'dot', font:{{color:'#f3f4f6', size:14}}}}, edges:{{smooth:{{type:'dynamic'}}, length:180}}, physics:{{stabilization:{{iterations:300}}, barnesHut:{{gravitationalConstant:-2200, springLength:180}}}}, interaction:{{hover:true, navigationButtons:true, keyboard:true}}}});
    }}
    if (window.vis) render(); else container.innerHTML = '<div class="empty">Не вдалося завантажити інтерактивну бібліотеку. Нижче доступні списки вузлів і зв’язків.</div>';
    document.getElementById('filter').addEventListener('input', e => render(e.target.value));
    document.getElementById('fit').addEventListener('click', () => network && network.fit({{animation:true}}));
    document.getElementById('reset').addEventListener('click', () => {{ document.getElementById('filter').value=''; render(); }});
  </script>
</body>
</html>"""
    return html.encode("utf-8")


def _label_for_id(nodes: list[dict[str, Any]], node_id: str) -> str:
    for node in nodes:
        if node["id"] == node_id:
            return str(node["label"])
    return node_id
