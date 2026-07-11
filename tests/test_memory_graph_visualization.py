from mental_state_bot.services.memory_graph_visualization import _render_html


def test_render_html_contains_interactive_graph_and_fallback_lists() -> None:
    html = _render_html(
        [
            {
                "id": "1",
                "label": "Проєкт",
                "kind": "project",
                "summary": "Творчий проєкт",
                "confidence": 0.8,
                "weight": 0.7,
                "status": "hypothesis",
                "aliases": [],
            },
            {
                "id": "2",
                "label": "Назва",
                "kind": "concept",
                "summary": "Назва <треку>",
                "confidence": 0.9,
                "weight": 0.5,
                "status": "candidate",
                "aliases": ["трек"],
            },
        ],
        [
            {
                "id": "edge",
                "from": "1",
                "to": "2",
                "label": "містить",
                "summary": "зв’язок із назвою треку",
                "confidence": 0.7,
                "weight": 0.6,
                "evidence_count": 2,
            }
        ],
    ).decode("utf-8")

    assert "vis-network" in html
    assert "Проєкт" in html
    assert "Назва" in html
    assert "містить" in html
    assert "Список вузлів" in html
    assert "\\u003c" in html


def test_render_html_handles_empty_graph() -> None:
    html = _render_html([], []).decode("utf-8")

    assert "У графі поки немає вузлів." in html
    assert "У графі поки немає зв’язків." in html
