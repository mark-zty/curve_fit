from .base import AnalyticPanel


def build_dashboard(panels: list[tuple[str, AnalyticPanel]], **kwargs) -> str:
    sections = "\n".join(
        f"<section><h2>{title}</h2>{panel.render(**kwargs)}</section>"
        for title, panel in panels
    )
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Risk Dashboard</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    section {{ margin-bottom: 40px; }}
  </style>
</head>
<body>
  <h1>Risk Dashboard</h1>
  {sections}
</body>
</html>"""
