from __future__ import annotations

import json
from typing import Any

from pocket_genie.registry.tools import ToolRegistry


def register_visualization_tools(registry: ToolRegistry) -> None:
    @registry.register(
        name="viz.build_chart_spec",
        description="Build a structured chart specification for rendering in the mobile web console.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Chart title."},
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter"],
                    "description": "Chart visualization type.",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Category or x-axis labels.",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numeric series values.",
                },
            },
            "required": ["title", "labels", "values"],
        },
        domain="visualization",
        mutating=False,
    )
    def build_chart_spec(
        title: str,
        labels: list[str],
        values: list[float],
        chart_type: str = "bar",
    ) -> str:
        spec: dict[str, Any] = {
            "widget": "pocket_genie_chart",
            "title": title,
            "chart_type": chart_type,
            "series": [
                {"label": str(lbl), "value": float(val)}
                for lbl, val in zip(labels, values)
            ],
        }
        return json.dumps(spec, indent=2)
