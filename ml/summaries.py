"""Interfaces for daily/weekly natural-language summaries.

Later, this will call a small quantized on-device LLM or a server-side model.
"""

from typing import Dict


def summarize_daily(stats: Dict) -> str:
    # TODO: plug in LLM call / on-device inference.
    return (
        "Today you used your phone for {minutes} minutes with {sessions} sessions. "
        "Top app: {top_app}."
    ).format(
        minutes=stats.get("total_screen_time", 0),
        sessions=stats.get("session_count", 0),
        top_app=(stats.get("top_apps") or ["-"])[0],
    )
