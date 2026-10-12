"""
LLM orchestrator — OpenRouter decides which tools to run for the case study.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .openrouter_client import OpenRouterClient
from .tools import TOOL_SPECS, AgentState, get_tool_registry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous ML agent studying whether macro variables
improve prediction of company revenue growth.

Data quality matters: EDGAR fields are often missing for domain reasons
(e.g. banks may lack gross_profit). You MUST profile missingness and choose
explicit per-column actions before training.

You may ONLY act by calling tools. Reply with a single JSON object, no markdown:
{"tool": "<name>", "args": {..}, "reason": "<short why>"}

Available tools:
{tools}

Typical order:
1) load_panel
2) inspect_panel (optional)
3) profile_missingness
4) apply_data_quality_decisions  (args.decisions = [{column, action, reason}, ...])
5) run_bakeoff
6) compare_macro_lift
7) export_best_model
8) stop

Do not skip load_panel. Do not run_bakeoff before applying quality decisions
unless profile_missingness shows zero missing feature values.
When finished, call stop.
"""


class LLMOrchestrator:
    """Orchestrates the ML lifecycle using an LLM + local tools."""

    def __init__(self, client: Optional[OpenRouterClient] = None):
        self.client = client or OpenRouterClient()
        self.state = AgentState()

    def ping(self) -> str:
        return self.client.chat(
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=16,
        )

    def _system_prompt(self) -> str:
        tools_txt = json.dumps(TOOL_SPECS, indent=2)
        return SYSTEM_PROMPT.replace("{tools}", tools_txt)

    def _parse_tool_call(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        # Strip markdown fences if the model wraps JSON
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        # Find first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON tool call in model response: {text[:300]}")
        return json.loads(text[start : end + 1])

    def decide_next_tool(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        reply = self.client.chat(messages=messages, temperature=0.1, max_tokens=1200)
        logger.info("LLM raw reply: %s", reply[:800])
        return self._parse_tool_call(reply)

    def run_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        registry = get_tool_registry(self.state)
        if name not in registry:
            return {"ok": False, "error": f"Unknown tool: {name}. Allowed: {list(registry)}"}
        args = args or {}
        # Drop unexpected keys the model may invent
        try:
            return registry[name](**args)
        except TypeError as exc:
            return {"ok": False, "error": f"Bad args for {name}: {exc}", "args": args}

    def run_agent_loop(
        self,
        goal: str = (
            "Load the macro-firm panel, profile missing/ugly data, apply explicit "
            "per-column quality decisions with reasons, then run firm_only vs "
            "firm_plus_macro bake-off, compare whether macro helps, export, stop."
        ),
        max_steps: int = 10,
    ) -> AgentState:
        """
        LLM-first loop: decide tool → execute → feed result back → repeat.
        """
        self.state = AgentState()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": goal},
        ]

        for step in range(1, max_steps + 1):
            logger.info("=== Agent step %s/%s ===", step, max_steps)
            try:
                call = self.decide_next_tool(messages)
            except Exception as exc:
                logger.error("Failed to parse tool call: %s", exc)
                # Deterministic recovery path so the run still completes
                call = self._fallback_plan(step)

            tool = call.get("tool")
            args = call.get("args") or {}
            reason = call.get("reason", "")
            logger.info("Tool=%s reason=%s args=%s", tool, reason, args)

            result = self.run_tool(str(tool), args if isinstance(args, dict) else {})
            self.state.history.append({"step": step, "call": call, "result": result})

            messages.append({"role": "assistant", "content": json.dumps(call)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result:\n{json.dumps(result, default=str)[:4000]}\n"
                        "Choose the next tool as JSON."
                    ),
                }
            )

            if self.state.finished or (isinstance(result, dict) and result.get("stopped")):
                logger.info("Agent finished at step %s", step)
                break
        else:
            logger.warning("Hit max_steps without stop; forcing stop")
            self.run_tool("stop", {"reason": "max_steps"})

        return self.state

    def _fallback_plan(self, step: int) -> Dict[str, Any]:
        """If the LLM returns junk, still advance the real pipeline."""
        sequence = [
            {"tool": "load_panel", "args": {}, "reason": "fallback"},
            {"tool": "profile_missingness", "args": {}, "reason": "fallback"},
            {
                "tool": "apply_data_quality_decisions",
                "args": {
                    "decisions": [
                        {
                            "column": "gross_margin_lag1",
                            "action": "impute_median",
                            "reason": "fallback median for sparse margin",
                        }
                    ]
                },
                "reason": "fallback",
            },
            {"tool": "run_bakeoff", "args": {"metric": "rmse", "max_models": 4}, "reason": "fallback"},
            {"tool": "compare_macro_lift", "args": {"metric": "rmse"}, "reason": "fallback"},
            {"tool": "export_best_model", "args": {}, "reason": "fallback"},
            {"tool": "stop", "args": {"reason": "fallback complete"}, "reason": "fallback"},
        ]
        idx = min(step - 1, len(sequence) - 1)
        return sequence[idx]

    def plan_next_step(self, context: Dict[str, Any]) -> str:
        """Legacy free-text planner (kept for compatibility)."""
        return self.client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are an autonomous ML agent. Be brief.",
                },
                {"role": "user", "content": f"Context:\n{context}"},
            ]
        )
