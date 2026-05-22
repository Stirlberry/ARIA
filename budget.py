"""
ARIA Budget Tracker
Tracks Claude API spending per session and lifetime.
Enforces a configurable session cap — when reached, Claude calls are
skipped and rule-based fallbacks apply automatically.

Costs are based on actual token counts returned by the API response,
not estimates. Lifetime totals persist in logs/budget.json across runs.
"""

import os
import json
from datetime import datetime
from config import API_BUDGET_CAP, HAIKU_INPUT_COST, HAIKU_OUTPUT_COST, BUDGET_LOG_PATH


class CostTracker:

    def __init__(self):
        self.cap_usd          = API_BUDGET_CAP
        self._session_input   = 0
        self._session_output  = 0
        self._session_calls   = 0
        self._data            = self._load()

    # ── Recording ──────────────────────────────────────────────────────────────

    def record(self, input_tokens, output_tokens, call_type=''):
        self._session_input  += input_tokens
        self._session_output += output_tokens
        self._session_calls  += 1

        self._data['input_tokens']  += input_tokens
        self._data['output_tokens'] += output_tokens
        self._data['total_calls']   += 1
        self._data['lifetime_cost']  = round(
            self._calc(self._data['input_tokens'], self._data['output_tokens']), 6
        )
        if call_type:
            self._data['by_type'][call_type] = self._data['by_type'].get(call_type, 0) + 1
        self._save()

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def session_cost(self):
        return self._calc(self._session_input, self._session_output)

    @property
    def lifetime_cost(self):
        return self._data['lifetime_cost']

    @property
    def session_calls(self):
        return self._session_calls

    @property
    def lifetime_calls(self):
        return self._data['total_calls']

    def over_budget(self):
        return self.session_cost >= self.cap_usd

    def budget_remaining(self):
        return max(0.0, self.cap_usd - self.session_cost)

    # ── Display ────────────────────────────────────────────────────────────────

    def session_str(self):
        pct = (self.session_cost / self.cap_usd * 100) if self.cap_usd > 0 else 0
        return (f'${self.session_cost:.4f} of ${self.cap_usd:.2f} cap '
                f'({pct:.1f}%)  calls {self._session_calls}')

    def lifetime_str(self):
        return (f'${self.lifetime_cost:.4f} lifetime  '
                f'{self.lifetime_calls} total calls')

    # ── Persistence ────────────────────────────────────────────────────────────

    def _calc(self, inp, out):
        return (inp / 1_000_000 * HAIKU_INPUT_COST +
                out / 1_000_000 * HAIKU_OUTPUT_COST)

    def _load(self):
        os.makedirs(os.path.dirname(BUDGET_LOG_PATH), exist_ok=True)
        if os.path.exists(BUDGET_LOG_PATH):
            try:
                with open(BUDGET_LOG_PATH) as f:
                    data = json.load(f)
                data.setdefault('by_type', {})
                return data
            except Exception:
                pass
        return {'input_tokens': 0, 'output_tokens': 0,
                'total_calls': 0, 'lifetime_cost': 0.0,
                'by_type': {}}

    def _save(self):
        self._data['last_updated'] = datetime.now().isoformat()
        with open(BUDGET_LOG_PATH, 'w') as f:
            json.dump(self._data, f, indent=2)


# Singleton — import this directly everywhere
cost_tracker = CostTracker()
