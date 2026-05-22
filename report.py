"""
ARIA Progress Report
Run: python report.py
Reads saved state and logs — ARIA does not need to be running.
"""

import json
import os
import sys
from datetime import datetime

SAVE_DIR   = 'logs/saves'
LEX_LOG    = 'logs/lexicon.jsonl'
GOALS_LOG  = 'logs/discovered_goals.jsonl'
BUDGET_LOG = 'logs/budget.json'

W = 72


def _rule():
    print('=' * W)


def _section(title):
    print()
    print(f'  {title}')
    print('  ' + '─' * (W - 4))


def _load_latest_save():
    ptr = os.path.join(SAVE_DIR, 'latest.json')
    if not os.path.exists(ptr):
        return None
    with open(ptr) as f:
        ref = json.load(f)
    json_path = ref.get('json')
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_budget():
    if not os.path.exists(BUDGET_LOG):
        return {}
    with open(BUDGET_LOG) as f:
        return json.load(f)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    meta = _load_latest_save()
    if meta is None:
        print('No save file found. Run ARIA first.')
        sys.exit(1)

    lex_records   = _load_jsonl(LEX_LOG)
    goals_records = _load_jsonl(GOALS_LOG)
    budget        = _load_budget()
    now           = datetime.now().strftime('%Y-%m-%d %H:%M')

    _rule()
    print('  ARIA — Progress Report')
    print(f'  Generated : {now}')
    _rule()

    # ── Snapshot ──────────────────────────────────────────────────────────────
    _section('SNAPSHOT')
    episode    = meta['episode']
    generation = meta['generation']
    lineage    = ' > '.join(meta['all_ids_ever'])
    print(f'  Episode    : {episode:,}')
    print(f'  Generation : {generation}')
    print(f'  Saved      : {meta["timestamp"][:19]}')
    print(f'  Lineage    : {lineage}')

    # ── Active agents ─────────────────────────────────────────────────────────
    _section('ACTIVE AGENTS')
    print(f'  {"Agent":<16} {"Episodes":>10} {"Total Reward":>14} {"Epsilon":>9}')
    print(f'  {"─"*16} {"─"*10} {"─"*14} {"─"*9}')
    for agent_id in meta['active_agents']:
        am      = meta['agent_meta'].get(agent_id, {})
        eps_c   = am.get('episodes', 0)
        total_r = am.get('total_reward', 0.0)
        epsilon = am.get('epsilon', 0.0)
        print(f'  {agent_id:<16} {eps_c:>10,} {total_r:>14,.1f} {epsilon:>9.4f}')

    # ── Lexicon ───────────────────────────────────────────────────────────────
    _section('LEXICON')
    ch            = meta.get('channel', {})
    total_signals = ch.get('total_signals', 0)
    lex           = ch.get('lexicon', {})
    compounds     = ch.get('compound_lexicon', [])
    sequences     = ch.get('sequence_lexicon', [])
    n_comp        = sum(1 for e in compounds if e.get('crystallised'))
    n_seq         = sum(1 for e in sequences if e.get('crystallised'))

    print(f'  Total signals broadcast : {total_signals:,}')
    print()
    print(f'  {"Sig":>5} {"Symbol":>8} {"Uses":>10} {"Coord":>8}')
    print(f'  {"─"*5} {"─"*8} {"─"*10} {"─"*8}')
    for sig_str, ed in sorted(lex.items(), key=lambda x: int(x[0])):
        sym   = ed.get('symbol', '[?]')
        uses  = ed.get('use_count', 0)
        coord = ed.get('coord_successes', 0)
        print(f'  {sig_str:>5} {sym:>8} {uses:>10,} {coord:>8,}')
    print()
    print(f'  Compound symbols crystallised : {n_comp}')
    print(f'  Sequence symbols crystallised : {n_seq}')

    # ── Events this session ───────────────────────────────────────────────────
    _section('EVENTS THIS SESSION')
    repls = [r for r in lex_records if r.get('event') == 'replication']
    if repls:
        for r in repls:
            ep      = r.get('episode', '?')
            retired = r.get('retired_id', '?')
            child   = r.get('child_id', '?')
            ret_r   = r.get('retired_total_reward', 0)
            print(f'  Ep {ep:>5} | retired: {retired:<12} (reward {ret_r:>10,.1f}) | born: {child}')
    else:
        print('  No replications this session.')

    # ── Goals discovered ──────────────────────────────────────────────────────
    _section('GOALS DISCOVERED (ALL TIME)')
    if goals_records:
        shown = goals_records[-10:]
        print(f'  {"Agent":<16} {"Episode":>8} {"Condition":<22} {"Lift":>6}')
        print(f'  {"─"*16} {"─"*8} {"─"*22} {"─"*6}')
        for g in shown:
            agent = g.get('agent', '?')
            ep    = g.get('episode', 0)
            cond  = g.get('condition', '?')
            lift  = g.get('lift', 0.0)
            print(f'  {agent:<16} {ep:>8,} {cond:<22} {lift:>6.3f}')
        if len(goals_records) > 10:
            print(f'  ... and {len(goals_records) - 10} earlier discoveries')
    else:
        print('  None yet.')

    # ── API budget ────────────────────────────────────────────────────────────
    _section('API BUDGET')
    session_cost  = budget.get('session_cost', 0.0)
    lifetime_cost = budget.get('lifetime_cost', 0.0)
    print(f'  Session spend  : ${session_cost:.4f}')
    print(f'  Lifetime spend : ${lifetime_cost:.4f}')

    _rule()
    print()


if __name__ == '__main__':
    main()
