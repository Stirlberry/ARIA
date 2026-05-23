"""
ARIA Help System — Phase 2

Monitors agent performance and detects struggle.
When an agent is struggling, requests human permission to access
the internet for strategies. Fetches search results, optionally
analyses them via Claude API, applies parameter adjustments, and
assigns a Claude-generated sub-goal.

Struggle triggers (after HELP_MIN_EPISODE):
  - Rolling average episode reward below HELP_REWARD_THRESH
  - Coordination rate below HELP_COORD_THRESH
"""

import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error
from collections import deque
from datetime import datetime

import config
from config import (
    HELP_WINDOW, HELP_REWARD_THRESH, HELP_COORD_THRESH,
    HELP_MIN_EPISODE, HELP_COOLDOWN, HELP_LOG_PATH, SUBGOAL_LOG_PATH,
)
from budget import cost_tracker

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')


def _extract_json(text):
    """
    Robustly extract the first JSON object from a Claude response.
    Handles cases where Claude adds explanation text before or after the JSON.
    """
    text = text.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} block and parse that
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

SEARCH_QUERIES = {
    'low_reward':      'Q-learning reinforcement learning low reward convergence improvement',
    'no_coordination': 'multi-agent reinforcement learning coordination emergence strategies',
    'general':         'reinforcement learning agent survival improvement techniques'
}

ADJUSTABLE_PARAMS = {
    'REWARD_COORD':    (5.0,  100.0),
    'LEARNING_RATE':   (0.01,   0.5),
    'EPSILON_DECAY':   (0.990, 0.999),
    'REWARD_CURRENCY': (5.0,  50.0)
}

_PARAM_TO_AGENT_ATTR = {
    'LEARNING_RATE': 'learning_rate',
    'EPSILON_DECAY': 'epsilon_decay',
}

_SUB_GOAL_TEMPLATES = (
    'face_coord', 'partner_contact', 'currency_dir',
    'coord_nearby', 'partner_close', 'spec'
)

_VALID_SPEC_CONDITIONS = frozenset({
    'coord_close', 'coord_near', 'currency_close', 'partner_close',
    'face_coord', 'face_currency', 'face_partner', 'avoid_partner',
})


class HelpMonitor:
    def __init__(self):
        self.reward_history   = {}
        self.coord_history    = {}
        self.last_help_ep     = {}
        self.help_count       = {}
        self._suppressed = set()   # agent IDs that chose "don't ask again"
        os.makedirs(os.path.dirname(HELP_LOG_PATH), exist_ok=True)

    def register_agent(self, agent_id):
        self.reward_history[agent_id] = deque(maxlen=HELP_WINDOW)
        self.coord_history[agent_id]  = deque(maxlen=HELP_WINDOW)
        self.last_help_ep[agent_id]   = 0
        self.help_count[agent_id]     = 0

    def record_episode(self, agent_id, ep_reward, coord_achieved):
        if agent_id not in self.reward_history:
            self.register_agent(agent_id)
        self.reward_history[agent_id].append(ep_reward)
        self.coord_history[agent_id].append(1 if coord_achieved else 0)

    def check_struggle(self, agent_id, episode):
        if episode < HELP_MIN_EPISODE:
            return False, ''
        if episode - self.last_help_ep.get(agent_id, 0) < HELP_COOLDOWN:
            return False, ''

        rewards = self.reward_history.get(agent_id, deque())
        coords  = self.coord_history.get(agent_id, deque())

        if len(rewards) < HELP_WINDOW:
            return False, ''

        avg_reward = sum(rewards) / len(rewards)
        coord_rate = sum(coords) / len(coords)

        if avg_reward < HELP_REWARD_THRESH and coord_rate < HELP_COORD_THRESH:
            return True, 'low_reward+no_coordination'
        if coord_rate < HELP_COORD_THRESH:
            return True, 'no_coordination'
        if avg_reward < HELP_REWARD_THRESH:
            return True, 'low_reward'

        return False, ''

    def check_all(self, agents, episode):
        struggling = []
        for agent_id in agents:
            if agent_id in self._suppressed:
                continue
            is_struggling, reason = self.check_struggle(agent_id, episode)
            if is_struggling:
                struggling.append((agent_id, reason))
        return struggling

    def request_permission(self, agent_id, reason, avg_reward, coord_rate):
        want = SEARCH_QUERIES.get(reason.split('+')[0], SEARCH_QUERIES['general'])

        print()
        print('  ' + '─' * 58)
        print(f'  HELP REQUEST — {agent_id}')
        print('  ' + '─' * 58)
        print(f'  Situation  : {reason.replace("_", " ")}')
        print(f'  Avg reward : {avg_reward:.2f}  (last {HELP_WINDOW} episodes)')
        print(f'  Coord rate : {coord_rate:.1%}')
        print()
        print(f'  {agent_id} wants to search the internet for:')
        print(f'    "{want}"')
        print(f'  and apply the findings to its own parameters.')
        print()
        print('    1  Yes')
        print('    2  Yes — don\'t ask again')
        print('    3  No')
        print()

        while True:
            try:
                response = input('  Choice [1/2/3]: ').strip()
            except (EOFError, KeyboardInterrupt):
                print('  (No terminal input — skipping help this episode.)\n')
                return None, False   # None = skipped, not denied
            if response == '1':
                print('  Permission granted.\n')
                return True, False
            if response == '2':
                print('  Permission granted. Won\'t ask again for this agent.\n')
                return True, True
            if response == '3':
                print('  Permission denied. Agent can ask again later.\n')
                return False, False
            print('  Please enter 1, 2, or 3.')

    def fetch_internet_help(self, struggle_type):
        query = SEARCH_QUERIES.get(struggle_type.split('+')[0], SEARCH_QUERIES['general'])
        url   = (f'https://api.duckduckgo.com/?q={urllib.parse.quote(query)}'
                 f'&format=json&no_redirect=1&no_html=1&skip_disambig=1')

        print(f'  Searching: "{query}"')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ARIA/1.0'})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode('utf-8'))

            text = (data.get('AbstractText') or
                    data.get('Answer') or
                    '; '.join(r.get('Text', '') for r in data.get('RelatedTopics', [])[:3]))

            source = data.get('AbstractSource', 'DuckDuckGo')
            if text:
                print(f'  Fetched {len(text)} chars from {source}.')
                return text[:2000], source
            print('  No useful result from DuckDuckGo.')
            return '', ''

        except urllib.error.URLError as e:
            print(f'  Internet fetch failed: {e.reason}')
            return '', ''
        except Exception as e:
            print(f'  Internet fetch error: {e}')
            return '', ''

    def analyse_with_claude(self, agent_id, struggle_type, avg_reward,
                            coord_rate, search_text, episode):
        if not ANTHROPIC_API_KEY:
            return {}
        if cost_tracker.over_budget():
            print(f'  [Budget] Cap reached ({cost_tracker.session_str()}) — using rule-based fallback.')
            return {}

        prompt = f"""You are analysing a DQN agent called {agent_id} in a multi-agent grid world.

Agent state:
- Episode: {episode}
- Struggle type: {struggle_type}
- Average episode reward (last {HELP_WINDOW} episodes): {avg_reward:.2f}
- Coordination success rate: {coord_rate:.1%}

Relevant research context:
{search_text or "No search results available."}

Adjustable parameters and their ranges:
- REWARD_COORD: coordination reward (range 5.0-100.0)
- LEARNING_RATE: Adam learning rate (range 0.01-0.5)
- EPSILON_DECAY: exploration decay per episode (range 0.990-0.999)
- REWARD_CURRENCY: solo currency reward (range 5.0-50.0)

Respond ONLY with a JSON object of parameter adjustments. Only include parameters that genuinely need changing.
Example: {{"REWARD_COORD": 45.0, "LEARNING_RATE": 0.15}}
If no changes are warranted: {{}}"""

        try:
            import urllib.request as ur
            payload = json.dumps({
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 256,
                'messages': [{'role': 'user', 'content': prompt}]
            }).encode()

            req = ur.Request(
                'https://api.anthropic.com/v1/messages',
                data=payload,
                headers={
                    'Content-Type':      'application/json',
                    'x-api-key':         ANTHROPIC_API_KEY,
                    'anthropic-version': '2023-06-01'
                }
            )
            with ur.urlopen(req, timeout=15) as r:
                result = json.loads(r.read().decode())

            usage = result.get('usage', {})
            cost_tracker.record(usage.get('input_tokens', 0),
                                usage.get('output_tokens', 0), 'help_analysis')
            print(f'  [Budget] {cost_tracker.session_str()}')

            text = result['content'][0]['text'].strip()
            text = text.replace('```json', '').replace('```', '').strip()
            suggestions = _extract_json(text)

            validated = {}
            for param, value in suggestions.items():
                if param in ADJUSTABLE_PARAMS:
                    lo, hi = ADJUSTABLE_PARAMS[param]
                    validated[param] = max(lo, min(hi, float(value)))
            return validated

        except Exception as e:
            print(f'  Claude analysis failed: {e}')
            return {}

    def generate_subgoal(self, agent_id, agent, episode):
        """
        Ask Claude (Haiku) to design a spec-based sub-goal for the agent's role.
        Returns a SubGoal object, falling back to face_coord on any error.
        """
        if not ANTHROPIC_API_KEY:
            return None
        if cost_tracker.over_budget():
            print(f'  [Budget] Cap reached — skipping sub-goal generation.')
            return None

        from agent import SubGoal

        prompt = (
            f"An ARIA agent called {agent_id} (role: {agent.role}) in a multi-agent "
            f"grid world is struggling at episode {episode} "
            f"with total reward {agent.total_reward:.1f}.\n\n"
            f"State layout: s[2]=currency_dir, s[3]=coord_dir, s[4]=partner_dir "
            f"(8=absent); s[5]=currency_dist_bin, s[6]=coord_dist_bin, "
            f"s[7]=partner_dist_bin (0=close≤3, 1=mid 4-8, 2=far >8).\n\n"
            f"Available conditions:\n"
            f"- coord_close:    coord dist bin == 0 (very close)\n"
            f"- coord_near:     coord dist bin <= 1\n"
            f"- currency_close: currency dist bin == 0\n"
            f"- partner_close:  partner dist bin == 0\n"
            f"- face_coord:     coord direction visible\n"
            f"- face_currency:  currency direction visible\n"
            f"- face_partner:   partner direction visible\n"
            f"- avoid_partner:  partner dist bin == 2 (useful for explorers)\n\n"
            f"Design the best sub-goal for a '{agent.role}' agent. "
            f"Respond with ONLY a JSON spec:\n"
            f'{{\"condition\": \"<name>\", \"bonus\": <0.1-0.5>, \"negate\": false}}'
        )

        try:
            import urllib.request as ur
            payload = json.dumps({
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 64,
                'messages': [{'role': 'user', 'content': prompt}]
            }).encode()

            req = ur.Request(
                'https://api.anthropic.com/v1/messages',
                data=payload,
                headers={
                    'Content-Type':      'application/json',
                    'x-api-key':         ANTHROPIC_API_KEY,
                    'anthropic-version': '2023-06-01'
                }
            )
            with ur.urlopen(req, timeout=10) as r:
                result = json.loads(r.read().decode())

            usage = result.get('usage', {})
            cost_tracker.record(usage.get('input_tokens', 0),
                                usage.get('output_tokens', 0), 'subgoal')

            text = result['content'][0]['text'].strip()
            text = text.replace('```json', '').replace('```', '').strip()
            spec = _extract_json(text)

            if spec.get('condition') not in _VALID_SPEC_CONDITIONS:
                spec['condition'] = 'face_coord'
            spec['bonus']  = max(0.1, min(0.5, float(spec.get('bonus', 0.3))))
            spec['negate'] = bool(spec.get('negate', False))

            sg = SubGoal('spec', spec)
            label = f'spec({spec["condition"]}, bonus={spec["bonus"]}, negate={spec["negate"]})'
            print(f'  Sub-goal assigned: {label}')
            self._log_subgoal(agent_id, f'spec:{spec["condition"]}', episode)
            return sg

        except Exception as e:
            print(f'  Sub-goal generation failed: {e} — falling back to face_coord')
            sg = SubGoal('face_coord')
            self._log_subgoal(agent_id, 'face_coord', episode)
            return sg

    def _fallback_suggestions(self, struggle_type):
        if 'no_coordination' in struggle_type:
            return {'REWARD_COORD': 40.0}
        if 'low_reward' in struggle_type:
            return {'LEARNING_RATE': 0.15, 'EPSILON_DECAY': 0.993}
        return {'REWARD_COORD': 40.0, 'LEARNING_RATE': 0.12}

    def apply_adjustments(self, agent, suggestions, config_module):
        applied = {}
        for param, value in suggestions.items():
            old = getattr(config_module, param, None)
            if old is not None and old != value:
                setattr(config_module, param, value)
                attr = _PARAM_TO_AGENT_ATTR.get(param)
                if attr and hasattr(agent, attr):
                    setattr(agent, attr, value)
                    if attr == 'learning_rate' and hasattr(agent, 'rebuild_optimizer'):
                        agent.rebuild_optimizer()
                applied[param] = {'from': round(old, 4), 'to': round(value, 4)}

        if applied:
            print('  Parameters adjusted:')
            for p, v in applied.items():
                print(f'    {p}: {v["from"]} -> {v["to"]}')
        else:
            print('  No parameter adjustments applied.')

        return applied

    def run_help_cycle(self, agent_id, agent, reason, episode, config_module):
        """
        Full help cycle: diagnostics → permission → fetch → analyse → adjust → sub-goal.
        """
        rewards    = self.reward_history.get(agent_id, deque())
        coords     = self.coord_history.get(agent_id, deque())
        avg_reward = sum(rewards) / max(len(rewards), 1)
        coord_rate = sum(coords) / max(len(coords), 1)

        self.last_help_ep[agent_id] = episode

        granted, dont_ask_again = self.request_permission(
            agent_id, reason, avg_reward, coord_rate
        )
        if granted is None:
            # No terminal available — skip silently, don't log as denial
            return False
        if dont_ask_again:
            self._suppressed.add(agent_id)
        if not granted:
            self._log({'event': 'help_denied', 'agent': agent_id,
                       'episode': episode, 'reason': reason})
            return False

        search_text, source = self.fetch_internet_help(reason)

        suggestions = {}
        if ANTHROPIC_API_KEY:
            print('  Analysing with Claude...')
            suggestions = self.analyse_with_claude(
                agent_id, reason, avg_reward, coord_rate, search_text, episode
            )
            if not suggestions:
                print('  Claude returned no suggestions. Using rule-based fallback.')
                suggestions = self._fallback_suggestions(reason)
        else:
            print('  (No ANTHROPIC_API_KEY — using rule-based suggestions.)')
            suggestions = self._fallback_suggestions(reason)
            if search_text:
                print(f'\n  Search result excerpt:\n  {search_text[:400]}...\n')

        applied = self.apply_adjustments(agent, suggestions, config_module)

        # Assign Claude-generated sub-goal if possible
        sub_goal = self.generate_subgoal(agent_id, agent, episode)
        if sub_goal is not None:
            agent.sub_goal = sub_goal
            remaining = sub_goal.duration - sub_goal.active_steps
            print(f'  Sub-goal active: {sub_goal.template} ({remaining} steps)')

        self.help_count[agent_id] = self.help_count.get(agent_id, 0) + 1

        self._log({
            'event':           'help_applied',
            'agent':           agent_id,
            'episode':         episode,
            'reason':          reason,
            'avg_reward':      round(avg_reward, 2),
            'coord_rate':      round(coord_rate, 4),
            'search_source':   source,
            'suggestions':     suggestions,
            'applied':         applied,
            'sub_goal':        sub_goal.template if sub_goal else None,
            'claude_used':     bool(ANTHROPIC_API_KEY),
            'never_ask_again': dont_ask_again
        })

        print('  ' + '─' * 58 + '\n')
        return True

    def _log(self, record):
        record['timestamp'] = datetime.now().isoformat()
        with open(HELP_LOG_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def _log_subgoal(self, agent_id, name, episode):
        os.makedirs(os.path.dirname(SUBGOAL_LOG_PATH), exist_ok=True)
        record = {
            'event':     'subgoal_assigned',
            'agent':     agent_id,
            'name':      name,
            'episode':   episode,
            'timestamp': datetime.now().isoformat()
        }
        with open(SUBGOAL_LOG_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def get_status(self, agent_id):
        count = self.help_count.get(agent_id, 0)
        last  = self.last_help_ep.get(agent_id, 0)
        if count == 0:
            return 'no help yet'
        return f'helped {count}x  last ep {last}'


class LexiconAdvisor:
    """
    Periodically asks Claude to analyse the emerging communication lexicon and
    tune SIGNAL_REWARD / SIGNAL_WINDOW in the live config module.
    """

    ADVICE_INTERVAL = 100
    MIN_EPISODE     = 200

    def __init__(self):
        self.last_advice_ep = 0

    def maybe_advise(self, channel, agents, episode, config_module):
        if not config.LEXICON_ADVISOR_ON:
            return
        if not ANTHROPIC_API_KEY:
            return
        if cost_tracker.over_budget():
            return
        if episode < self.MIN_EPISODE:
            return
        if episode - self.last_advice_ep < self.ADVICE_INTERVAL:
            return

        lexicon_lines = []
        for sig_idx, entry in channel.lexicon.items():
            status = entry.symbol if entry.assigned else 'unassigned'
            lexicon_lines.append(
                f"  Signal {sig_idx}: {status}  "
                f"uses={entry.use_count}  coord_successes={entry.coord_successes}"
            )

        compound_lines = []
        for (a, b), e in channel.compound_lexicon.entries.items():
            if e.crystallised:
                compound_lines.append(f"  ({a},{b})={e.symbol}  uses={e.use_count}")

        roles_str = ', '.join(f'{aid}: {agents[aid].role}' for aid in agents)

        prompt = (
            f"ARIA multi-agent simulation, episode {episode}.\n\n"
            f"Emergent communication lexicon:\n"
            + '\n'.join(lexicon_lines)
            + f"\n\nCrystallised compound pairs:\n"
            + ('\n'.join(compound_lines) if compound_lines else '  none')
            + f"\n\nTotal signals sent: {channel.total_signals}\n"
            f"Agent roles: {roles_str}\n"
            f"Current SIGNAL_REWARD: {config_module.SIGNAL_REWARD}\n"
            f"Current SIGNAL_WINDOW: {config_module.SIGNAL_WINDOW}\n\n"
            f"Are agents using signals effectively for coordination? "
            f"If signal use is low relative to episode count, increase SIGNAL_REWARD. "
            f"If the timing window is too tight, increase SIGNAL_WINDOW.\n"
            f"Respond ONLY with JSON: "
            f'{{\"SIGNAL_REWARD\": float, \"SIGNAL_WINDOW\": int}}\n'
            f"Only include parameters that need changing. If none needed: {{}}"
        )

        try:
            import urllib.request as ur
            payload = json.dumps({
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 64,
                'messages': [{'role': 'user', 'content': prompt}]
            }).encode()

            req = ur.Request(
                'https://api.anthropic.com/v1/messages',
                data=payload,
                headers={
                    'Content-Type':      'application/json',
                    'x-api-key':         ANTHROPIC_API_KEY,
                    'anthropic-version': '2023-06-01'
                }
            )
            with ur.urlopen(req, timeout=15) as r:
                result = json.loads(r.read().decode())

            usage = result.get('usage', {})
            cost_tracker.record(usage.get('input_tokens', 0),
                                usage.get('output_tokens', 0), 'lexicon_advisor')

            text = result['content'][0]['text'].strip()
            text = text.replace('```json', '').replace('```', '').strip()
            adjustments = _extract_json(text)

            applied = {}
            if 'SIGNAL_REWARD' in adjustments:
                val = max(1.0, min(20.0, float(adjustments['SIGNAL_REWARD'])))
                if val != config_module.SIGNAL_REWARD:
                    applied['SIGNAL_REWARD'] = (config_module.SIGNAL_REWARD, val)
                    config_module.SIGNAL_REWARD = val
            if 'SIGNAL_WINDOW' in adjustments:
                val = max(2, min(15, int(adjustments['SIGNAL_WINDOW'])))
                if val != config_module.SIGNAL_WINDOW:
                    applied['SIGNAL_WINDOW'] = (config_module.SIGNAL_WINDOW, val)
                    config_module.SIGNAL_WINDOW = val

            self.last_advice_ep = episode

            if applied:
                print(f'\n  [Lexicon Advisor] Episode {episode}:')
                for p, (old, new) in applied.items():
                    print(f'    {p}: {old} → {new}')
            else:
                print(f'\n  [Lexicon Advisor] Episode {episode}: lexicon OK, no changes.')

        except Exception as e:
            print(f'  [Lexicon Advisor] Analysis failed: {e}')
