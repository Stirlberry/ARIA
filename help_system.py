"""
ARIA Help System

Monitors agent performance and detects struggle.
When an agent is struggling, shows a diagnostic and optionally
fetches web resources for the operator. Observation only —
no parameters are changed and no sub-goals are injected.

Struggle triggers (after HELP_MIN_EPISODE):
  - Rolling average episode reward below HELP_REWARD_THRESH
  - Coordination rate below HELP_COORD_THRESH
"""

import os
import json
import urllib.request
import urllib.parse
import urllib.error
from collections import deque
from datetime import datetime

import config
from config import (
    HELP_WINDOW, HELP_REWARD_THRESH, HELP_COORD_THRESH,
    HELP_MIN_EPISODE, HELP_COOLDOWN, HELP_LOG_PATH,
)

SEARCH_QUERIES = {
    'low_reward':      'Q-learning reinforcement learning low reward convergence improvement',
    'no_coordination': 'multi-agent reinforcement learning coordination emergence strategies',
    'general':         'reinforcement learning agent survival improvement techniques'
}

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

    def deregister_agent(self, agent_id):
        self.reward_history.pop(agent_id, None)
        self.coord_history.pop(agent_id, None)
        self.last_help_ep.pop(agent_id, None)
        self.help_count.pop(agent_id, None)
        self._suppressed.discard(agent_id)

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
        print()
        print('  ' + '─' * 58)
        print(f'  DIAGNOSTICS — {agent_id}')
        print('  ' + '─' * 58)
        print(f'  Situation  : {reason.replace("_", " ")}')
        print(f'  Avg reward : {avg_reward:.2f}  (last {HELP_WINDOW} episodes)')
        print(f'  Coord rate : {coord_rate:.1%}')
        print()
        print(f'  Fetch web resources? [informational only — no parameters changed]')
        print()
        print('    1  Yes')
        print('    2  Yes — don\'t ask again')
        print('    3  No')
        print()

        while True:
            try:
                response = input('  Choice [1/2/3]: ').strip()
            except (EOFError, KeyboardInterrupt):
                print('  (No terminal input — skipping.)\n')
                return None, False
            if response == '1':
                print('  Fetching...\n')
                return True, False
            if response == '2':
                print('  Fetching. Won\'t ask again for this agent.\n')
                return True, True
            if response == '3':
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

    def run_help_cycle(self, agent_id, agent, reason, episode, config_module):
        """
        Diagnostic cycle: show agent status, optionally fetch web resources.
        Observation only — no parameters changed, no sub-goals injected.
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
            return False
        if dont_ask_again:
            self._suppressed.add(agent_id)
        if not granted:
            self._log({'event': 'diagnostic_declined', 'agent': agent_id,
                       'episode': episode, 'reason': reason})
            return False

        search_text, source = self.fetch_internet_help(reason)
        if search_text:
            print(f'\n  Resources from {source}:')
            print(f'  {search_text[:600]}')

        self.help_count[agent_id] = self.help_count.get(agent_id, 0) + 1

        self._log({
            'event':         'diagnostic_shown',
            'agent':         agent_id,
            'episode':       episode,
            'reason':        reason,
            'avg_reward':    round(avg_reward, 2),
            'coord_rate':    round(coord_rate, 4),
            'search_source': source,
        })

        print('  ' + '─' * 58 + '\n')
        return True

    def _log(self, record):
        record['timestamp'] = datetime.now().isoformat()
        with open(HELP_LOG_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def get_status(self, agent_id):
        count = self.help_count.get(agent_id, 0)
        last  = self.last_help_ep.get(agent_id, 0)
        if count == 0:
            return 'no help yet'
        return f'helped {count}x  last ep {last}'


class LexiconAdvisor:
    """Periodically prints a lexicon summary to the console. Observation only."""

    ADVICE_INTERVAL = 100
    MIN_EPISODE     = 200

    def __init__(self):
        self.last_advice_ep = 0

    def maybe_advise(self, channel, agents, episode, config_module):
        if not config.LEXICON_ADVISOR_ON:
            return
        if episode < self.MIN_EPISODE:
            return
        if episode - self.last_advice_ep < self.ADVICE_INTERVAL:
            return

        self.last_advice_ep = episode
        assigned  = channel.assigned_count()
        compounds = channel.compound_lexicon.crystallised_count()
        sequences = channel.sequence_lexicon.crystallised_count()
        roles_str = ', '.join(f'{aid}:{agents[aid].role}' for aid in agents)
        print(f'\n  [Lexicon] Ep {episode}: {assigned}/16 signals  '
              f'{compounds} compounds  {sequences} sequences  '
              f'total={channel.total_signals}  roles=[{roles_str}]')
