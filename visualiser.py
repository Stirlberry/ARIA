"""
ARIA Visualiser — Phase 2
Renders every step at training speed (no FPS cap).
"""

import os
import sys
import pygame
from collections import deque
from config import GRID_SIZE, CELL_SIZE

GRAPH_H = 150   # height of reward trend strip below the grid

BG           = (12,  12,  18)
GRID_CELL    = (22,  22,  32)
GRID_LINE    = (40,  40,  55)
PANEL_BG     = (16,  16,  26)
PANEL_LINE   = (50,  50,  70)
WHITE        = (230, 230, 230)
MUTED        = (110, 110, 140)
GOLD         = (255, 220, 80)
REPL_COL     = (100, 230, 160)
CURRENCY_COL = (70,  210, 110)
COORD_COL    = (200, 90,  210)
COMPOUND_COL = (255, 160, 80)
PANEL_W      = 320

AGENT_COLOURS = {
    'ARIA-CAFE': (90,  170, 255),
    'ARIA-BABE': (255, 140, 70),
    'ARIA-DEAD': (180, 110, 255),
    'ARIA-BEEF': (255, 100, 140),
    'ARIA-FAFE': (80,  210, 200),
    'ARIA-FADE': (255, 200, 60),
    'ARIA-FEED': (160, 230, 100),
}

ROLE_COL = {
    'coordinator': (100, 200, 255),
    'forager':     (255, 200,  80),
    'generalist':  (160, 230, 100),
    'exploring':   (130, 130, 160),
}


def _agent_colour(agent_id):
    if agent_id in AGENT_COLOURS:
        return AGENT_COLOURS[agent_id]
    h = hash(agent_id) & 0xFFFFFF
    return (max(80, (h >> 16) & 0xFF), max(80, (h >> 8) & 0xFF), max(80, h & 0xFF))


class Visualiser:
    def __init__(self):
        pygame.init()
        self.cell    = CELL_SIZE
        grid_px      = GRID_SIZE * self.cell
        self.width   = grid_px + PANEL_W
        self.height  = GRID_SIZE * self.cell
        self.screen  = pygame.display.set_mode((self.width, self.height + GRAPH_H))
        self.reward_history = {}   # agent_id -> deque of episode rewards
        pygame.display.set_caption('ARIA — Adaptive Reasoning and Interaction Agent')
        self.font_lg = pygame.font.SysFont('monospace', 17, bold=True)
        self.font_md = pygame.font.SysFont('monospace', 13)
        self.font_sm = pygame.font.SysFont('monospace', 11)
        self.signal_history      = []
        self.replication_flash   = 0
        self.last_replication    = None
        self._save_screenshot    = False

        self.screen.fill(BG)
        cx = self.width // 2
        t1 = self.font_lg.render('ARIA', True, WHITE)
        t2 = self.font_md.render('Adaptive Reasoning and Interaction Agent', True, MUTED)
        t3 = self.font_md.render('Initialising — training will begin shortly...', True, MUTED)
        self.screen.blit(t1, (cx - t1.get_width() // 2, self.height // 2 - 50))
        self.screen.blit(t2, (cx - t2.get_width() // 2, self.height // 2 - 10))
        self.screen.blit(t3, (cx - t3.get_width() // 2, self.height // 2 + 20))
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit()
                if event.key == pygame.K_s:
                    self._save_screenshot = True

    def _quit(self):
        pygame.quit()
        raise KeyboardInterrupt

    def add_signal(self, sender_id, symbol):
        self.signal_history.append((sender_id, symbol))
        if len(self.signal_history) > 10:
            self.signal_history.pop(0)

    def notify_replication(self, summary):
        self.last_replication  = summary
        self.replication_flash = 90

    def _draw_grid(self, env):
        grid_px = GRID_SIZE * self.cell
        currency_nodes, coord_nodes = env.get_nodes()
        positions = env.get_positions()

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                pygame.draw.rect(self.screen, GRID_CELL,
                                 pygame.Rect(x*self.cell, y*self.cell,
                                             self.cell-1, self.cell-1))

        for (cx, cy) in currency_nodes:
            pad = 10
            pygame.draw.rect(self.screen, CURRENCY_COL,
                             pygame.Rect(cx*self.cell+pad, cy*self.cell+pad,
                                         self.cell-pad*2, self.cell-pad*2),
                             border_radius=3)
            t = self.font_sm.render('$', True, BG)
            self.screen.blit(t, (cx*self.cell+self.cell//2-4,
                                  cy*self.cell+self.cell//2-6))

        for (cx, cy) in coord_nodes:
            pad = 10
            pygame.draw.rect(self.screen, COORD_COL,
                             pygame.Rect(cx*self.cell+pad, cy*self.cell+pad,
                                         self.cell-pad*2, self.cell-pad*2),
                             border_radius=3)
            t = self.font_sm.render('CO', True, BG)
            self.screen.blit(t, (cx*self.cell+self.cell//2-8,
                                  cy*self.cell+self.cell//2-6))

        for agent_id, (ax, ay) in positions.items():
            col = _agent_colour(agent_id)
            tag = agent_id.split('-')[1]
            cx  = ax*self.cell + self.cell//2
            cy  = ay*self.cell + self.cell//2
            pygame.draw.circle(self.screen, col, (cx, cy), self.cell//2 - 5)
            lbl = self.font_sm.render(tag, True, BG)
            self.screen.blit(lbl, (cx - len(tag)*3, cy - 6))

        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (i*self.cell, 0), (i*self.cell, grid_px), 1)
            pygame.draw.line(self.screen, GRID_LINE,
                             (0, i*self.cell), (grid_px, i*self.cell), 1)

    def _draw_panel(self, agents, channel, episode, step,
                    ep_rewards, generation):
        grid_px = GRID_SIZE * self.cell
        pygame.draw.rect(self.screen, PANEL_BG,
                         pygame.Rect(grid_px, 0, PANEL_W, self.height))
        pygame.draw.line(self.screen, PANEL_LINE,
                         (grid_px, 0), (grid_px, self.height), 2)

        px = grid_px + 14
        py = 14

        def blit(s, col, size='md'):
            nonlocal py
            f = self.font_lg if size == 'lg' else (
                self.font_md if size == 'md' else self.font_sm)
            surf = f.render(s, True, col)
            if px + surf.get_width() > self.width - 4:
                chars = max(1, int(len(s) * (self.width - px - 8) / surf.get_width()))
                surf  = f.render(s[:chars], True, col)
            self.screen.blit(surf, (px, py))
            py += 24 if size == 'lg' else (20 if size == 'md' else 16)

        def divider():
            nonlocal py
            py += 4
            pygame.draw.line(self.screen, PANEL_LINE,
                             (px-4, py), (self.width-8, py), 1)
            py += 8

        blit('ARIA', WHITE, 'lg')
        blit(f'Gen {generation}  Ep {episode:5d}  Step {step:4d}', MUTED, 'sm')
        blit(f'epsilon: {list(agents.values())[0].epsilon:.4f}', MUTED, 'sm')
        divider()

        blit('ACTIVE', WHITE, 'md')
        for agent_id, agent in agents.items():
            col    = _agent_colour(agent_id)
            reward = ep_rewards.get(agent_id, 0.0)
            sg_txt = (f'  sg:{agent.sub_goal.template[:8]}'
                      if agent.sub_goal and agent.sub_goal.is_active else '')
            blit(f'{agent_id}', col, 'md')
            blit(f'  ep:{reward:7.1f}  tot:{agent.total_reward:9.1f}', MUTED, 'sm')
            blit(f'  {agent.role}  mem:{len(agent.cultural_memory)}{sg_txt}',
                 ROLE_COL.get(agent.role, MUTED), 'sm')

        divider()

        lex_summary  = channel.get_lexicon_summary()
        n_assigned   = sum(1 for e in lex_summary.values() if e['assigned'])
        blit(f'LEXICON  {n_assigned}/{len(lex_summary)} crystallised', WHITE, 'md')
        active = sorted(
            [e for e in lex_summary.values() if e['coord_successes'] > 0],
            key=lambda e: e['coord_successes'], reverse=True
        )
        if active:
            for entry in active[:4]:
                blit(f'  {entry["symbol"]:5s}  coord:{entry["coord_successes"]:4d}',
                     GOLD, 'sm')
        else:
            blit('  no coord signal yet', MUTED, 'sm')

        n_compound = channel.compound_lexicon.crystallised_count()
        blit(f'COMPOUNDS  [{n_compound} crystallised]', WHITE, 'md')
        for entry in list(channel.compound_lexicon.entries.values())[-4:]:
            col = COMPOUND_COL if entry.crystallised else MUTED
            sym = entry.symbol if entry.crystallised else '...'
            blit(f'  {sym:6s}  coord:{entry.coord_successes:3d}/{entry.use_count:4d}',
                 col, 'sm')

        divider()

        blit('SIGNALS', WHITE, 'md')
        for sender, symbol in self.signal_history[-5:]:
            blit(f'  {sender.split("-")[1]} -> {symbol}', _agent_colour(sender), 'sm')

        if self.replication_flash > 0 and self.last_replication:
            self.replication_flash -= 1
            s  = self.last_replication
            hp = s.get('child_hyperparams', {})
            divider()
            blit('REPLICATION', REPL_COL, 'md')
            blit(f'  born:    {s["child_id"]}', REPL_COL, 'sm')
            blit(f'  retired: {s["retired_id"]}', MUTED, 'sm')
            blit(f'  layers:{hp.get("n_layers","?")}  act:{hp.get("activation","?")}',
                 MUTED, 'sm')

        leg_y = self.height - 36
        pygame.draw.line(self.screen, PANEL_LINE,
                         (px-4, leg_y), (self.width-8, leg_y), 1)
        self.screen.blit(
            self.font_sm.render('$ solo   CO coord   S save   ESC quit', True, MUTED),
            (px, leg_y + 8)
        )

    def record_episode(self, ep_rewards):
        """Call at the end of each episode to update the reward trend graph."""
        for agent_id, reward in ep_rewards.items():
            if agent_id not in self.reward_history:
                self.reward_history[agent_id] = deque(maxlen=100)
            self.reward_history[agent_id].append(reward)

    def _draw_graph(self, agents, channel):
        LINE_SECTION = 460   # px allocated to line chart
        BAR_SECTION  = 180   # px allocated to reward bars
        # lexicon takes the remainder (~320px for 16 hex signal bars)

        gy = self.height + 22   # top of plot area
        gh = GRAPH_H - 42       # plot height

        # Background strip
        pygame.draw.rect(self.screen, PANEL_BG,
                         pygame.Rect(0, self.height, self.width, GRAPH_H))
        pygame.draw.line(self.screen, PANEL_LINE,
                         (0, self.height), (self.width, self.height), 2)

        # Section dividers
        pygame.draw.line(self.screen, PANEL_LINE,
                         (LINE_SECTION, self.height + 1),
                         (LINE_SECTION, self.height + GRAPH_H), 1)
        pygame.draw.line(self.screen, PANEL_LINE,
                         (LINE_SECTION + BAR_SECTION, self.height + 1),
                         (LINE_SECTION + BAR_SECTION, self.height + GRAPH_H), 1)

        # ── 1. LINE CHART ────────────────────────────────────────────
        gx = 55
        gw = LINE_SECTION - gx - 15

        title = self.font_sm.render('EPISODE REWARD TREND  (last 100 eps)', True, MUTED)
        self.screen.blit(title, (gx, self.height + 5))

        histories = {
            aid: list(self.reward_history[aid])
            for aid in agents
            if aid in self.reward_history and len(self.reward_history[aid]) > 1
        }

        if histories:
            all_vals = [v for h in histories.values() for v in h]
            y_min    = min(0.0, min(all_vals))
            y_max    = max(1.0, max(all_vals))
            y_range  = y_max - y_min

            def to_px(idx, n, reward):
                sx = gx + int(idx / max(n - 1, 1) * gw)
                sy = gy + gh - int((reward - y_min) / y_range * gh)
                sy = max(gy, min(gy + gh, sy))
                return sx, sy

            if y_min < 0 < y_max:
                zero_sy = gy + gh - int((0 - y_min) / y_range * gh)
                pygame.draw.line(self.screen, PANEL_LINE,
                                 (gx, zero_sy), (gx + gw, zero_sy), 1)

            for val in [y_max, (y_max + y_min) / 2, y_min]:
                sy  = gy + gh - int((val - y_min) / y_range * gh)
                lbl = self.font_sm.render(f'{val:.0f}', True, MUTED)
                self.screen.blit(lbl, (gx - lbl.get_width() - 3, sy - 5))

            for agent_id, history in histories.items():
                col    = _agent_colour(agent_id)
                n      = len(history)
                points = [to_px(i, n, v) for i, v in enumerate(history)]
                if len(points) >= 2:
                    pygame.draw.lines(self.screen, col, False, points, 1)

            pygame.draw.rect(self.screen, PANEL_LINE, pygame.Rect(gx, gy, gw, gh), 1)

        # ── 2. TOTAL REWARD BAR CHART ────────────────────────────────
        rx = LINE_SECTION + 12
        rw = BAR_SECTION - 24

        self.screen.blit(
            self.font_sm.render('TOTAL REWARD', True, MUTED),
            (rx, self.height + 5)
        )

        agent_list    = list(agents.items())
        n_agents      = len(agent_list)
        total_rewards = [max(0.0, ag.total_reward) for _, ag in agent_list]
        max_total     = max(max(total_rewards), 1.0)

        bar_gap    = 5
        bar_w_each = max(4, (rw - (n_agents - 1) * bar_gap) // n_agents)

        for i, (agent_id, agent) in enumerate(agent_list):
            col    = _agent_colour(agent_id)
            bx     = rx + i * (bar_w_each + bar_gap)
            reward = max(0.0, agent.total_reward)
            fill_h = max(2, int(reward / max_total * gh))
            by     = gy + gh - fill_h

            pygame.draw.rect(self.screen, col,
                             pygame.Rect(bx, by, bar_w_each, fill_h))
            pygame.draw.rect(self.screen, PANEL_LINE,
                             pygame.Rect(bx, gy, bar_w_each, gh), 1)

            tag = self.font_sm.render(agent_id.split('-')[1], True, col)
            self.screen.blit(tag, (bx + bar_w_each // 2 - tag.get_width() // 2,
                                   gy + gh + 2))

            val_lbl = self.font_sm.render(f'{agent.total_reward:.0f}', True, col)
            val_y   = by - 13
            if val_y >= gy:
                self.screen.blit(val_lbl,
                                 (bx + bar_w_each // 2 - val_lbl.get_width() // 2, val_y))

        # ── 3. LEXICON SIGNAL BAR CHART ──────────────────────────────
        lx = LINE_SECTION + BAR_SECTION + 12
        lw = self.width - lx - 8

        self.screen.blit(
            self.font_sm.render('LEXICON SIGNALS', True, MUTED),
            (lx, self.height + 5)
        )

        lex_entries = list(channel.get_lexicon_summary().values())
        n_lex       = len(lex_entries)
        if n_lex > 0:
            max_use   = max((e['use_count'] for e in lex_entries), default=1)
            max_use   = max(max_use, 1)
            lex_gap   = 2
            lex_bar_w = max(4, (lw - (n_lex - 1) * lex_gap) // n_lex)

            for i, entry in enumerate(lex_entries):
                col    = GOLD if entry['assigned'] else MUTED
                bx     = lx + i * (lex_bar_w + lex_gap)
                fill_h = max(2, int(entry['use_count'] / max_use * gh))
                by     = gy + gh - fill_h

                pygame.draw.rect(self.screen, col,
                                 pygame.Rect(bx, by, lex_bar_w, fill_h))
                pygame.draw.rect(self.screen, PANEL_LINE,
                                 pygame.Rect(bx, gy, lex_bar_w, gh), 1)

                # white tick showing coordination success ratio
                if entry['use_count'] > 0:
                    ratio  = entry['coord_successes'] / entry['use_count']
                    tick_y = gy + gh - int(ratio * gh)
                    pygame.draw.line(self.screen, WHITE,
                                     (bx, tick_y), (bx + lex_bar_w, tick_y), 1)

                # single hex digit label (0-F) below bar
                hex_lbl = self.font_sm.render(format(entry['signal_idx'], 'X'), True, col)
                self.screen.blit(hex_lbl, (bx + lex_bar_w // 2 - hex_lbl.get_width() // 2,
                                           gy + gh + 2))

    def _do_screenshot(self, episode):
        path = os.path.join('logs', 'screenshots')
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'aria_ep{episode:06d}.png')
        pygame.image.save(self.screen, filename)
        print(f'  [Screenshot] Saved → {filename}')

    def render(self, env, agents, channel, episode, step,
               ep_rewards, generation):
        self.handle_events()
        self.screen.fill(BG)
        self._draw_grid(env)
        self._draw_panel(agents, channel, episode, step, ep_rewards, generation)
        self._draw_graph(agents, channel)
        pygame.display.flip()
        if self._save_screenshot:
            self._do_screenshot(episode)
            self._save_screenshot = False
