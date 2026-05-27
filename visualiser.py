"""
ARIA-2 Visualiser — Lewis Signaling Game
Type-0 targets shown in cyan, type-1 targets in magenta.
Type-0 agents drawn as circles, type-1 agents drawn as diamonds.
"""

import os
import pygame
from collections import deque
from config import GRID_W, GRID_H, CELL_SIZE, FOG_RADIUS, MAX_POPULATION, ENERGY_MAX

GRAPH_H = 150

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
TARGET0_COL  = (80,  220, 220)   # cyan  — type-0 targets
TARGET1_COL  = (220, 80,  220)   # magenta — type-1 targets
COMPOUND_COL = (255, 160, 80)
GHOST_COL    = (120, 120, 150)
BIRTH_COL    = (255, 240, 100)
PANEL_W      = 320

AGENT_COLOURS = {
    'ARIA-CAFE': (90,  170, 255),
    'ARIA-BABE': (255, 140, 70),
    'ARIA-DEAD': (180, 110, 255),
    'ARIA-BEEF': (255, 100, 140),
}

ROLE_COL = {
    'coordinator': (100, 200, 255),
    'forager':     (255, 200,  80),
    'generalist':  (160, 230, 100),
    'exploring':   (130, 130, 160),
}

TYPE_COL = {0: TARGET0_COL, 1: TARGET1_COL}


def _agent_colour(agent_id):
    if agent_id in AGENT_COLOURS:
        return AGENT_COLOURS[agent_id]
    h = hash(agent_id) & 0xFFFFFF
    return (max(80, (h >> 16) & 0xFF), max(80, (h >> 8) & 0xFF), max(80, h & 0xFF))


class Visualiser:
    def __init__(self):
        pygame.init()
        self.cell    = CELL_SIZE
        self.width   = GRID_W * self.cell + PANEL_W
        self.height  = GRID_H * self.cell
        self.screen  = pygame.display.set_mode((self.width, self.height + GRAPH_H))
        self.reward_history = {}
        pygame.display.set_caption('ARIA-2 — Lewis Signaling Game')
        self.font_lg = pygame.font.SysFont('monospace', 17, bold=True)
        self.font_md = pygame.font.SysFont('monospace', 13)
        self.font_sm = pygame.font.SysFont('monospace', 9)
        self.signal_history    = []
        self.replication_flash = 0
        self.last_replication  = None
        self._save_screenshot  = False
        self._agent_scroll     = 0

        self.screen.fill(BG)
        cx = self.width // 2
        t1 = self.font_lg.render('ARIA-2', True, WHITE)
        t2 = self.font_md.render('Lewis Signaling Game', True, MUTED)
        t3 = self.font_md.render('Initialising...', True, MUTED)
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
            if event.type == pygame.MOUSEWHEEL:
                mx, _ = pygame.mouse.get_pos()
                if mx > GRID_W * self.cell:
                    self._agent_scroll -= event.y

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

    def _draw_diamond(self, surface, col, cx, cy, r):
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        pygame.draw.polygon(surface, col, pts)

    def _draw_grid(self, env, agents, agent_types, spawn_event=None):
        grid_px_w = GRID_W * self.cell
        grid_px_h = GRID_H * self.cell
        currency_nodes, target0_nodes, target1_nodes = env.get_nodes()
        positions = env.get_positions()

        for x in range(GRID_W):
            for y in range(GRID_H):
                pygame.draw.rect(self.screen, GRID_CELL,
                                 pygame.Rect(x*self.cell, y*self.cell,
                                             self.cell-1, self.cell-1))

        # Currency nodes (green)
        for (cx, cy) in currency_nodes:
            pad = 8
            pygame.draw.rect(self.screen, CURRENCY_COL,
                             pygame.Rect(cx*self.cell+pad, cy*self.cell+pad,
                                         self.cell-pad*2, self.cell-pad*2),
                             border_radius=3)
            t = self.font_sm.render('$', True, BG)
            self.screen.blit(t, (cx*self.cell+self.cell//2 - t.get_width()//2,
                                  cy*self.cell+self.cell//2 - t.get_height()//2))

        # Type-0 targets (cyan)
        for (tx, ty) in target0_nodes:
            pad = 8
            pygame.draw.rect(self.screen, TARGET0_COL,
                             pygame.Rect(tx*self.cell+pad, ty*self.cell+pad,
                                         self.cell-pad*2, self.cell-pad*2),
                             border_radius=3)
            t = self.font_sm.render('T0', True, BG)
            self.screen.blit(t, (tx*self.cell+self.cell//2 - t.get_width()//2,
                                  ty*self.cell+self.cell//2 - t.get_height()//2))

        # Type-1 targets (magenta)
        for (tx, ty) in target1_nodes:
            pad = 8
            pygame.draw.rect(self.screen, TARGET1_COL,
                             pygame.Rect(tx*self.cell+pad, ty*self.cell+pad,
                                         self.cell-pad*2, self.cell-pad*2),
                             border_radius=3)
            t = self.font_sm.render('T1', True, BG)
            self.screen.blit(t, (tx*self.cell+self.cell//2 - t.get_width()//2,
                                  ty*self.cell+self.cell//2 - t.get_height()//2))

        # Ghost nodes
        for (gx, gy) in env.ghost_nodes.keys():
            pcx = gx * self.cell + self.cell // 2
            pcy = gy * self.cell + self.cell // 2
            pygame.draw.circle(self.screen, GHOST_COL, (pcx, pcy), self.cell // 2 - 8, 1)
            t = self.font_sm.render('G', True, GHOST_COL)
            self.screen.blit(t, (pcx - t.get_width() // 2, pcy - 5))

        # Agents: circle=type-0, diamond=type-1
        for agent_id, (ax, ay) in positions.items():
            col   = _agent_colour(agent_id)
            tag   = agent_id.split('-')[1]
            pcx   = ax*self.cell + self.cell//2
            pcy   = ay*self.cell + self.cell//2
            atype = agent_types.get(agent_id, 0) if agent_types else 0
            r     = self.cell // 2 - 5

            if atype == 0:
                pygame.draw.circle(self.screen, col, (pcx, pcy), r)
            else:
                self._draw_diamond(self.screen, col, pcx, pcy, r)

            lbl = self.font_sm.render(tag, True, BG)
            self.screen.blit(lbl, (pcx - lbl.get_width()//2, pcy - lbl.get_height()//2))

            if agent_id in agents:
                energy   = agents[agent_id].energy
                bar_w    = self.cell - 10
                bar_x    = ax * self.cell + 5
                bar_y    = (ay + 1) * self.cell - 6
                filled_w = max(0, int(energy / ENERGY_MAX * bar_w))
                e_ratio  = energy / ENERGY_MAX
                e_col    = ((60, 200, 80) if e_ratio > 0.6 else
                            (220, 180, 40) if e_ratio > 0.3 else
                            (220, 60, 60))
                pygame.draw.rect(self.screen, (40, 40, 55),
                                 pygame.Rect(bar_x, bar_y, bar_w, 3))
                if filled_w > 0:
                    pygame.draw.rect(self.screen, e_col,
                                     pygame.Rect(bar_x, bar_y, filled_w, 3))

        if spawn_event is not None:
            self._draw_birth_ring(spawn_event)

        for i in range(GRID_W + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (i*self.cell, 0), (i*self.cell, grid_px_h), 1)
        for i in range(GRID_H + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (0, i*self.cell), (grid_px_w, i*self.cell), 1)

        fog = pygame.Surface((grid_px_w, grid_px_h))
        fog.fill((8, 8, 16))
        fog.set_colorkey((255, 0, 255))
        fog.set_alpha(200)
        reveal_r = int((FOG_RADIUS + 0.5) * self.cell)
        for (ax, ay) in positions.values():
            pcx = ax * self.cell + self.cell // 2
            pcy = ay * self.cell + self.cell // 2
            pygame.draw.circle(fog, (255, 0, 255), (pcx, pcy), reveal_r)
        self.screen.blit(fog, (0, 0))

    def _draw_birth_ring(self, spawn_event):
        sx, sy = spawn_event['spawn_point']
        cx = sx * self.cell + self.cell // 2
        cy = sy * self.cell + self.cell // 2
        pause_rem   = spawn_event['pause_remaining']
        pause_total = spawn_event['pause_total']
        t = 1.0 - pause_rem / max(pause_total, 1)
        for phase in (0.0, 0.33, 0.66):
            tp   = (t + phase) % 1.0
            r    = int(self.cell * 0.25 + tp * self.cell * 1.1)
            fade = max(0.0, 1.0 - tp)
            col  = (int(BIRTH_COL[0] * fade),
                    int(BIRTH_COL[1] * fade),
                    int(BIRTH_COL[2] * fade))
            if r > 1 and (col[0] > 8 or col[1] > 8):
                pygame.draw.circle(self.screen, col, (cx, cy), r, 2)

    def _draw_panel(self, agents, agent_types, channel, episode, step,
                    ep_rewards, generation):
        grid_px_w = GRID_W * self.cell
        pygame.draw.rect(self.screen, PANEL_BG,
                         pygame.Rect(grid_px_w, 0, PANEL_W, self.height))
        pygame.draw.line(self.screen, PANEL_LINE,
                         (grid_px_w, 0), (grid_px_w, self.height), 2)

        px = grid_px_w + 14
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

        blit('ARIA-2', WHITE, 'lg')
        blit(f'Gen {generation}  Ep {episode:5d}  Step {step:4d}', MUTED, 'sm')
        type_counts = {0: 0, 1: 0}
        for a in agents:
            type_counts[agent_types.get(a, 0)] += 1
        blit(f'T0:{type_counts[0]} ◯  T1:{type_counts[1]} ◇', MUTED, 'sm')
        divider()

        def _avg_reward(a):
            return a.total_reward / a.episodes if a.episodes > 0 else 0.0
        alpha_id = max(agents.values(), key=_avg_reward).agent_id if agents else None

        blit(f'ACTIVE  {len(agents)} agents', WHITE, 'md')

        _ITEM_H  = 68
        _VISIBLE = 4
        _LIST_H  = _ITEM_H * _VISIBLE
        agent_items = list(agents.items())
        n_agents    = len(agent_items)
        n_vis       = min(n_agents, _VISIBLE)
        self._agent_scroll = max(0, min(n_agents - n_vis, self._agent_scroll))
        list_top = py

        self.screen.set_clip(pygame.Rect(grid_px_w, list_top, PANEL_W, _LIST_H))
        for agent_id, agent in agent_items[self._agent_scroll: self._agent_scroll + n_vis]:
            col          = _agent_colour(agent_id)
            reward       = ep_rewards.get(agent_id, 0.0)
            atype        = agent_types.get(agent_id, 0)
            type_marker  = '◯' if atype == 0 else '◇'
            alpha_marker = ' [a]' if agent_id == alpha_id else ''
            blit(f'{agent_id} T{atype}{type_marker}{alpha_marker}', col, 'md')
            blit(f'  ep:{reward:7.1f}  tot:{agent.total_reward:9.1f}', MUTED, 'sm')
            blit(f'  nrg:{agent.energy:5.0f}  drain:{agent.drain_rate:.1f}', MUTED, 'sm')
            blit(f'  {agent.role}  mem:{len(agent.cultural_memory)}',
                 ROLE_COL.get(agent.role, MUTED), 'sm')
        self.screen.set_clip(None)

        if n_agents > _VISIBLE:
            sb_x = self.width - 5
            th_h = max(20, _LIST_H * _VISIBLE // n_agents)
            th_y = list_top + (_LIST_H - th_h) * self._agent_scroll // max(1, n_agents - _VISIBLE)
            pygame.draw.rect(self.screen, PANEL_LINE, pygame.Rect(sb_x, list_top, 2, _LIST_H))
            pygame.draw.rect(self.screen, MUTED,      pygame.Rect(sb_x, th_y,    2, th_h))

        py = list_top + _LIST_H
        divider()

        lex_summary = channel.get_lexicon_summary()
        n_assigned  = sum(1 for e in lex_summary.values() if e['assigned'])
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
        top_compounds = sorted(
            channel.compound_lexicon.entries.values(),
            key=lambda e: (e.crystallised, e.coord_successes),
            reverse=True
        )[:4]
        for entry in top_compounds:
            col = COMPOUND_COL if entry.crystallised else MUTED
            sym = entry.symbol if entry.crystallised else '[?]'
            blit(f'  {sym:6s}  coord:{entry.coord_successes:3d}/{entry.use_count:4d}',
                 col, 'sm')

        divider()

        blit('SIGNALS', WHITE, 'md')
        for sender, symbol in self.signal_history[-5:]:
            t_lbl = f'T{agent_types.get(sender, "?")}' if agent_types else ''
            blit(f'  {sender.split("-")[1]}{t_lbl} → {symbol}',
                 _agent_colour(sender), 'sm')

        if self.replication_flash > 0 and self.last_replication:
            self.replication_flash -= 1
            s  = self.last_replication
            hp = s.get('child_hyperparams', {})
            divider()
            blit('REPLICATION', (100, 230, 160), 'md')
            blit(f'  born: {s["child_id"]} T{s.get("child_type","?")}',
                 (100, 230, 160), 'sm')
            blit(f'  layers:{hp.get("n_layers","?")}  act:{hp.get("activation","?")}',
                 MUTED, 'sm')

        leg_y = self.height - 36
        pygame.draw.line(self.screen, PANEL_LINE, (px-4, leg_y), (self.width-8, leg_y), 1)
        self.screen.blit(
            self.font_sm.render('cyan=T0  magenta=T1  ◯T0  ◇T1  S  ESC', True, MUTED),
            (px, leg_y + 8)
        )

    def record_episode(self, ep_rewards):
        for agent_id, reward in ep_rewards.items():
            if agent_id not in self.reward_history:
                self.reward_history[agent_id] = deque(maxlen=100)
            self.reward_history[agent_id].append(reward)

    def _draw_graph(self, agents, agent_types, channel):
        LINE_SECTION = GRID_W * CELL_SIZE // 2
        BAR_SECTION  = GRID_W * CELL_SIZE - LINE_SECTION

        gy = self.height + 22
        gh = GRAPH_H - 42

        pygame.draw.rect(self.screen, PANEL_BG,
                         pygame.Rect(0, self.height, self.width, GRAPH_H))
        pygame.draw.line(self.screen, PANEL_LINE,
                         (0, self.height), (self.width, self.height), 2)
        pygame.draw.line(self.screen, PANEL_LINE,
                         (LINE_SECTION, self.height + 1),
                         (LINE_SECTION, self.height + GRAPH_H), 1)
        pygame.draw.line(self.screen, PANEL_LINE,
                         (LINE_SECTION + BAR_SECTION, self.height + 1),
                         (LINE_SECTION + BAR_SECTION, self.height + GRAPH_H), 1)

        # ── 1. LINE CHART ──────────────────────────────────────────────────────
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
            y_min = min(0.0, min(all_vals))
            y_max = max(1.0, max(all_vals))
            y_range = y_max - y_min

            def to_px(idx, n, reward):
                sx = gx + int(idx / max(n - 1, 1) * gw)
                sy = gy + gh - int((reward - y_min) / y_range * gh)
                return sx, max(gy, min(gy + gh, sy))

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
                faint  = tuple(int(c * 0.25 + PANEL_BG[k] * 0.75)
                               for k, c in enumerate(col))
                if len(points) >= 2:
                    pygame.draw.lines(self.screen, faint, False, points, 1)
                window  = 20
                avg     = [sum(history[max(0, i - window + 1): i + 1]) / min(i + 1, window)
                           for i in range(n)]
                avg_pts = [to_px(i, n, v) for i, v in enumerate(avg)]
                if len(avg_pts) >= 2:
                    pygame.draw.lines(self.screen, col, False, avg_pts, 2)

            pygame.draw.rect(self.screen, PANEL_LINE, pygame.Rect(gx, gy, gw, gh), 1)

        # ── 2. TOTAL REWARD BAR CHART ──────────────────────────────────────────
        rx = LINE_SECTION + 12
        rw = BAR_SECTION - 24
        self.screen.blit(self.font_sm.render('TOTAL REWARD', True, MUTED),
                         (rx, self.height + 5))

        agent_list    = list(agents.items())
        n_agents      = len(agent_list)
        n_bars        = MAX_POPULATION
        total_rewards = [max(0.0, ag.total_reward) for _, ag in agent_list]
        max_total     = max(max(total_rewards), 1.0) if total_rewards else 1.0

        bar_gap    = 5
        bar_w_each = max(4, (rw - (n_bars - 1) * bar_gap) // n_bars)

        for i in range(n_bars):
            bx = rx + i * (bar_w_each + bar_gap)
            if i < n_agents:
                agent_id, agent = agent_list[i]
                col    = _agent_colour(agent_id)
                reward = max(0.0, agent.total_reward)
                fill_h = max(2, int(reward / max_total * gh))
                by     = gy + gh - fill_h
                pygame.draw.rect(self.screen, col, pygame.Rect(bx, by, bar_w_each, fill_h))
                pygame.draw.rect(self.screen, PANEL_LINE, pygame.Rect(bx, gy, bar_w_each, gh), 1)
                tag = self.font_sm.render(agent_id.split('-')[1], True, col)
                self.screen.blit(tag, (bx + bar_w_each // 2 - tag.get_width() // 2,
                                       gy + gh + 2))
            else:
                pygame.draw.rect(self.screen, PANEL_LINE, pygame.Rect(bx, gy, bar_w_each, gh), 1)
                tag = self.font_sm.render('---', True, MUTED)
                self.screen.blit(tag, (bx + bar_w_each // 2 - tag.get_width() // 2,
                                       gy + gh + 2))

        # ── 3. LEXICON SIGNAL BAR CHART ────────────────────────────────────────
        lx = LINE_SECTION + BAR_SECTION + 12
        lw = self.width - lx - 8
        self.screen.blit(self.font_sm.render('LEXICON SIGNALS', True, MUTED),
                         (lx, self.height + 5))

        lex_entries = list(channel.get_lexicon_summary().values())
        n_lex = len(lex_entries)
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
                pygame.draw.rect(self.screen, col, pygame.Rect(bx, by, lex_bar_w, fill_h))
                pygame.draw.rect(self.screen, PANEL_LINE, pygame.Rect(bx, gy, lex_bar_w, gh), 1)
                if entry['use_count'] > 0:
                    ratio  = entry['coord_successes'] / entry['use_count']
                    tick_y = gy + gh - int(ratio * gh)
                    pygame.draw.line(self.screen, WHITE,
                                     (bx, tick_y), (bx + lex_bar_w, tick_y), 1)
                hex_lbl = self.font_sm.render(format(entry['signal_idx'], 'X'), True, col)
                self.screen.blit(hex_lbl, (bx + lex_bar_w // 2 - hex_lbl.get_width() // 2,
                                           gy + gh + 2))

    def _do_screenshot(self, episode):
        path = os.path.join('logs', 'screenshots')
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'aria2_ep{episode:06d}.png')
        pygame.image.save(self.screen, filename)
        print(f'  [Screenshot] Saved → {filename}')

    def render(self, env, agents, agent_types, channel, episode, step,
               ep_rewards, generation, spawn_event=None):
        self.handle_events()
        self.screen.fill(BG)
        self._draw_grid(env, agents, agent_types, spawn_event)
        self._draw_panel(agents, agent_types, channel, episode, step, ep_rewards, generation)
        self._draw_graph(agents, agent_types, channel)
        pygame.display.flip()
        if self._save_screenshot:
            self._do_screenshot(episode)
            self._save_screenshot = False
