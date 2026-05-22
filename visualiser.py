"""
ARIA Visualiser — Phase 2
Renders every step at training speed (no FPS cap).
"""

import sys
import pygame
from config import GRID_SIZE, CELL_SIZE

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
        self.cell   = CELL_SIZE
        grid_px     = GRID_SIZE * self.cell
        self.width  = grid_px + PANEL_W
        self.height = GRID_SIZE * self.cell
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('ARIA — Adaptive Reasoning and Interaction Agent')
        self.font_lg = pygame.font.SysFont('monospace', 17, bold=True)
        self.font_md = pygame.font.SysFont('monospace', 13)
        self.font_sm = pygame.font.SysFont('monospace', 11)
        self.signal_history    = []
        self.replication_flash = 0
        self.last_replication  = None

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
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit()

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

        blit('LEXICON', WHITE, 'md')
        for entry in channel.get_lexicon_summary().values():
            col  = GOLD if entry['assigned'] else MUTED
            blit(f'  {entry["symbol"]:5s}  use:{entry["use_count"]:5d}'
                 f'  coord:{entry["coord_successes"]:4d}', col, 'sm')

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
            self.font_sm.render('$ solo   CO coord   ESC quit', True, MUTED),
            (px, leg_y + 8)
        )

    def render(self, env, agents, channel, episode, step,
               ep_rewards, generation, all_ids_ever):
        self.handle_events()
        self.screen.fill(BG)
        self._draw_grid(env)
        self._draw_panel(agents, channel, episode, step, ep_rewards, generation)
        pygame.display.flip()
