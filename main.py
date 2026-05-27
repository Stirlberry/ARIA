"""
ARIA-2 — Lewis Signaling Game

Agents are split into type-0 and type-1. Each type can only SEE its own
target nodes. Reward fires only when both types occupy the SAME target tile
simultaneously. Without signals, the blind agent must guess — coordination
drops to ~1/768 (random walk on a 32×24 grid). With signals, the sighted
agent can guide the blind agent to the exact tile.

Run:
    python main.py
"""

import sys
import argparse
sys.stdout.reconfigure(line_buffering=True)

import os
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                if _k.strip() and _v.strip() not in ('', 'your_key_here'):
                    os.environ.setdefault(_k.strip(), _v.strip())

import random
import torch
import config
import save_system
from environment import Environment
from agent import ARIAAgent, make_replay_buffer
from communication import CommunicationChannel
from genetics import (select_weakest_per_type, kill_agent,
                      cross_type_reproduce, get_alpha,
                      log_energy_death, log_ghost_absorption, log_extinction)
from visualiser import Visualiser
from config import (
    INITIAL_AGENTS, FOUNDER_TYPES, MAX_EPISODES, MAX_STEPS_PER_EPISODE,
    LEXICON_LOG_PATH,
    AUTOSAVE_EVERY, ENV_DRIFT_INTERVAL,
    MAX_POPULATION, MIN_POPULATION,
    ENERGY_MAX, REWARD_DEATH, REWARD_REPRODUCE,
    REPRODUCTION_THRESHOLD, REPRODUCTION_COST, SPAWN_PAUSE_STEPS,
)


def _absorb_ghost_weights(agent, ghost_state_dict):
    try:
        live_params  = list(agent.online_net.parameters())
        ghost_values = list(ghost_state_dict.values())
        with torch.no_grad():
            for i, param in enumerate(live_params):
                if i < len(ghost_values) and ghost_values[i].shape == param.shape:
                    mask = torch.rand_like(param) < 0.5
                    param.copy_(torch.where(mask, ghost_values[i], param))
        agent.target_net.load_state_dict(agent.online_net.state_dict())
    except Exception:
        pass



_SEPARATION_DIRS = [(0, 1), (2, 3), (4, 7), (5, 6)]

def _choose_separation():
    return random.choice(_SEPARATION_DIRS)


def main():
    shared_replay = make_replay_buffer()

    # Build founding agents with assigned types
    agents      = {}
    agent_types = {}
    for agent_id in INITIAL_AGENTS:
        t = FOUNDER_TYPES.get(agent_id, 0)
        agents[agent_id]      = ARIAAgent(agent_id, agent_type=t, replay=shared_replay)
        agent_types[agent_id] = t

    env     = Environment(list(agents.keys()), agent_types=agent_types)
    channel = CommunicationChannel()
    vis     = Visualiser()

    all_ids_ever           = list(INITIAL_AGENTS)
    generation             = 0
    last_replication_ep    = 0
    total_signalled_coords = 0
    total_random_coords    = 0

    print('=' * 64)
    print('  ARIA-2 — Lewis Signaling Game')
    print('  Type-0 agents see type-0 targets. Type-1 see type-1.')
    print('  Reward fires only when BOTH types meet on the same tile.')
    print('  Without signals: coordination ≈ 1/768. With: learned language.')
    print('=' * 64)
    type_str = '  '.join(f'{a.split("-")[1]}:T{agent_types[a]}' for a in INITIAL_AGENTS)
    print(f'  Founders  : {type_str}')
    print(f'  Balance   : checked every episode — majority type culled if uneven')
    print(f'  ESC to quit')
    print('=' * 64)

    start_episode = 1
    meta, pt_path = save_system.prompt_resume()
    if meta is not None:
        (agents, agent_types, channel, generation, last_replication_ep,
         all_ids_ever, _, start_episode, _) = save_system.restore(meta, pt_path, shared_replay)
        env.reset(agent_ids=list(agents.keys()), agent_types=agent_types)
        print(f'  Resumed at episode {start_episode - 1:,}  '
              f'gen {generation}  agents: {", ".join(agents)}\n')

    ep_rewards   = {a: 0.0 for a in agents}
    ep_coord_any = False
    episode      = start_episode - 1

    try:
        for episode in range(start_episode, MAX_EPISODES + 1):

            if episode > 1 and (episode - 1) % ENV_DRIFT_INTERVAL == 0:
                env.drift_nodes()
                print(f'  [Drift] Nodes relocated at episode {episode}')

            # ── Balance check: every episode, cull weakest of majority type ──
            counts = {0: 0, 1: 0}
            for aid in agents:
                counts[agent_types.get(aid, 0)] += 1
            if counts[0] != counts[1] and len(agents) > MIN_POPULATION:
                majority_type = 0 if counts[0] > counts[1] else 1
                pool    = {aid: a for aid, a in agents.items()
                           if agent_types.get(aid, 0) == majority_type}
                weakest = min(pool.values(), key=lambda a: a.total_reward)
                retire_pos     = env.agent_positions.get(weakest.agent_id)
                retire_weights = weakest.online_net.state_dict()
                agents, death_summary = kill_agent(agents, episode, weakest)
                agent_types.pop(death_summary['retired_id'], None)
                if retire_pos:
                    env.add_ghost_node(retire_pos, retire_weights)
                env.reset(agent_ids=list(agents.keys()), agent_types=agent_types, soft=True)
                print(f'  [Balance] ep {episode} — T0:{counts[0]} T1:{counts[1]} '
                      f'→ removed {death_summary["retired_id"]} T{majority_type} '
                      f'(reward {death_summary["retired_total_reward"]:.1f})')

            states             = env.reset(agent_ids=list(agents.keys()),
                                           agent_types=agent_types, soft=True)
            ep_rewards        = {a: 0.0 for a in agents}
            ep_coord_any      = False
            extinct           = False
            last_signal_step  = {a: -(config.SIGNAL_WINDOW + 1) for a in agents}
            last_signal_idx   = {}
            last_message_step = {}
            last_message_idx  = {}
            spawn_event       = None

            for step in range(MAX_STEPS_PER_EPISODE):

                actions      = {}
                signals_sent = {a: None for a in agents}

                for agent_id, agent in agents.items():
                    action = agent.select_action(states[agent_id])
                    actions[agent_id]      = action
                    signals_sent[agent_id] = ARIAAgent.get_signal_from_action(action)

                # ── Birth sequence: T0×T1 collision produces one T0 + one T1 ──
                if spawn_event is not None:
                    pa_id = spawn_event['parent_a_id']
                    pb_id = spawn_event['parent_b_id']
                    if pa_id not in agents or pb_id not in agents:
                        spawn_event = None
                    else:
                        spawn_event['pause_remaining'] -= 1
                        if spawn_event['pause_remaining'] > 0:
                            actions[pa_id]      = 8
                            actions[pb_id]      = 8
                            signals_sent[pa_id] = None
                            signals_sent[pb_id] = None
                        else:
                            actions[pa_id]      = spawn_event['dir_a']
                            actions[pb_id]      = spawn_event['dir_b']
                            signals_sent[pa_id] = None
                            signals_sent[pb_id] = None
                            parent_a = agents[pa_id]
                            parent_b = agents[pb_id]
                            agents, new_channel, children = cross_type_reproduce(
                                parent_a, parent_b, agents, channel, episode,
                                all_ids_ever, shared_replay=shared_replay
                            )
                            channel.flush_log()
                            channel = new_channel
                            for child_id, child_type, child_agent in children:
                                all_ids_ever.append(child_id)
                                agent_types[child_id] = child_type
                                generation           += 1
                                last_replication_ep   = episode
                                env.add_newborn(child_id, spawn_event['spawn_point'], child_type)
                                last_signal_step[child_id]  = -(config.SIGNAL_WINDOW + 1)
                                last_signal_idx[child_id]   = None
                                last_message_step[child_id] = -(config.SIGNAL_WINDOW + 1)
                                last_message_idx[child_id]  = None
                                ep_rewards[child_id]        = 0.0
                                actions[child_id]           = 8
                                signals_sent[child_id]      = None
                                c_e = agents[child_id].energy
                                print(f'\n  [Gen {generation}] Born {child_id} '
                                      f'T{child_type} (energy={c_e:.0f}) '
                                      f'at {spawn_event["spawn_point"]} ep {episode}')
                            print(f'  Parents : {pa_id} (T{agent_types.get(pa_id,"?")}) '
                                  f'+ {pb_id} (T{agent_types.get(pb_id,"?")})\n')
                            spawn_event = None

                for agent_id, sig in signals_sent.items():
                    if sig is not None:
                        last_signal_step[agent_id] = step
                        last_signal_idx[agent_id]  = sig

                next_states, rewards, done, info, signals_received = env.step(
                    actions, signals_sent,
                    agent_energies={a: agents[a].energy for a in agents}
                )

                positions = env.get_positions()

                # ── Energy ────────────────────────────────────────────────────
                pre_energies = {}
                for agent_id, agent in list(agents.items()):
                    pre_energies[agent_id] = agent.energy
                    agent.energy -= agent.drain_rate
                    agent.energy += info['energy_gains'].get(agent_id, 0.0)
                    agent.energy  = min(ENERGY_MAX, max(0.0, agent.energy))

                # ── Ghost absorption ───────────────────────────────────────────
                for agent_id, ghost_data in info['ghost_collected']:
                    if agent_id in agents:
                        _absorb_ghost_weights(agents[agent_id], ghost_data)
                        log_ghost_absorption(agent_id, episode, step,
                                             agents[agent_id].total_reward,
                                             agents[agent_id].energy)
                        print(f'  [Ghost] {agent_id} absorbed knowledge ep {episode}')

                # ── Death: energy depleted ────────────────────────────────────
                newly_dead = [aid for aid, a in agents.items() if a.energy <= 0]

                for agent_id in newly_dead:
                    agent = agents[agent_id]
                    rewards[agent_id] += REWARD_DEATH
                    agent.update(
                        states[agent_id], actions[agent_id],
                        rewards[agent_id], next_states[agent_id],
                        info['coord_achieved'],
                        pre_energy=pre_energies[agent_id],
                    )
                    ep_rewards[agent_id] = ep_rewards.get(agent_id, 0.0) + rewards[agent_id]
                    death_pos = env.get_positions().get(agent_id)
                    if death_pos:
                        env.add_ghost_node(death_pos, agent.online_net.state_dict())
                    print(f'  [Death] {agent_id} (T{agent.agent_type}) energy depleted '
                          f'ep {episode} step {step}')
                    log_energy_death(agent, episode, step)
                    del agents[agent_id]
                    agent_types.pop(agent_id, None)

                if newly_dead:
                    env.reset(agent_ids=list(agents.keys()),
                               agent_types=agent_types, soft=True)

                if not agents:
                    extinct = True
                    break

                if info['coord_achieved']:
                    ep_coord_any = True
                    if info.get('n_coord_signalled', 0) > 0:
                        total_signalled_coords += info['n_coord_signalled']
                        print(f'  [Coord] ep {episode} step {step} — SIGNALLED '
                              f'| agents: {" + ".join(info["coord_agents"])}')
                    if info.get('n_coord_random', 0) > 0:
                        total_random_coords += info['n_coord_random']
                        print(f'  [Coord] ep {episode} step {step} — RANDOM (no reward) '
                              f'| agents: {" + ".join(info["coord_agents"])}')

                # Record signals and compounds
                active_sigs = []
                for agent_id, sig in signals_sent.items():
                    if sig is not None and agent_id not in newly_dead:
                        channel.record_signal(
                            agent_id, sig, states[agent_id], info['coord_achieved'])
                        active_sigs.append(sig)
                        vis.add_signal(agent_id, channel.get_display_symbol(sig))

                for i in range(len(active_sigs)):
                    for j in range(i + 1, len(active_sigs)):
                        channel.record_signal_pair(
                            active_sigs[i], active_sigs[j], info['coord_achieved'])

                for agent_id, msg_tokens in info.get('messages_sent', {}).items():
                    channel.record_message(agent_id, msg_tokens, info['coord_achieved'])
                    last_message_step[agent_id] = step
                    last_message_idx[agent_id]  = msg_tokens

                if info['coord_achieved']:
                    signals_by_sender = {}
                    for aid in info['coord_agents']:
                        if signals_sent.get(aid) is None:
                            sig_step = last_signal_step.get(aid, -(config.SIGNAL_WINDOW + 1))
                            if step - sig_step <= config.SIGNAL_WINDOW:
                                sig_idx = last_signal_idx.get(aid)
                                if sig_idx is not None:
                                    signals_by_sender[aid] = sig_idx
                    window_sigs = list(signals_by_sender.values())
                    channel.record_coord_event(episode, step,
                                               list(info['coord_agents']),
                                               signals_by_sender)
                    for i in range(len(window_sigs)):
                        for j in range(i + 1, len(window_sigs)):
                            channel.compound_lexicon.credit_coord_pair(
                                window_sigs[i], window_sigs[j])

                    for aid in info['coord_agents']:
                        if aid not in info.get('messages_sent', {}):
                            msg_step = last_message_step.get(aid)
                            if msg_step is not None and step - msg_step <= config.SIGNAL_WINDOW:
                                msg_tokens = last_message_idx.get(aid)
                                if msg_tokens:
                                    channel.sequence_lexicon.credit_coord(msg_tokens)

                for agent_id, agent in agents.items():
                    if agent_id in newly_dead:
                        continue
                    if agent_id not in states or agent_id not in next_states:
                        continue
                    agent.update(
                        states[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_states[agent_id],
                        info['coord_achieved'],
                        pre_energy=pre_energies.get(agent_id, agent.energy),
                    )
                    ep_rewards[agent_id] += rewards[agent_id]

                states = {a: next_states[a] for a in agents if a in next_states}

                # ── Cross-type reproduction: T0+T1 collision → one T0 + one T1 ─
                agent_list = list(agents.keys())
                if spawn_event is None and len(agents) + 2 <= MAX_POPULATION:
                    for i in range(len(agent_list)):
                        for j in range(i + 1, len(agent_list)):
                            a_id, b_id = agent_list[i], agent_list[j]
                            if a_id in newly_dead or b_id in newly_dead:
                                continue
                            if agent_types.get(a_id, 0) == agent_types.get(b_id, 0):
                                continue  # same type — cannot reproduce
                            ag_a = agents.get(a_id)
                            ag_b = agents.get(b_id)
                            if ag_a is None or ag_b is None:
                                continue
                            if (positions.get(a_id) == positions.get(b_id) and
                                    ag_a.energy >= REPRODUCTION_THRESHOLD and
                                    ag_b.energy >= REPRODUCTION_THRESHOLD):
                                ag_a.energy = max(0.0, ag_a.energy - REPRODUCTION_COST)
                                ag_b.energy = max(0.0, ag_b.energy - REPRODUCTION_COST)
                                rewards[a_id] = rewards.get(a_id, 0.0) + REWARD_REPRODUCE
                                rewards[b_id] = rewards.get(b_id, 0.0) + REWARD_REPRODUCE
                                dir_a, dir_b = _choose_separation()
                                spawn_event = {
                                    'parent_a_id':     a_id,
                                    'parent_b_id':     b_id,
                                    'spawn_point':     positions.get(a_id),
                                    'pause_remaining': SPAWN_PAUSE_STEPS + 1,
                                    'pause_total':     SPAWN_PAUSE_STEPS + 1,
                                    'dir_a':           dir_a,
                                    'dir_b':           dir_b,
                                }
                                break
                        if spawn_event is not None:
                            break

                vis.render(env, agents, agent_types, channel, episode, step,
                           ep_rewards, generation, spawn_event=spawn_event)

                if done:
                    break

            # End of episode
            for agent_id, agent in agents.items():
                agent.decay_epsilon()
                agent.end_episode()
                partner_ids = [aid for aid in agents if aid != agent_id]
                agent.update_reputation(partner_ids, ep_coord_any)

            vis.record_episode(ep_rewards)

            if extinct:
                print('\n' + '=' * 64)
                print(f'  [EXTINCTION] All agents died — episode {episode}')
                print(f'  Generation reached  : {generation}')
                print(f'  Total signals       : {channel.total_signals}')
                print('=' * 64)
                log_extinction(episode, generation, all_ids_ever, channel)
                channel.flush_log()
                return

            if episode % 50 == 0 and agents:
                assigned    = channel.assigned_count()
                compound    = channel.compound_lexicon.crystallised_count()
                seqs        = channel.sequence_lexicon.crystallised_count()
                ids         = list(agents.keys())
                alpha_label = get_alpha(agents).split('-')[1]
                type_counts = {0: 0, 1: 0}
                for a in ids:
                    type_counts[agent_types.get(a, 0)] += 1
                print(f'  Ep {episode:5d} | gen {generation} | '
                      f'pop {len(agents)}/{MAX_POPULATION} | '
                      f'T0:{type_counts[0]} T1:{type_counts[1]} | '
                      f'alpha:{alpha_label}')
                e_vals = '  '.join(f'{a.split("-")[1]}:T{agent_types.get(a,0)}:{agents[a].energy:.0f}'
                                   for a in ids)
                print(f'    energy   {e_vals}')
                print(f'    lexicon {assigned}/16 | compounds {compound} | sequences {seqs}')
                print(f'    coord    signalled:{total_signalled_coords}  '
                      f'random:{total_random_coords}')
                if channel.coord_pairs:
                    top_pair  = max(channel.coord_pairs, key=channel.coord_pairs.get)
                    top_count = channel.coord_pairs[top_pair]
                    pair_str  = (f'{top_pair[0].split("-")[1]}'
                                 f'↔{top_pair[1].split("-")[1]} ×{top_count}')
                    print(f'    top coord pair  {pair_str}')

            if episode % AUTOSAVE_EVERY == 0:
                channel.flush_log()
                save_system.save_checkpoint(
                    episode, agents, agent_types, channel, generation,
                    last_replication_ep, all_ids_ever
                )

    except KeyboardInterrupt:
        print('\n  Interrupted — saving checkpoint...')
        channel.flush_log()
        save_system.save_checkpoint(
            episode, agents, agent_types, channel, generation,
            last_replication_ep, all_ids_ever
        )
        print('  Exiting.')
        return

    channel.flush_log()
    save_system.save_checkpoint(
        MAX_EPISODES, agents, agent_types, channel, generation,
        last_replication_ep, all_ids_ever
    )
    print('\n  Training complete.')
    print(f'  Generations reached : {generation}')
    print(f'  Total signals       : {channel.total_signals}')
    print(f'  Lexicon assigned    : {channel.assigned_count()}/16')
    print(f'  Compounds           : {channel.compound_lexicon.crystallised_count()}')


if __name__ == '__main__':
    main()
