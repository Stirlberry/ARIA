"""
ARIA — Adaptive Reasoning and Interaction Agent

Phase 3: Energy-based survival, death, ghost nodes, emergent communication,
         population dynamics, brain evolution, world models, cultural inheritance.

Run:
    python main.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

# Load .env file if present — sets ANTHROPIC_API_KEY without needing export
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
from genetics import (kill_weakest, energy_reproduce,
                      PlateauMonitor, get_alpha,
                      log_energy_death, log_extinction)
from visualiser import Visualiser
from config import (
    INITIAL_AGENTS, MAX_EPISODES, MAX_STEPS_PER_EPISODE,
    LEXICON_LOG_PATH,
    MIN_REPLICATION_INTERVAL, MAX_REPLICATION_INTERVAL,
    AUTOSAVE_EVERY, ENV_DRIFT_INTERVAL,
    MAX_POPULATION,
    ENERGY_MAX, REWARD_DEATH, REWARD_REPRODUCE,
    REPRODUCTION_THRESHOLD, REPRODUCTION_COST,
    SPAWN_PAUSE_STEPS,
)


def _absorb_ghost_weights(agent, ghost_state_dict):
    """Absorb a dead agent's network weights into a living agent via 50/50 crossover."""
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
        pass  # architecture mismatch — skip silently


_SEPARATION_DIRS = [(0, 1), (2, 3), (4, 7), (5, 6)]   # (dir_a, dir_b) opposite movement pairs

def _choose_separation():
    """Return (dir_a, dir_b) as opposite movement action indices."""
    return random.choice(_SEPARATION_DIRS)


def main():
    shared_replay  = make_replay_buffer()
    agents         = {a: ARIAAgent(a, replay=shared_replay) for a in INITIAL_AGENTS}
    env            = Environment(list(agents.keys()))
    channel        = CommunicationChannel()
    vis         = Visualiser()
    plateau_mon = PlateauMonitor()

    for agent_id in agents:
        plateau_mon.register(agent_id)

    all_ids_ever        = list(INITIAL_AGENTS)
    generation          = 0
    last_replication_ep = 0   # tracks births — resets on every reproduction
    last_plateau_ep     = 0   # tracks plateau retirements — separate clock

    print('=' * 64)
    print('  ARIA — Adaptive Reasoning and Interaction Agent')
    print('  Phase 3: Survival, Reproduction, Emergent Communication')
    print('=' * 64)
    print(f'  Founders        : {" + ".join(INITIAL_AGENTS)}')
    print(f'  Replication     : self-directed  '
          f'(min {MIN_REPLICATION_INTERVAL} eps / max {MAX_REPLICATION_INTERVAL} eps)')
    print(f'  Env drift       : every {ENV_DRIFT_INTERVAL} episodes')
    print(f'  Lexicon         : {LEXICON_LOG_PATH}')
    print(f'  ESC to quit')
    print('=' * 64)

    start_episode = 1
    meta, pt_path = save_system.prompt_resume()
    if meta is not None:
        (agents, channel, generation, last_replication_ep,
         all_ids_ever, plateau_history, start_episode) = save_system.restore(meta, pt_path, shared_replay)
        for agent_id, history in plateau_history.items():
            plateau_mon.register(agent_id)
            for r in history:
                plateau_mon.history[agent_id].append(r)
        env.reset(agent_ids=list(agents.keys()))
        print(f'  Resumed at episode {start_episode - 1:,}  '
              f'gen {generation}  agents: {", ".join(agents)}\n')

    ep_rewards   = {a: 0.0 for a in agents}
    ep_coord_any = False
    episode      = start_episode - 1

    try:
        for episode in range(start_episode, MAX_EPISODES + 1):

            # Environmental co-evolution: drift nodes every ENV_DRIFT_INTERVAL episodes
            if episode > 1 and (episode - 1) % ENV_DRIFT_INTERVAL == 0:
                env.drift_nodes()
                print(f'  [Drift] Nodes relocated at episode {episode}')

            # ── Death phase: plateau triggers kill-weakest (needs ≥ 2 to compare) ──
            # if len(agents) > 1:                          # old: fired regardless of population size
            if len(agents) >= MAX_POPULATION:
                should_kill, kill_reason = plateau_mon.should_replicate(
                    agents, episode, last_plateau_ep
                )
                if should_kill:
                    print(f'\n  [Plateau] Retirement at episode {episode} — {kill_reason}')
                    retiring = min(agents.values(), key=lambda a: a.total_reward)
                    retire_pos     = env.agent_positions.get(retiring.agent_id)
                    retire_weights = retiring.online_net.state_dict()
                    agents, death_summary = kill_weakest(agents, episode)
                    last_plateau_ep = episode
                    plateau_mon.deregister(death_summary['retired_id'])
                    if retire_pos:
                        env.add_ghost_node(retire_pos, retire_weights)
                    env.reset(agent_ids=list(agents.keys()), soft=True)
                    print(f'  Died    : {death_summary["retired_id"]} '
                          f'(reward {death_summary["retired_total_reward"]:.1f})'
                          f'{f"  ghost at {retire_pos}" if retire_pos else ""}')
                    print(f'  Survivors: {", ".join(agents)}\n')

            # Episode setup
            states             = env.reset(agent_ids=list(agents.keys()), soft=True)
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

                # ── Birth sequence: pause parents, reveal child on countdown ──
                if spawn_event is not None:
                    pa_id = spawn_event['parent_a_id']
                    pb_id = spawn_event['parent_b_id']
                    if pa_id not in agents or pb_id not in agents:
                        spawn_event = None   # cancel — a parent died during pause
                    else:
                        spawn_event['pause_remaining'] -= 1
                        if spawn_event['pause_remaining'] > 0:
                            actions[pa_id] = 8
                            actions[pb_id] = 8
                            signals_sent[pa_id] = None
                            signals_sent[pb_id] = None
                        else:
                            # Pause over — separate parents and birth child
                            actions[pa_id]      = spawn_event['dir_a']
                            actions[pb_id]      = spawn_event['dir_b']
                            signals_sent[pa_id] = None
                            signals_sent[pb_id] = None
                            parent_a = agents[pa_id]
                            parent_b = agents[pb_id]
                            agents, new_channel, summary = energy_reproduce(
                                parent_a, parent_b, agents, channel, episode,
                                set(all_ids_ever), shared_replay=shared_replay
                            )
                            channel.flush_log()
                            channel             = new_channel
                            child_id            = summary['child_id']
                            all_ids_ever.append(child_id)
                            generation         += 1
                            last_replication_ep = episode
                            plateau_mon.register(child_id)
                            env.add_newborn(child_id, spawn_event['spawn_point'])
                            last_signal_step[child_id]  = -(config.SIGNAL_WINDOW + 1)
                            last_signal_idx[child_id]   = None
                            last_message_step[child_id] = -(config.SIGNAL_WINDOW + 1)
                            last_message_idx[child_id]  = None
                            ep_rewards[child_id]        = 0.0
                            actions[child_id]            = 8
                            signals_sent[child_id]       = None
                            vis.notify_replication(summary)
                            hp  = summary['child_hyperparams']
                            c_e = agents[child_id].energy
                            print(f'\n  [Gen {generation}] Born {child_id} '
                                  f'(energy={c_e:.0f}) at {spawn_event["spawn_point"]} '
                                  f'ep {episode} step {step}')
                            print(f'  Parents : {pa_id} + {pb_id}')
                            print(f'  Hyperparams: lr={hp["lr"]}  '
                                  f'layers={hp["n_layers"]}  act={hp["activation"]}  '
                                  f'drain={hp["drain_rate"]}\n')
                            spawn_event = None

                # Signal tracking: runs after spawn override so parents aren't tracked during pause
                for agent_id, sig in signals_sent.items():
                    if sig is not None:
                        last_signal_step[agent_id] = step
                        last_signal_idx[agent_id]  = sig

                next_states, rewards, done, info, signals_received = env.step(
                    actions, signals_sent,
                    agent_energies={a: agents[a].energy for a in agents}
                )

                # Clear spawn-pause parents' message buffers to suppress phantom signals
                if spawn_event is not None:
                    env.clear_msg_buffer(spawn_event['parent_a_id'])
                    env.clear_msg_buffer(spawn_event['parent_b_id'])

                positions = env.get_positions()

                # ── Energy: drain, gain, cap ───────────────────────────────────
                pre_energies = {}
                for agent_id, agent in list(agents.items()):
                    pre_energies[agent_id] = agent.energy
                    agent.energy -= agent.drain_rate
                    agent.energy += info['energy_gains'].get(agent_id, 0.0)
                    agent.energy  = min(ENERGY_MAX, max(0.0, agent.energy))

                # ── Ghost node absorption ──────────────────────────────────────
                for agent_id, ghost_data in info['ghost_collected']:
                    if agent_id in agents:
                        _absorb_ghost_weights(agents[agent_id], ghost_data)
                        print(f'  [Ghost] {agent_id} absorbed knowledge at '
                              f'ep {episode} step {step}')

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
                    print(f'  [Death] {agent_id} energy depleted at '
                          f'ep {episode} step {step} — ghost at {death_pos}')
                    log_energy_death(agent, episode, step)
                    del agents[agent_id]

                if newly_dead:
                    env.reset(agent_ids=list(agents.keys()), soft=True)

                if not agents:
                    extinct = True
                    break

                if info['coord_achieved']:
                    ep_coord_any = True

                # Record individual signals and compound pairs
                active_sigs = []
                for agent_id, sig in signals_sent.items():
                    if sig is not None and agent_id not in newly_dead:
                        channel.record_signal(
                            agent_id, sig,
                            states[agent_id],
                            info['coord_achieved']
                        )
                        active_sigs.append(sig)
                        vis.add_signal(agent_id, channel.get_display_symbol(sig))

                # Compound lexicon: record all pairs of co-occurring signals
                for i in range(len(active_sigs)):
                    for j in range(i + 1, len(active_sigs)):
                        channel.record_signal_pair(
                            active_sigs[i], active_sigs[j],
                            info['coord_achieved']
                        )

                # Sequence lexicon: record completed messages
                for agent_id, msg_tokens in info.get('messages_sent', {}).items():
                    channel.record_message(agent_id, msg_tokens, info['coord_achieved'])
                    last_message_step[agent_id] = step
                    last_message_idx[agent_id]  = msg_tokens

                # Window-based coord credit for compounds and sequences.
                # When coord fires, agents used move actions to arrive — no same-step signals.
                # Credit pairs/sequences from within SIGNAL_WINDOW of the coord event.
                if info['coord_achieved']:
                    window_sigs = []
                    for aid in info['coord_agents']:
                        if signals_sent.get(aid) is None:  # didn't signal this step
                            sig_step = last_signal_step.get(aid, -(config.SIGNAL_WINDOW + 1))
                            if step - sig_step <= config.SIGNAL_WINDOW:
                                sig_idx = last_signal_idx.get(aid)
                                if sig_idx is not None:
                                    window_sigs.append(sig_idx)
                    for i in range(len(window_sigs)):
                        for j in range(i + 1, len(window_sigs)):
                            channel.compound_lexicon.credit_coord_pair(
                                window_sigs[i], window_sigs[j])

                    for aid in info['coord_agents']:
                        if aid not in info.get('messages_sent', {}):  # no message this step
                            msg_step = last_message_step.get(aid)
                            if msg_step is not None and step - msg_step <= config.SIGNAL_WINDOW:
                                msg_tokens = last_message_idx.get(aid)
                                if msg_tokens:
                                    channel.sequence_lexicon.credit_coord(msg_tokens)

                agent_list = list(agents.keys())

                # ── Energy-based reproduction detection ───────────────────────
                if spawn_event is None and len(agents) < MAX_POPULATION:
                    for i in range(len(agent_list)):
                        for j in range(i + 1, len(agent_list)):
                            a_id, b_id = agent_list[i], agent_list[j]
                            if a_id in newly_dead or b_id in newly_dead:
                                continue
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


                for agent_id, agent in agents.items():
                    if agent_id in newly_dead:
                        continue  # already updated in death handler above
                    if agent_id not in states or agent_id not in next_states:
                        continue  # agent joined mid-episode — skip
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

                vis.render(env, agents, channel, episode, step,
                           ep_rewards, generation, spawn_event=spawn_event)

                if done:
                    break

            # End of episode
            for agent_id, agent in agents.items():
                agent.decay_epsilon()
                agent.end_episode()
                partner_ids = [aid for aid in agents if aid != agent_id]
                agent.update_reputation(partner_ids, ep_coord_any)
                plateau_mon.record(agent_id, ep_rewards[agent_id])

            vis.record_episode(ep_rewards)

            # ── Extinction handler ─────────────────────────────────────────────
            if extinct:
                print('\n' + '=' * 64)
                print(f'  [EXTINCTION] All agents died — episode {episode}')
                print(f'  Generation reached  : {generation}')
                print(f'  Full lineage        : {" > ".join(all_ids_ever)}')
                print(f'  Total signals       : {channel.total_signals}')
                print(f'  Lexicon assigned    : {channel.assigned_count()}/16')
                print('=' * 64)
                log_extinction(episode, generation, all_ids_ever, channel)
                channel.flush_log()
                return

            # Console summary every 25 episodes
            if episode % 25 == 0 and agents:
                eps      = list(agents.values())[0].epsilon
                assigned = channel.assigned_count()
                compound = channel.compound_lexicon.crystallised_count()
                seqs     = channel.sequence_lexicon.crystallised_count()
                ids      = list(agents.keys())
                r_str    = '  '.join(f'{a.split("-")[1]}:{ep_rewards.get(a,0):.1f}'
                                      for a in ids)
                roles    = '/'.join(agents[a].role[0] for a in ids)
                goals    = '/'.join(agents[a].goal_label[:10] for a in ids)
                alpha_label = get_alpha(agents).split('-')[1]
                e_str = ' '.join(f'{a.split("-")[1]}:{agents[a].energy:.0f}' for a in ids)
                print(f'  Ep {episode:5d} | gen {generation} | '
                      f'pop {len(agents)}/{MAX_POPULATION} α:{alpha_label} | '
                      f'eps {eps:.4f} | {r_str} | nrg [{e_str}] | '
                      f'lex {assigned}/16 cmp {compound} seq {seqs} | '
                      f'roles [{roles}] | goals [{goals}]')

            # Autosave
            if episode % AUTOSAVE_EVERY == 0:
                channel.flush_log()
                save_system.save_checkpoint(
                    episode, agents, channel, generation,
                    last_replication_ep, all_ids_ever, plateau_mon
                )

    except KeyboardInterrupt:
        print('\n  Interrupted — saving checkpoint...')
        channel.flush_log()
        save_system.save_checkpoint(
            episode, agents, channel, generation,
            last_replication_ep, all_ids_ever, plateau_mon
        )
        print('  Exiting.')
        return

    channel.flush_log()
    save_system.save_checkpoint(
        MAX_EPISODES, agents, channel, generation,
        last_replication_ep, all_ids_ever, plateau_mon
    )
    print('\n  Training complete.')
    print(f'  Generations reached : {generation}')
    print(f'  Full lineage        : {" > ".join(all_ids_ever)}')
    print(f'  Total signals       : {channel.total_signals}')
    print(f'  Lexicon assigned    : {channel.assigned_count()}/16')
    print(f'  Compounds           : {channel.compound_lexicon.crystallised_count()}')


if __name__ == '__main__':
    main()
