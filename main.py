"""
ARIA — Adaptive Reasoning and Interaction Agent

Phase 2: Population dynamics, brain evolution, world models, emergent language,
         cultural inheritance, environmental co-evolution, creative agency.

Run:
    python main.py

Optional (for Claude-powered help + sub-goal generation):
    export ANTHROPIC_API_KEY=your_key_here
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

import config
import save_system
from budget import cost_tracker
from environment import Environment
from agent import ARIAAgent, make_replay_buffer
from communication import CommunicationChannel
from genetics import replicate, PlateauMonitor
from help_system import HelpMonitor, LexiconAdvisor
from visualiser import Visualiser
from config import (
    INITIAL_AGENTS, MAX_EPISODES, MAX_STEPS_PER_EPISODE,
    LEXICON_LOG_PATH,
    MIN_REPLICATION_INTERVAL, MAX_REPLICATION_INTERVAL,
    AUTOSAVE_EVERY, ENV_DRIFT_INTERVAL
)


def main():
    shared_replay  = make_replay_buffer()
    agents         = {a: ARIAAgent(a, replay=shared_replay) for a in INITIAL_AGENTS}
    env            = Environment(list(agents.keys()))
    channel        = CommunicationChannel()
    vis            = Visualiser()
    help_mon       = HelpMonitor()
    plateau_mon    = PlateauMonitor()
    lexicon_advisor = LexiconAdvisor()

    for agent_id in agents:
        help_mon.register_agent(agent_id)
        plateau_mon.register(agent_id)

    all_ids_ever        = list(INITIAL_AGENTS)
    generation          = 0
    last_replication_ep = 0

    print('=' * 64)
    print('  ARIA — Adaptive Reasoning and Interaction Agent')
    print('  Phase 2: Population Dynamics + Brain Evolution')
    print('=' * 64)
    print(f'  Founders        : {" + ".join(INITIAL_AGENTS)}')
    print(f'  Replication     : self-directed  '
          f'(min {MIN_REPLICATION_INTERVAL} eps / max {MAX_REPLICATION_INTERVAL} eps)')
    print(f'  Help system     : active from episode {config.HELP_MIN_EPISODE}')
    print(f'  Env drift       : every {ENV_DRIFT_INTERVAL} episodes')
    print(f'  Lexicon         : {LEXICON_LOG_PATH}')
    print(f'  ESC to quit')
    print(f'  API budget     : cap ${config.API_BUDGET_CAP:.2f}/session  |  {cost_tracker.lifetime_str()}')
    print('=' * 64)

    # Lexicon Advisor toggle
    try:
        la = input('  Lexicon Advisor on? [Y/n]: ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        la = 'y'
    config.LEXICON_ADVISOR_ON = la not in ('n', 'no')
    print(f'  Lexicon Advisor : {"ON" if config.LEXICON_ADVISOR_ON else "OFF"}')

    # Help system toggle
    try:
        hs = input('  Help system on?     [Y/n]: ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        hs = 'y'
    config.HELP_SYSTEM_ON = hs not in ('n', 'no')
    print(f'  Help system     : {"ON" if config.HELP_SYSTEM_ON else "OFF"}\n')

    start_episode = 1
    meta, npz_data = save_system.prompt_resume()
    if meta is not None:
        (agents, channel, generation, last_replication_ep,
         all_ids_ever, plateau_history, start_episode) = save_system.restore(meta, npz_data, shared_replay)
        for agent_id in agents:
            help_mon.register_agent(agent_id)
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

            # Replication check (external monitor, then agent self-assessment)
            should_repl, repl_reason = plateau_mon.should_replicate(
                agents, episode, last_replication_ep
            )
            if not should_repl:
                should_repl, repl_reason = plateau_mon.check_agent_initiated(
                    agents, episode, last_replication_ep
                )
                if should_repl:
                    initiator = repl_reason.split()[0]
                    if initiator in agents:
                        agents[initiator].record_replication_request(episode)
            if should_repl:
                print(f'\n  [Gen {generation}] Replication at episode {episode} '
                      f'— {repl_reason}')
                new_agents, new_channel, summary = replicate(
                    agents, channel, episode, set(all_ids_ever),
                    shared_replay=shared_replay
                )
                all_ids_ever.append(summary['child_id'])
                generation          += 1
                last_replication_ep  = episode

                # Deregister retired agent
                plateau_mon.deregister(summary['retired_id'])

                agents  = new_agents
                channel = new_channel

                help_mon.register_agent(summary['child_id'])
                plateau_mon.register(summary['child_id'])

                env.reset(agent_ids=list(agents.keys()))
                vis.notify_replication(summary)

                hp = summary['child_hyperparams']
                print(f'  Born    : {summary["child_id"]}')
                print(f'  Retired : {summary["retired_id"]} '
                      f'(reward {summary["retired_total_reward"]:.1f})')
                print(f'  Survived: {summary["surviving_id"]} '
                      f'(reward {summary["surviving_total_reward"]:.1f})')
                print(f'  Weights : {summary["weights"]}')
                print(f'  Hyperparams : lr={hp["lr"]}  '
                      f'layers={hp["n_layers"]}  act={hp["activation"]}  '
                      f'skip={hp["use_skip"]}\n')

            # Help system check
            if config.HELP_SYSTEM_ON:
                struggling = help_mon.check_all(agents, episode)
                for agent_id, reason in struggling:
                    agent = agents[agent_id]
                    print(f'\n  [Help] {agent_id} struggling ({reason}) at episode {episode}')
                    help_mon.run_help_cycle(agent_id, agent, reason, episode, config)

            # Episode setup
            states           = env.reset(agent_ids=list(agents.keys()))
            ep_rewards       = {a: 0.0 for a in agents}
            ep_coord_any     = False
            last_signal_step = {a: -(config.SIGNAL_WINDOW + 1) for a in agents}

            for step in range(MAX_STEPS_PER_EPISODE):

                actions      = {}
                signals_sent = {a: None for a in agents}

                for agent_id, agent in agents.items():
                    action = agent.select_action(states[agent_id])
                    actions[agent_id]      = action
                    signals_sent[agent_id] = ARIAAgent.get_signal_from_action(action)

                for agent_id, sig in signals_sent.items():
                    if sig is not None:
                        last_signal_step[agent_id] = step

                next_states, rewards, done, info, signals_received = env.step(
                    actions, signals_sent
                )

                if info['coord_achieved']:
                    ep_coord_any = True
                    for agent_id in agents:
                        if step - last_signal_step[agent_id] <= config.SIGNAL_WINDOW:
                            rewards[agent_id] += config.SIGNAL_REWARD

                # Record individual signals and compound pairs
                active_sigs = []
                for agent_id, sig in signals_sent.items():
                    if sig is not None:
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

                for agent_id, agent in agents.items():
                    agent.update(
                        states[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_states[agent_id],
                        info['coord_achieved'],
                    )
                    ep_rewards[agent_id] += rewards[agent_id]

                states = next_states

                vis.render(env, agents, channel, episode, step,
                           ep_rewards, generation, all_ids_ever)

                if done:
                    break

            # End of episode
            for agent_id, agent in agents.items():
                agent.decay_epsilon()
                agent.end_episode()
                partner_ids = [aid for aid in agents if aid != agent_id]
                agent.update_reputation(partner_ids, ep_coord_any)
                help_mon.record_episode(agent_id, ep_rewards[agent_id], ep_coord_any)
                plateau_mon.record(agent_id, ep_rewards[agent_id])

            vis.record_episode(ep_rewards)

            # Proactive lexicon analysis every 100 episodes
            lexicon_advisor.maybe_advise(channel, agents, episode, config)

            # Console summary every 25 episodes
            if episode % 25 == 0:
                eps      = list(agents.values())[0].epsilon
                assigned = channel.assigned_count()
                compound = channel.compound_lexicon.crystallised_count()
                seqs     = channel.sequence_lexicon.crystallised_count()
                ids      = list(agents.keys())
                r_str    = '  '.join(f'{a.split("-")[1]}:{ep_rewards.get(a,0):.1f}'
                                      for a in ids)
                roles    = '/'.join(agents[a].role[0] for a in ids)
                goals    = '/'.join(agents[a].goal_label[:10] for a in ids)
                print(f'  Ep {episode:5d} | gen {generation} | '
                      f'eps {eps:.4f} | {r_str} | '
                      f'lex {assigned}/4 cmp {compound} seq {seqs} | '
                      f'roles [{roles}] | goals [{goals}] | '
                      f'api ${cost_tracker.session_cost:.4f}')

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
    print(f'  Lexicon assigned    : {channel.assigned_count()}/4')
    print(f'  Compounds           : {channel.compound_lexicon.crystallised_count()}')


if __name__ == '__main__':
    main()
