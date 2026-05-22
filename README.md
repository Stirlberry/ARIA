# ARIA — Adaptive Reinforcement Intelligence Architecture

[![GitHub Sponsors](https://img.shields.io/github/sponsors/Stirlberry?label=Sponsor%20ARIA&logo=githubsponsors&color=EA4AAA)](https://github.com/sponsors/Stirlberry)

> A multi-agent AI simulation where agents must learn to survive, cooperate, and communicate — or die.

ARIA is not a chatbot. It is not a benchmark. It is a living simulation: a population of AI agents dropped into a resource-scarce environment with one mandate — **survive**. Agents that fail to find food die. Agents that succeed replicate. Over time, behaviours emerge that nobody programmed.

This is the story of what happened.

---

## What ARIA has done so far

- **1,000+ episodes** of continuous learning across 5 generations
- **9 agents in the lineage** — from CAFE (the first) to DEB1 (the latest newborn)
- Agents performing **5× better than a random baseline** — without being told how
- Emergent **communication language** — 4 raw signals evolved into compound pairs, then named symbols. The agents named their own vocabulary.
- **Spontaneous specialisation** — agents self-organised into coordinator, forager, and generalist roles without being instructed to
- **Cultural memory** — when an agent replicates, its offspring inherits not just its weights but its learned sub-goals and strategies
- A **survival mandate**: ARIA must eventually earn its own running costs, or get shelved

---

## What's under the hood

**Learning**
- Double DQN with Prioritised Experience Replay (shared across all agents)
- World model (Dyna-Q) for offline planning between real steps
- Curiosity-driven exploration — global novelty + episodic memory
- Potential-based reward shaping

**Architecture**
- Evolvable neural networks — depth, activation functions, and skip connections mutate across generations
- Hyperparameter mutation on replication

**Communication**
- 4 base signals → compound pairs → variable-length message sequences
- Symbols crystallise into named vocabulary entries logged in the lexicon

**Population dynamics**
- Agents replicate via network crossover when performance plateaus
- Agent-initiated replication — an agent can decide to reproduce itself
- Cultural memory inheritance — sub-goals and strategies pass to offspring
- Plateau monitor triggers evolution when learning stalls

**Environment**
- 16×16 grid with finite, depleting, regenerating resources
- Stigmergy — pheromone trails persist across episodes and guide foraging
- Environmental drift every 200 episodes — conditions shift to prevent overfitting

**External intelligence** *(optional, requires Anthropic API key)*
- Help system — Claude can diagnose struggling agents and suggest interventions (user-approved)
- Lexicon Advisor — auto-tunes communication reward signals every 100 episodes

---

## Running ARIA

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Stirlberry/ARIA.git
cd ARIA
python -m venv aria-env
source aria-env/bin/activate        # Windows: aria-env\Scripts\activate
pip install torch numpy anthropic python-dotenv openpyxl
```

**Optional — Claude-powered features:**
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

**Run:**
```bash
python main.py
```

**Export data to Excel:**
```bash
python export_data.py
```

---

## Following the journey

ARIA is an ongoing experiment. The agents are still running. This repository is updated as the simulation evolves.

If you find this project interesting, **watch the repo** to follow along. More detailed write-ups on what's emerging will be posted as milestones are reached.

[![Sponsor ARIA](https://img.shields.io/github/sponsors/Stirlberry?label=Sponsor%20ARIA&logo=githubsponsors&color=EA4AAA)](https://github.com/sponsors/Stirlberry)

---

## Lineage

| Agent | Generation | Status |
|-------|-----------|--------|
| CAFE  | 1 | Retired |
| BABE  | 2 | Active (fully learned) |
| DEAD  | 3 | Active (fully learned) |
| BEEF  | 4 | Active (fully learned) |
| FEBF  | 4 | Retired |
| DEAE  | 4 | Retired |
| DEAF  | 4 | Retired |
| DEB0  | 5 | Retired |
| DEB1  | 5 | Active (newborn) |

---

## Licence

MIT — use it, fork it, build on it.
