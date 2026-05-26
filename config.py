# ARIA — Adaptive Reasoning and Interaction Agent
# Phase 3 Configuration

# Grid
GRID_SIZE = 24
CELL_SIZE = 32

# Nodes
N_CURRENCY_NODES = 24
N_COORD_NODES    = 12

# Communication
N_SIGNALS = 16
MAX_MSG_LEN = 4            # max tokens per variable-length message
SENDER_ID_MAX = 0xFFFF     # normalisation ceiling for sender identity channel (Option 2)
UNASSIGNED_SYMBOL = '[?]'
LEXICON_STABILITY_THRESHOLD = 12   # coordination successes required to crystallise a signal
COMPOUND_THRESHOLD  = 8    # coordinated successes before a signal pair crystallises
SEQUENCE_THRESHOLD  = 6    # coordinated successes before a message sequence crystallises

# Q-Learning / DQN
LEARNING_RATE   = 1e-3
DISCOUNT_FACTOR = 0.99
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.995

# DQN
HIDDEN_SIZE        = 128
BATCH_SIZE         = 64
REPLAY_BUFFER_SIZE = 20000
MIN_REPLAY_SIZE    = 500
TARGET_UPDATE_FREQ = 200
TRAIN_FREQ         = 4

# Brain evolution
N_LAYERS_DEFAULT = 2
ACTIVATION_OPTIONS = ['relu', 'tanh', 'gelu']   # evolvable per-agent

# World model + Dyna-Q
DYNA_STEPS         = 3      # imagined Q-updates per real training step
WORLD_MODEL_LR     = 5e-4
WORLD_MODEL_HIDDEN = 128
WORLD_MODEL_WARMUP = 200    # real training steps before Dyna kicks in

# Rewards

REWARD_CURRENCY = 10.0
REWARD_COORD    = 30.0
REWARD_STEP     = -0.1
SIGNAL_WINDOW   = 5
INTRINSIC_BETA  = 1.0
EPISODIC_BETA   = 0.5

# Rewards — new events (Phase 3)
REWARD_REPRODUCE       =  10.0   # successful reproduction
REWARD_DEATH           = -50.0   # agent dies (energy hits zero)

# Cultural inheritance
MEMORY_SIZE            = 150   # max high-value transitions per agent
MEMORY_REWARD_THRESH   = 3.0   # augmented-reward threshold for storing a transition

# Sub-goals (creative agency)
SUB_GOAL_DURATION  = 300   # steps a sub-goal bonus is active
SUB_GOAL_MAX_BONUS = 2.0   # max reward bonus per step from a sub-goal

# Training
MAX_EPISODES          = 20000
MAX_STEPS_PER_EPISODE = 500
AUTOSAVE_EVERY        = 50

# Population
INITIAL_AGENTS   = ['ARIA-CAFE', 'ARIA-BABE', 'ARIA-DEAD', 'ARIA-BEEF']
MAX_POPULATION       = 10   # population ceiling
MIN_COORD_AGENTS     = 2    # any 2 agents on a CO node triggers coordination

# Energy system (Phase 3)
ENERGY_START         = 400   # starting energy for all new agents
ENERGY_MAX           = 600   # maximum energy cap — agents cannot exceed this
ENERGY_NEWBORN       = 300   # starting energy for agents born via reproduction
ENERGY_DRAIN_LOW     = 0.3   # passive energy drain per step (low drain rate)
ENERGY_DRAIN_MED     = 0.5   # passive energy drain per step (medium drain rate)
ENERGY_DRAIN_HIGH    = 0.7   # passive energy drain per step (high drain rate)
ENERGY_FROM_CURRENCY = 50    # energy gained from consuming a currency node
ENERGY_FROM_CO       = 150   # energy gained per agent from consuming a CO node

# Survival and reproduction (Phase 3)
REPRODUCTION_THRESHOLD = 500  # both agents must be above this energy level to reproduce
REPRODUCTION_COST      = 200  # energy deducted from each parent on reproduction

SPAWN_PAUSE_STEPS      = 60   # steps parents freeze at birth point before separating

# Ghost nodes — Tier 2 knowledge preservation (Phase 3)
GHOST_NODE_ACCESSES = 1       # times a ghost node can be accessed before it disappears

# Replication
MUTATION_STD             = 0.02
HYPERPARAM_MUTATION_STD  = 0.10
PLATEAU_WINDOW           = 50
PLATEAU_DELTA_THRESH     = 0.10   # fraction of agent's mean absolute reward (scaled)
MIN_REPLICATION_INTERVAL = 500
MAX_REPLICATION_INTERVAL = 1000

# Environmental co-evolution
ENV_DRIFT_INTERVAL = 200   # episodes between node-drift events
ENV_DRIFT_N_NODES  = 2     # nodes moved per drift event

# Finite resources
CURRENCY_NODE_CAPACITY = 5      # collections before a node depletes
REGEN_MIN_STEPS = 100           # minimum steps before a consumed node respawns at a new random position
REGEN_MAX_STEPS = 500           # maximum steps before a consumed node respawns

# Fog of war
FOG_RADIUS    = 5              # Chebyshev radius; nodes and agents beyond this are invisible — hard ceiling, no agent exceeds this

# Communication ranges (Phase 3)
CHATTER_RANGE = 1              # adjacent only (Chebyshev distance ≤ 1)
SHOUT_RANGE   = FOG_RADIUS - 1 # broadcast range = 4; less than fog so agents must close the gap

# Shared replay buffer
SHARED_REPLAY_SIZE = 80_000

# Prioritised Experience Replay
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_STEPS = 2_000_000

# Theory of Mind
TOM_LR               = 1e-3
TOM_POS_WEIGHT       = 20.0   # upweight rare coordination events in BCELoss
TOM_INTENT_THRESHOLD = 0.5
TOM_BONUS_SCALE      = 1.0

# Internal goal discovery
GOAL_DISCOVERY_MIN_SAMPLES       = 100   # min met/unmet samples before crystallising
GOAL_DISCOVERY_CRYSTALLISE_EVERY = 50    # episodes between discovery attempts
GOAL_DISCOVERY_MIN_LIFT          = 0.3   # min avg-reward lift over baseline
GOAL_DISCOVERY_BONUS             = 0.3   # fixed bonus for self-discovered goals

# Logging
LEXICON_LOG_PATH        = 'logs/lexicon.jsonl'
RETIRED_LOG_PATH        = 'logs/retired'
SUBGOAL_LOG_PATH        = 'logs/subgoals.jsonl'
GOAL_DISCOVERY_LOG_PATH = 'logs/discovered_goals.jsonl'
