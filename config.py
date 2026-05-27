# ARIA-2 — Lewis Signaling Game Configuration
#
# Agents are split into two types (0 and 1). Each type can only see its own
# target nodes. Reward fires only when BOTH types occupy the same target tile
# simultaneously. Signals are the ONLY way for a sighted agent to guide a
# blind agent to the target — a proper Lewis signaling game.

# Grid
GRID_W    = 32
GRID_H    = 24
CELL_SIZE = 32

# Nodes
N_CURRENCY_NODES       = 48
N_TARGET_NODES_PER_TYPE = 12   # 12 visible to type-0, 12 visible to type-1 (24 total)

# Agent types
FOUNDER_TYPES = {
    'ARIA-CAFE': 0,
    'ARIA-BABE': 1,
    'ARIA-DEAD': 0,
    'ARIA-BEEF': 1,
}

# Communication
N_SIGNALS   = 16
MAX_MSG_LEN = 4
SENDER_ID_MAX             = 0xFFFF
UNASSIGNED_SYMBOL         = '[?]'
LEXICON_STABILITY_THRESHOLD = 12
COMPOUND_THRESHOLD          = 8
SEQUENCE_THRESHOLD          = 6

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
N_LAYERS_DEFAULT   = 2
ACTIVATION_OPTIONS = ['relu', 'tanh', 'gelu']

# World model + Dyna-Q
DYNA_STEPS         = 3
WORLD_MODEL_LR     = 5e-4
WORLD_MODEL_HIDDEN = 128
WORLD_MODEL_WARMUP = 200

# Rewards
REWARD_CURRENCY = 10.0
REWARD_COORD    = 30.0
REWARD_STEP     = -0.1
SIGNAL_WINDOW   = 5
INTRINSIC_BETA  = 1.0
EPISODIC_BETA   = 0.5

REWARD_REPRODUCE = 10.0
REWARD_DEATH     = -50.0

# Cultural inheritance
MEMORY_SIZE          = 150
MEMORY_REWARD_THRESH = 3.0

# Sub-goals
SUB_GOAL_DURATION  = 300
SUB_GOAL_MAX_BONUS = 2.0

# Training
MAX_EPISODES          = 20000
MAX_STEPS_PER_EPISODE = 500
AUTOSAVE_EVERY        = 50

# Population
INITIAL_AGENTS   = ['ARIA-CAFE', 'ARIA-BABE', 'ARIA-DEAD', 'ARIA-BEEF']
MAX_POPULATION   = 10
MIN_POPULATION   = 5
MIN_COORD_AGENTS = 2

# Energy
ENERGY_START         = 400
ENERGY_MAX           = 600
ENERGY_NEWBORN       = 300
ENERGY_DRAIN_LOW     = 0.3
ENERGY_DRAIN_MED     = 0.5
ENERGY_DRAIN_HIGH    = 0.7
ENERGY_FROM_CURRENCY = 50
ENERGY_FROM_CO       = 150

# Survival and reproduction
REPRODUCTION_THRESHOLD = 500
REPRODUCTION_COST      = 200
SPAWN_PAUSE_STEPS      = 60

# Ghost nodes
GHOST_NODE_ACCESSES = 1

# Replication / Monitor
MUTATION_STD             = 0.02
HYPERPARAM_MUTATION_STD  = 0.10
PLATEAU_WINDOW           = 25
PLATEAU_DELTA_THRESH     = 0.10
MIN_REPLICATION_INTERVAL = 200
MAX_REPLICATION_INTERVAL = 1000

# Environmental co-evolution
ENV_DRIFT_INTERVAL = 200
ENV_DRIFT_N_NODES  = 2

# Finite resources
CURRENCY_NODE_CAPACITY = 5
REGEN_MIN_STEPS = 100
REGEN_MAX_STEPS = 500

# Fog of war
FOG_RADIUS    = 5

# Communication ranges
CHATTER_RANGE = 1
SHOUT_RANGE   = FOG_RADIUS - 1

# Shared replay buffer
SHARED_REPLAY_SIZE = 80_000

# Prioritised Experience Replay
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_STEPS = 2_000_000

# Theory of Mind
TOM_LR               = 1e-3
TOM_POS_WEIGHT       = 20.0
TOM_INTENT_THRESHOLD = 0.5
TOM_BONUS_SCALE      = 1.0

# Internal goal discovery
GOAL_DISCOVERY_MIN_SAMPLES       = 100
GOAL_DISCOVERY_CRYSTALLISE_EVERY = 50
GOAL_DISCOVERY_MIN_LIFT          = 0.3
GOAL_DISCOVERY_BONUS             = 0.3

# Logging
LEXICON_LOG_PATH        = 'logs/lexicon.jsonl'
RETIRED_LOG_PATH        = 'logs/retired'
SUBGOAL_LOG_PATH        = 'logs/subgoals.jsonl'
GOAL_DISCOVERY_LOG_PATH = 'logs/discovered_goals.jsonl'
