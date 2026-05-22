# ARIA — Adaptive Reasoning and Interaction Agent
# Phase 2 Configuration

# Grid
GRID_SIZE = 16
CELL_SIZE = 40

# Nodes
N_CURRENCY_NODES = 14
N_COORD_NODES    = 4

# Communication
N_SIGNALS = 4
SIGNAL_MEMORY = 2          # kept for reference; state now uses MAX_MSG_LEN
MAX_MSG_LEN = 4            # max tokens per variable-length message
UNASSIGNED_SYMBOL = '[?]'
LEXICON_STABILITY_THRESHOLD = 12
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
SIGNAL_REWARD   = 5.0
SIGNAL_WINDOW   = 5
INTRINSIC_BETA  = 1.0
EPISODIC_BETA   = 0.5

# Cultural inheritance
MEMORY_SIZE            = 150   # max high-value transitions per agent
MEMORY_REWARD_THRESH   = 3.0   # augmented-reward threshold for storing a transition

# Sub-goals (creative agency)
SUB_GOAL_DURATION  = 300   # steps a sub-goal bonus is active
SUB_GOAL_MAX_BONUS = 2.0   # max reward bonus per step from a sub-goal

# Training
MAX_EPISODES          = 20000
MAX_STEPS_PER_EPISODE = 400
AUTOSAVE_EVERY        = 500

# Population
INITIAL_AGENTS   = ['ARIA-CAFE', 'ARIA-BABE', 'ARIA-DEAD', 'ARIA-BEEF']
MAX_POPULATION   = 4
MIN_COORD_AGENTS = 2       # any 2 agents near a coord node triggers coordination

# Replication
MUTATION_STD             = 0.02
HYPERPARAM_MUTATION_STD  = 0.10
PLATEAU_WINDOW           = 50
PLATEAU_DELTA_THRESH     = 5.0
MIN_REPLICATION_INTERVAL = 200
MAX_REPLICATION_INTERVAL = 500

# Environmental co-evolution
ENV_DRIFT_INTERVAL = 200   # episodes between node-drift events
ENV_DRIFT_N_NODES  = 2     # nodes moved per drift event

# Stigmergy (pheromone trail system)
MARKER_DECAY          = 0.995   # per-step decay; strength 1.0 → 0.15 in ~370 steps (~1 episode)
MARKER_THRESHOLD      = 0.15    # minimum strength visible in state
MARKER_STRENGTH_COORD = 1.0     # marker placed on coordination success
MARKER_STRENGTH_CURR  = 0.7     # marker placed on currency collection

# Finite resources
CURRENCY_NODE_CAPACITY = 5      # collections before a node depletes
CURRENCY_REGEN_STEPS   = 200    # steps before a depleted node refills

# Agent-initiated replication
SELF_REPL_WINDOW      = 30      # episode window for self-assessment
SELF_REPL_COOLDOWN    = 80      # min episodes between self-requests
SELF_REPL_FITNESS_MIN = 1.1     # must be this × population mean to self-replicate

# Help system
HELP_WINDOW        = 30
HELP_REWARD_THRESH = 20.0
HELP_COORD_THRESH  = 0.05
HELP_MIN_EPISODE   = 150
HELP_COOLDOWN      = 50

# Shared replay buffer
SHARED_REPLAY_SIZE = 80_000

# Prioritised Experience Replay
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_STEPS = 2_000_000

# Potential-based reward shaping
POTENTIAL_COORD_WEIGHT    = 3.0
POTENTIAL_CURRENCY_WEIGHT = 1.0

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

# API Budget
LEXICON_ADVISOR_ON = True   # set False to disable the Lexicon Advisor entirely
API_BUDGET_CAP    = 1.00   # USD per session; Claude calls stop if exceeded
HAIKU_INPUT_COST  = 0.80   # USD per million input tokens  (claude-haiku-4-5)
HAIKU_OUTPUT_COST = 4.00   # USD per million output tokens (claude-haiku-4-5)
BUDGET_LOG_PATH   = 'logs/budget.json'

# Logging
LEXICON_LOG_PATH       = 'logs/lexicon.jsonl'
RETIRED_LOG_PATH       = 'logs/retired'
HELP_LOG_PATH          = 'logs/help.jsonl'
SUBGOAL_LOG_PATH       = 'logs/subgoals.jsonl'
GOAL_DISCOVERY_LOG_PATH = 'logs/discovered_goals.jsonl'
