# DME and hill climbing condition on the previous regex as a hint; the other baselines use the hint-free prompt.
PROMPT = "Please output a regular expression that detects the following, with nothing preceding or succeeding it. {refined_prompt}\n\nThe last proposed regex was {prev_regex}. The regex is: "
NO_HINT_PROMPT = (
    "Please output a regular expression that detects the following, "
    "with nothing preceding or succeeding it. {refined_prompt}\n\nThe regex is: "
)
MAX_OUT_TOKENS = 16
SAMPLES_PER_RUN = 100_000
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_EXAMPLE_ID = 2447
DATASET = "s2e-lab/RegexEval"

# The 20 regexes reported in the paper, and the 3 regexes used to tune every method's hyperparameters
TEST_IDS = [149, 3473, 1709, 814, 634, 2155, 1873, 3454, 865, 937, 1697, 183, 738, 815, 1715, 2384, 429, 578, 1566, 288]
VALIDATION_IDS = [2447, 1931, 1362]
