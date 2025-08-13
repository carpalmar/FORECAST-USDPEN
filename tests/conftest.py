import sys
from unittest.mock import MagicMock

# Mock heavy libraries that are not needed for most unit/integration tests
# to avoid installation issues in constrained environments.
MOCK_MODULES = [
    "torch",
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.models",
    "tensorflow.keras.layers",
    "tensorflow.keras.optimizers",
    "chronos",
    "chronos.chronos",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
