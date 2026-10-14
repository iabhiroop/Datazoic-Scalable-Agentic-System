"""Prevent application imports from loading developer credentials during tests."""

import os

# Integration database settings are supplied explicitly by the test runner.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
