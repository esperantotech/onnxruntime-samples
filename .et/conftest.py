import pytest

def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption("--long", action="store_true", help="Run long tests with additional combinations.", default=False)

# Hook to do stuff before runnign tests
def pytest_configure(config):
    pass

# Hook to cleanup after test run
def pytest_unconfigure(config):
    pass

