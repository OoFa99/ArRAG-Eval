def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        help="Run tests that require a live Ollama and Qdrant.",
    )
