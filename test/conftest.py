def pytest_configure(config):
    # Трассируемость по канону (docs/canon/REQUIREMENTS.md, раздел 12).
    config.addinivalue_line("markers", "req(*ids): ID требований из docs/canon/REQUIREMENTS.md")
    config.addinivalue_line("markers", "integration: прогон на реальных изображениях")
