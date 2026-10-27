"""Failure options available in an AHA task YAML."""


def get_available_failures(config):
    return [failure for failure in config.get('failures', [])
            if failure.get('type') != 'dummy']
