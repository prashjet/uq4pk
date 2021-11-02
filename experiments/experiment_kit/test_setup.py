class TestSetup:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        # make the test name
        name = ""
        for param_value in parameters.values():
            name = f"{name}_{param_value}"
        self.name = name