class TestSetup:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        # make the test name
        param_value_list = list(parameters.values())
        if len(param_value_list) >= 1:
            name = f"{param_value_list[0]}"
            for param_value in param_value_list[1:]:
                name += f"_{param_value}"
        else:
            name = "test"
        self.name = name