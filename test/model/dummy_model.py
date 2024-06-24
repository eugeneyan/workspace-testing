class DummyModel:
    def predict(self, input_data):
        if isinstance(input_data, (int, float)):
            return input_data ** 2
        else:
            return "Error: Input must be numeric."
