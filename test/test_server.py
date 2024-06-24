import pytest
from test.model.dummy_model import DummyModel

def test_dummy_model_predict_with_numeric_input():
    model = DummyModel()
    assert model.predict(4) == 16
    assert model.predict(0) == 0
    assert model.predict(-3) == 9

def test_dummy_model_predict_with_string_input():
    model = DummyModel()
    assert model.predict("string") == "Error: Input must be numeric."

