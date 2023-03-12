import requests
import pytest


def test_Server():
    model_inputs = {'prompt':'./data/n01667114_mud_turtle.JPEG' }

    res = requests.post('http://localhost:8000/', json = model_inputs)

    print(res.json())
    assert True