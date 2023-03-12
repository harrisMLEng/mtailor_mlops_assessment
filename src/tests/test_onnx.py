import pytest
from utils.model import Onnxmodel
import torch 
@pytest.mark.parametrize('model_input',[
    ({'prompt':'./data/n01667114_mud_turtle.JPEG'})
])
def test_onnx(model_input):
    onnxmodel =  Onnxmodel('./models/onnx_pytorch.onnx')
    actual = onnxmodel.inference(model_input)

    assert actual == torch.tensor(35)