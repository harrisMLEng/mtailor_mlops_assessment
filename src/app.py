from utils.model import Onnxmodel


def init(model_name : str = "./models/onnx_pytorch.onnx"):
    global onnx 
    onnx = Onnxmodel(model_name)

def inference(model_inputs:dict):
    global onnx

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    pred = onnx.inference(prompt)
    return pred
