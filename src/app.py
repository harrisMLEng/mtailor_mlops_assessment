from torchvision import transforms

from utils.preprocess import preprocess_numpy
import onnxruntime as runtime
import numpy as np
from PIL import Image

def init(model_name : str = "./models/pytorch_onnx.onnx"):
    global session
    session = runtime.InferenceSession(model_name)

def inference(model_inputs:dict):
    global session

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    img = Image.open(prompt)
    input_tensor = preprocess_numpy(img).unsqueeze(0)
    label_name = session.get_outputs()[0].name

    convert_tensor = transforms.ToTensor()

    pred = session.run([label_name], {input_tensor: np.asarray(input_tensor)})[0]
    pred = convert_tensor(pred)

    return pred


