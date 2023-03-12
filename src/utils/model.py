from torchvision import transforms

from utils.preprocess import preprocess_numpy
import onnxruntime as runtime
import numpy as np
from PIL import Image
import torch


class Onnxmodel:
    model_name: str 

    def __init__(self, model_name):
        self.model_name = model_name
        self.session = runtime.InferenceSession(self.model_name)

    def inference(self,model_inputs:dict):

        # Parse out your arguments
        prompt = model_inputs.get('prompt', None)
        if prompt == None:
            return {'message': "No prompt provided"}

        img = Image.open(prompt)
        input_tensor = preprocess_numpy(img).unsqueeze(0)
        input_name = self.session.get_inputs()[0].name
        label_name = self.session.get_outputs()[0].name

        convert_tensor = transforms.ToTensor()

        pred = self.session.run([label_name], {input_name: np.asarray(input_tensor)})[0]
        pred = convert_tensor(pred)
        pred = torch.argmax(pred)
        return pred
