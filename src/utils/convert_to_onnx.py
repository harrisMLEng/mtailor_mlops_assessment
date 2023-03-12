from .pytorch_model import Classifier, BasicBlock
from .preprocess import preprocess_numpy
import torch 
from PIL import Image

def convert_pytorch_to_onnx(pytorch_model : Classifier, img : Image, onnx_model_name : str):
    pytorch_model.load_state_dict(torch.load("./pytorch_model_weights.pth"))

    pytorch_model.eval()

    img = Image.open("./n01440764_tench.jpeg")
    input_tensor = preprocess_numpy(img).unsqueeze(0) 
    
    torch.onnx.export(pytorch_model, input_tensor, './models/'+onnx_model_name, input_names=['input_tensor'], output_names=['output_tensor'])
