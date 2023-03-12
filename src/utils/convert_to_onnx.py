from .pytorch_model import Classifier, BasicBlock
from .preprocess import preprocess_numpy
import torch 
from PIL import Image

def convert_pytorch_to_onnx(pytorch_model_input : str = "./models/pytorch_model_weights.pth", img_path : str = "./n01440764_tench.jpeg", onnx_model_name : str = "onnx_pytorch.onnx"):
    pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
    pytorch_model.load_state_dict(torch.load(pytorch_model_input))

    pytorch_model.eval()

    img = Image.open(img_path)
    input_tensor = preprocess_numpy(img).unsqueeze(0) 
    
    torch.onnx.export(pytorch_model, input_tensor, './models/'+onnx_model_name, input_names=['input_tensor'], output_names=['output_tensor'])
