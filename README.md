## Deploy Classification Neural Network on Serverless GPU platform of Banana De

# Setup 

<b>1. Download the repository </b>

``` $ git clone <repo name> ```

<b>2. After downloading navigate to te container ocr folder and run the dollowing commands </b>

``` 
$ python -m venv mtailor
$ source mtailor/bin/activate 
```
<b>3. Download Model weights into models/ directory</b>

```
https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0
```

## Development Environment Setup

<b>4. Install required dependencies</b>

``` 
pip install -r requirements.txt
```

<b>5. Convert Pytorch model to Onnx</b>

```
python src/utils/convert_to_onnx.py
```

<b>6. Running tests</b>

```
pytest src/tests/
```

