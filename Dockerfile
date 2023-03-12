# Must use a Cuda version 11+
FROM python:3.9-slim

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD ./src/server.py .

# Add your model weight files 
# (in this case we have a python script)
ADD ./models/onnx_pytorch.onnx ./src


# Add your custom app code, init() and inference()
ADD  ./src/app.py .

EXPOSE 8000

CMD python3 -u server.py
