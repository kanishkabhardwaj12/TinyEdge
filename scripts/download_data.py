import requests
import os

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
output_path = "data/cifar-10-python.tar.gz"

os.makedirs("data", exist_ok=True)

print("Downloading CIFAR-10...")
response = requests.get(url, stream=True)
with open(output_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
print("Download complete!")
