# pdf-decomposer

For the convenience of analysis, sometimes it is necessary to extract the content from PDF document. 
This project uses local tools to extract images, tables, and text from PDF file.


### Install

Install appropriate version of PaddlePaddle and PyTorch. Here is an example using CUDA 11.8.
```
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-chi-sim

pip install paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch torchvision torchaudio -i https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install from source code.
```
pip install .
```
or
```
pip install pdf-decomposer
```

If you are unable to download the layout-parser model automatically, please download it yourself based on the link provided in the error message, and then specify the local config_path and model_path.


### Solution based on PaddleOCR

Use layout analysis and table structure model in PaddleOCR ppstructure to recognize layout elements and reconstruct tables.
Then use PyMuPDF to parse text and organize paragraph.

```
import pdf_decomposer

decomposer = pdf_decomposer.PaddlePDFDecomposer()
decomposer(example_file, output_dir)
```


### Alternative local solution

Compared to the Paddle solution, use layout-parser to conduct layout analysis,
and then search adaptive thresholds with genetic algorithm to reconstruct tables.

```
import pdf_decomposer

decomposer = pdf_decomposer.OpenPDFDecomposer()
decomposer(example_file, output_dir)
```


### Solution based on Adobe service

Call Acrobat PDF Services API to extract various elements from PDF files. 
Please apply for the API key [here](https://acrobatservices.adobe.com/dc-integration-creation-app-cdn/main.html?api=pdf-extract-api#). 
Make sure to set api key before running.

```
export ADOBE_CLIENT_ID=<YOUR CLIENT ID>
export ADOBE_CLIENT_SECRET=<YOUR CLIENT SECRET>
```

```
import pdf_decomposer

decomposer = pdf_decomposer.AdobePDFDecomposer()
decomposer(example_file, output_dir)
```
