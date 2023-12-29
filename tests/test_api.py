
example_file = './example.pdf'
output_dir = './'


def paddle_example():
    import pdf_decomposer
    decomposer = pdf_decomposer.PaddlePDFDecomposer()
    decomposer(example_file, output_dir)


def open_alternative_example():
    import pdf_decomposer
    config_path = './pdf_decomposer/models/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config.yaml'
    model_path = './pdf_decomposer/models/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth'
    decomposer = pdf_decomposer.OpenPDFDecomposer(config_path, model_path)
    decomposer(example_file, output_dir)


def abobe_service_example():
    """
    Make sure to load api key before running:

    export ADOBE_CLIENT_ID=<YOUR CLIENT ID>
    export ADOBE_CLIENT_SECRET=<YOUR CLIENT SECRET>
    """
    import pdf_decomposer
    decomposer = pdf_decomposer.AdobePDFDecomposer()
    decomposer(example_file, output_dir)
