import os
import cv2
from tqdm import tqdm
from loguru import logger
from pdf_decomposer import rebuild_table
from pdf_decomposer.base_decomposer import PDFDecomposer
from layoutparser.models import Detectron2LayoutModel
from layoutparser.ocr import TesseractAgent, TesseractFeatureType


class OpenPDFDecomposer(PDFDecomposer):
    def __init__(self, config_path=None, model_path=None):
        super().__init__()
        try:
            import torch
        except ImportError:
            logger.error('PyTorch is not installed.')
        try:
            import torch
        except ImportError:
            logger.error(
                "Detectron2 is not installed. "
                "Try following: pip install 'git+https://github.com/facebookresearch/detectron2.git' "
            )
        label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        if config_path is not None and model_path is not None:
            if os.path.exists(config_path) and os.path.exists(model_path):
                self.layout_model = Detectron2LayoutModel(config_path=config_path,
                                                          model_path=model_path, label_map=label_map)
            else:
                raise Exception("Detectron2 model path doesn't exist.")
        else:
            config_path = 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config'
            self.layout_model = Detectron2LayoutModel(config_path=config_path, label_map=label_map)
        self.ocr_agent = TesseractAgent('eng+chi_sim')

    def _analyze_layout(self, file_path):
        pages = self._load_file(file_path)
        logger.info("Analyzing layout...")
        layout_results = list()
        for idx in tqdm(range(len(pages))):
            layout_prediction = self.layout_model.detect(pages[idx]['img'])
            # Needs plenty of postprocessing
            layout_prediction = [x.to_dict() for x in layout_prediction if x.score > 0.5]
            layout_prediction = sorted(layout_prediction, key=lambda x: (x['y_1'], x['x_1']))
            for box in layout_prediction:
                box['page_no'] = idx
                box.pop('block_type')
            layout_results.append({'page_no': pages[idx]['page_no'],
                                   'img': pages[idx]['img'],
                                   'width': pages[idx]['width'],
                                   'height': pages[idx]['height'],
                                   'elements': layout_prediction})
        figure_dir = os.path.join(self.temp_dir.name, 'figures')
        table_dir = os.path.join(self.temp_dir.name, 'tables')
        os.makedirs(figure_dir)
        os.makedirs(table_dir)
        img_idx = 0
        logger.info('Extracting tables and figures...')
        for page in tqdm(layout_results):
            for region in page['elements']:
                roi_img = page['img'][int(region['y_1']): int(region['y_2']), int(region['x_1']): int(region['x_2']), :]
                if region['type'].lower() == 'table':
                    region['img_idx'] = img_idx
                    img_path = os.path.join(table_dir, '{}.jpg'.format(img_idx))
                    cv2.imwrite(img_path, roi_img)
                    excel_path = os.path.join(table_dir, '{}.xlsx'.format(img_idx))
                    ocr_prediction = self.ocr_agent.detect(roi_img, return_only_text=False,
                                                           agg_output_level=TesseractFeatureType.WORD)
                    ocr_prediction = [x.to_dict() for x in ocr_prediction if len(x.text.strip()) > 0 and x.score > 0]
                    text_boxes = [[x['x_1'], x['y_1'], x['x_2'], x['y_2']] for x in ocr_prediction]
                    text_recs = [x['text'] for x in ocr_prediction]
                    rebuild_table.parse(text_boxes, text_recs, excel_path)
                    img_idx += 1
                elif region['type'].lower() == 'figure':
                    region['img_idx'] = img_idx
                    img_path = os.path.join(figure_dir, '{}.jpg'.format(img_idx))
                    cv2.imwrite(img_path, roi_img)
                    img_idx += 1
        return layout_results
