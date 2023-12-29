import os
import cv2
from tqdm import tqdm
from loguru import logger
from pdf_decomposer.base_decomposer import PDFDecomposer
from paddleocr import PPStructure
from paddleocr.ppstructure.table.tablepyxl import tablepyxl


class PaddlePDFDecomposer(PDFDecomposer):
    def __init__(self):
        super().__init__()
        try:
            import paddle
        except ImportError:
            logger.error('PaddlePaddle is not installed.')
        self.table_engine = PPStructure(show_log=False)

    def _analyze_layout(self, file_path):
        pages = self._load_file(file_path)
        logger.info("Analyzing layout...")
        layout_results = list()
        for idx in tqdm(range(len(pages))):
            result = self.table_engine(pages[idx]['img'], return_ocr_result_in_table=True)
            result = sorted(result, key=lambda x: (x['bbox'][1], x['bbox'][0]))
            for box in result:
                box['page_no'] = idx
                box.pop('img_idx')
            layout_results.append({'page_no': pages[idx]['page_no'],
                                   'width': pages[idx]['width'],
                                   'height': pages[idx]['height'],
                                   'elements': result})
        figure_dir = os.path.join(self.temp_dir.name, 'figures')
        table_dir = os.path.join(self.temp_dir.name, 'tables')
        os.makedirs(figure_dir)
        os.makedirs(table_dir)
        img_idx = 0
        logger.info('Extracting tables and figures...')
        for page in tqdm(layout_results):
            for region in page['elements']:
                roi_img = region.pop('img')
                if region['type'].lower() == 'table' and 'html' in region['res']:
                    region['img_idx'] = img_idx
                    img_path = os.path.join(table_dir, '{}.jpg'.format(img_idx))
                    cv2.imwrite(img_path, roi_img)
                    excel_path = os.path.join(table_dir, '{}.xlsx'.format(img_idx))
                    tablepyxl.document_to_xl(region['res']['html'], excel_path)
                    img_idx += 1
                elif region['type'].lower() == 'figure':
                    region['img_idx'] = img_idx
                    img_path = os.path.join(figure_dir, '{}.jpg'.format(img_idx))
                    cv2.imwrite(img_path, roi_img)
                    img_idx += 1
        return layout_results
