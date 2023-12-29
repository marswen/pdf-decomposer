import os
import re
import cv2
import fitz
import json
import shutil
import tempfile
import numpy as np
from PIL import Image
from loguru import logger
from abc import ABC, abstractmethod


class PDFDecomposer(ABC):
    def __init__(self):
        self.temp_dir = None

    def _load_file(self, file_path):
        scale = 2.5
        max_length = 2000
        pages = list()
        logger.info("Loading file...")
        with fitz.open(file_path) as pdf:
            for page_number, page in enumerate(pdf):
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                if pix.width > max_length or pix.height > max_length:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                pages.append({'page_no': page_number, 'width': pix.width, 'height': pix.height, 'img': img})
        return pages

    @abstractmethod
    def _analyze_layout(self, file_path):
        pass

    @staticmethod
    def _area_of(left_top, right_bottom):
        """Compute the areas of rectangles given two corners.
        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.
        Returns:
            area (N): return the area.
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    @staticmethod
    def _reorganize_page(page, max_block_distance=3, max_line_overlap=3, max_span_distance=5):
        line_dist = [r[i + 1]['bbox'][1] - r[i]['bbox'][3] for r in page for i in range(len(r) - 1)
                     if r[i + 1]['bbox'][1] > r[i]['bbox'][3]]
        if len(line_dist) > 0:
            major_line_gap = max(set(line_dist), key=line_dist.count)
            max_block_distance = max(max_block_distance, major_line_gap)
        doc2 = list()
        page2 = list()
        prev_box = None
        for block1 in page:
            block2 = list()
            y0 = None
            y1 = None
            x0 = None
            x1 = None
            line2 = list()
            for s in block1:
                if y0 is not None:
                    if (s['bbox'][1] <= y0 < s['bbox'][3] - max_line_overlap
                            or s['bbox'][1] + max_line_overlap < y1 <= s['bbox'][3]
                            or y0 <= s['bbox'][1] < y1 - max_line_overlap
                            or y0 + max_line_overlap < s['bbox'][3] <= y1
                            or s['bbox'][2] - max_line_overlap <= x0 <= s['bbox'][2] + max_span_distance
                            or x1 - max_line_overlap <= s['bbox'][0] <= x1 + max_line_overlap
                    ):
                        if len(line2) > 0 and s['size'] == line2[-1]['size'] and s['font'] == line2[-1]['font']:
                            box = tuple([max(x) for x in zip(s['bbox'], line2[-1]['bbox'])])
                            if s['bbox'][0] < line2[-1]['bbox'][0]:
                                line2[-1] = {'text': s['text'] + ' ' + line2[-1]['text'],
                                             'size': s['size'], 'font': s['font'], 'bbox': box}
                            else:
                                line2[-1] = {'text': line2[-1]['text'] + ' ' + s['text'],
                                             'size': s['size'], 'font': s['font'], 'bbox': box}
                        else:
                            line2.append(s)
                    else:
                        block2.append(sorted(line2, key=lambda x: x['bbox'][0]))
                        line2 = [s]
                else:
                    line2.append(s)
                y0 = s['bbox'][1]
                y1 = s['bbox'][3]
                x0 = s['bbox'][0]
                x1 = s['bbox'][2]
            if len(line2) > 0:
                block2.append(sorted(line2, key=lambda x: x['bbox'][0]))
            block_boxes = [s['bbox'] for line in block2 for s in line]
            curr_box = [min([x[0] for x in block_boxes]), min([x[1] for x in block_boxes]),
                        max([x[2] for x in block_boxes]), max([x[3] for x in block_boxes])]
            if prev_box is not None:
                if (curr_box[0] > prev_box[2] + max_block_distance or
                        prev_box[0] > curr_box[2] + max_block_distance or
                        curr_box[1] > prev_box[3] + max_block_distance or
                        prev_box[1] > curr_box[3] + max_block_distance):
                    page2.append(block2)
                else:
                    project_lr_dist = max(0, curr_box[0] - prev_box[2])
                    project_rl_dist = max(0, prev_box[0] - curr_box[2])
                    project_tb_dist = max(0, curr_box[1] - prev_box[3])
                    project_bt_dist = max(0, prev_box[1] - curr_box[3])
                    if max(project_lr_dist, project_rl_dist, project_tb_dist, project_bt_dist) > max_block_distance:
                        page2.append(block2)
                    else:
                        page2[-1].extend(block2)
            else:
                page2.append(block2)
            prev_box = curr_box
        doc2.append(page2)
        page3 = list()
        for block2 in page2:
            block3 = list()
            for line2 in block2:
                if re.search('bold', line2[0]['font'], re.I) is not None:
                    if len(block3) > 0:
                        page3.append(block3)
                    for i_s, s in enumerate(line2[::-1]):
                        if re.search('bold', s['font'], re.I) is not None:
                            bold_end = i_s
                            break
                    page3.append([line2[: len(line2) - bold_end]])
                    block3 = []
                    if len(line2[len(line2) - bold_end:]) > 0:
                        block3.append(line2[len(line2) - bold_end:])
                    continue
                if len(block3) > 0:
                    if min([x['size'] for x in line2]) > max([x['size'] for line in block3 for x in line]) or \
                            min([x['size'] for line in block3 for x in line]) > max([x['size'] for x in line2]):
                        page3.append(block3)
                        block3 = []
                block3.append(line2)
            if len(block3) > 0:
                page3.append(block3)
        return page3

    def _extract_elements(self, file_path, layout_results):
        bound_buffer = 5
        overlap_threshold = 0.8
        doc = fitz.Document(file_path)
        structured_data = list()
        logger.info('Extracting elements...')
        for layout_page, doc_page in zip(layout_results, doc):
            components = doc_page.get_text('dict', flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            height = components['height']
            scale = layout_page['width'] / components['width']
            header_line = 0
            footer_line = components['height']
            header_boxes = [x for x in layout_page['elements'] if x['type'] == 'header']
            if len(header_boxes) > 0:
                header_line = max([x['bbox'][3] / scale for x in header_boxes]) + bound_buffer
            footer_boxes = [x for x in layout_page['elements'] if x['type'] == 'footer']
            if len(footer_boxes) > 0:
                footer_line = min([x['bbox'][1] / scale for x in footer_boxes]) - bound_buffer
            table_boxes = [x for x in layout_page['elements'] if x['type'] == 'table']
            figure_boxes = [x for x in layout_page['elements'] if x['type'] == 'figure']
            table_figure_bboxes = np.array([x['bbox'] for x in table_boxes + figure_boxes]) / scale
            header_regions = [x for x in components['blocks'] if x['bbox'][3] <= header_line]
            footer_regions = [x for x in components['blocks'] if x['bbox'][1] >= footer_line]
            for region in header_regions:
                for line in region['lines']:
                    for span in line['spans']:
                        structured_data.append({'type': 'header', 'page_no': layout_page['page_no'],
                                                'bbox': line['bbox'],
                                                'res': {'text': span['text'], 'size': span['size'],
                                                        'font': span['font']}})
            text_regions = [x for x in components['blocks'] if
                            header_line < x['bbox'][1] and x['bbox'][3] < footer_line]
            text_info = list()
            table_figure_record = list()
            for region in text_regions:
                if len(table_figure_bboxes) > 0:
                    region_box = np.array(region['bbox'])
                    overlap_left_top = np.maximum(region_box[..., :2], table_figure_bboxes[..., :2])
                    overlap_right_bottom = np.minimum(region_box[..., 2:], table_figure_bboxes[..., 2:])
                    overlap_area = self._area_of(overlap_left_top, overlap_right_bottom)
                    area0 = self._area_of(region_box[..., :2], region_box[..., 2:])
                    overlap_boxes = np.where(overlap_area / area0 > overlap_threshold)[0]
                    if len(overlap_boxes) > 0:
                        table_figure_idx = overlap_boxes[0]
                        if table_figure_idx not in table_figure_record:
                            if table_figure_idx < len(table_boxes):
                                structured_data.append(table_boxes[table_figure_idx])
                            else:
                                structured_data.append(figure_boxes[table_figure_idx - len(table_boxes)])
                            table_figure_record.append(table_figure_idx)
                        continue
                region_info = list()
                for line in region['lines']:
                    for span in line['spans']:
                        if len(span['text'].strip()) > 0:
                            region_info.append({'text': span['text'],
                                                'size': span['size'],
                                                'font': span['font'],
                                                'bbox': span['bbox']})
                text_info.append(region_info)
            structured_data.extend([x for i, x in enumerate(table_boxes) if i not in table_figure_record])
            structured_data.extend([x for i, x in enumerate(figure_boxes)
                                    if i + len(table_boxes) not in table_figure_record])
            reorg_text_info = self._reorganize_page(text_info)
            for para in reorg_text_info:
                bboxes = np.array([span['bbox'] for line in para for span in line])
                para_bbox = np.concatenate((bboxes[..., :2].min(axis=0), bboxes[..., 2:].max(axis=0))).tolist()
                if re.search('[\u4e00-\u9fa5]', ''.join([span['text'] for line in para for span in line])) is None:
                    text = ' '.join([''.join([span['text'] for span in line]) for line in para])
                else:
                    text = ''.join([''.join([span['text'] for span in line]) for line in para])
                size = max([span['size'] for line in para for span in line])
                structured_data.append({'type': 'text', 'page_no': layout_page['page_no'], 'bbox': para_bbox,
                                        'res': {'text': text, 'size': size, 'font': para[0][0]['font']}})
            for region in footer_regions:
                for line in region['lines']:
                    for span in line['spans']:
                        structured_data.append(
                            {'type': 'footer', 'page_no': layout_page['page_no'], 'bbox': line['bbox'],
                             'res': {'text': span['text'], 'size': span['size'], 'font': span['font']}})
        text_regions = [x for x in structured_data if x['type'] == 'text']
        next_size = [region['res']['size'] for region in text_regions][1:]
        for i, region in enumerate(text_regions[:-1]):
            if (region['res']['size'] > next_size[i] or
                    re.search('bold', region['res']['font'], re.I) is not None):
                char_search = re.search('[a-zA-Z\u4e00-\u9fa5]+', region['res']['text'].replace(' ', ''))
                if char_search is not None and len(char_search.group()) > 1:
                    region['type'] = 'subtitle'
        candidates = [x for x in text_regions if x['type'] == 'subtitle' and
                      x['page_no'] < 2 and
                      x['bbox'][1] < height / 2]
        candidates = [x for x in candidates if len(x['res']['text'].split(' ')) > 3 and
                      x['bbox'][2] - x['bbox'][0] > x['bbox'][3] - x['bbox'][1]]
        candidates = sorted(candidates, key=lambda x: x['res']['size'], reverse=True)
        candidates[0]['type'] = 'title'
        with open(os.path.join(self.temp_dir.name, 'structuredData.json'), 'w') as f:
            json.dump({'metadata': doc.metadata, 'elements': structured_data}, f, indent=4)

    def _save_result(self, file_path, output_dir):
        output_file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_abs_path = os.path.abspath(os.path.join(output_dir, output_file_name+'.zip'))
        output_dir = os.path.dirname(output_abs_path)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(output_abs_path):
            shutil.make_archive(output_file_name, 'zip', self.temp_dir.name)
            return
        raise Exception(f"Output file {output_file_name+'.zip'} exists")

    def __call__(self, file_path, output_dir, *args, **kwargs):
        self.temp_dir = tempfile.TemporaryDirectory()
        layout_results = self._analyze_layout(file_path)
        self._extract_elements(file_path, layout_results)
        self._save_result(file_path, output_dir)
