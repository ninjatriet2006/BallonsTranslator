"""Document and plaintext import/export utilities for project translation."""
import json
import os
import os.path as osp
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import docx
import docx2txt
import numpy as np
import piexif
from docx import Document
from docx.shared import Inches

from ballontranslator.utils.system.io_utils import imread, imwrite, NumpyEncoder
from ballontranslator.utils.system.logger import logger as LOGGER
from ballontranslator.utils.fonts.fontformat import FontFormat, warn_ignored_legacy_effects
from ballontranslator.utils.text.textblock import TextBlock, normalize_textblock_effect_payload
from ballontranslator.utils.text.text_alpha_mask import AlphaBrushStroke, TextAlphaMask


page_start_pattern = re.compile(r'^###\s+', re.MULTILINE)
text_blkid_start_pattern = re.compile(r'^\d+\.', re.MULTILINE)


class TextBlkEncoder(NumpyEncoder):
    def default(self, obj):
        if isinstance(obj, TextBlock):
            return obj.to_dict()
        elif isinstance(obj, FontFormat):
            return obj.to_serializable_dict()
        elif isinstance(obj, (TextAlphaMask, AlphaBrushStroke)):
            return obj.to_serializable_dict()
        return NumpyEncoder.default(self, obj)


def write_jpg_metadata(imgpath: str, metadata="a metadata"):
    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(metadata, encoding='unicode')}}
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, imgpath)


def read_jpg_metadata(imgpath: str):
    exif_dict = piexif.load(imgpath)
    user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    bubdict = json.loads(user_comment)
    return bubdict


def parse_txt_translation(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = f.read()
    page_start = None
    page_list = []
    for matched in page_start_pattern.finditer(content):
        start, end = matched.span()
        if page_start is not None:
            page_list.append({'page_content': content[page_start: start]})
        page_start = start
    if page_start is not None:
        page_list.append({'page_content': content[page_start:]})

    for page_dict in page_list:
        page_content = page_dict['page_content']
        page_dict['page_name'] = page_start_pattern.sub('', page_content.split('\n')[0]).strip()
        blkid_start = blkid_end = None
        blk_list = []
        for matched in text_blkid_start_pattern.finditer(page_content):
            start, end = matched.span()
            if blkid_start is not None:
                blk_list.append(page_content[blkid_end: start].strip())
            blkid_start = start
            blkid_end = end
        if blkid_start is not None:
            blk_list.append(page_content[blkid_end:].strip())
        page_dict['blk_list'] = blk_list

    return page_list


def gen_ballon_cuts(cuts_dir: str, imgpath: str, blk_list: List[TextBlock], resize=True) -> Tuple[List[str], List[int]]:
    img = imread(imgpath)
    imgname = os.path.basename(imgpath)
    cuts_path_list = []
    cut_width_list = []
    for ii, blk in enumerate(blk_list):
        x, y, w, h = blk.bounding_rect()
        x, y = max(x, 0), max(y, 0)
        w = max(w, 1)
        h = max(h, 1)
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        cut_path = os.path.join(cuts_dir, f'{imgname}-{ii}.jpg')
        bub = img[y1:y2, x1:x2]
        max_width = 448

        if bub.shape[0] < 1 or bub.shape[1] < 1:
            emptyw = 60
            resized = np.full((emptyw, emptyw, 3), fill_value=0, dtype=np.uint8)
            width = emptyw
        else:
            scale_percent = min(1920 / img.shape[0], max_width / w)
            if scale_percent < 1:
                width = max(1, int(bub.shape[1] * scale_percent))
                height = max(1, int(bub.shape[0] * scale_percent))
                dim = (width, height)
                resized = cv2.resize(bub, dim, interpolation=cv2.INTER_AREA) if resize else bub
            else:
                width = w
                resized = bub

        imwrite(cut_path, resized, '.jpg')
        cuts_path_list.append(cut_path)
        cut_width_list.append(width)

    return cuts_path_list, cut_width_list


def export_project_doc(project, delete_tmp_folder=True, fin_page_signal=None):
    cuts_dir = os.path.join(project.directory, "bubcuts")
    if os.path.exists(cuts_dir):
        shutil.rmtree(cuts_dir)
    os.mkdir(cuts_dir)

    document = Document()
    style = document.styles['Normal']
    font = style.font
    target_font = 'Arial'
    font.name = target_font
    for pagename, blklist in project.pages.items():
        imgpath = os.path.join(project.directory, pagename)

        cuts_path_list, cut_width_list = gen_ballon_cuts(cuts_dir, imgpath, blklist)
        paragraph = document.add_paragraph(pagename)
        paragraph.style = document.styles['Normal']
        table = document.add_table(rows=len(cuts_path_list), cols=2, style='Table Grid')

        for index, (cut_path, width) in enumerate(zip(cuts_path_list, cut_width_list)):
            run = table.cell(index, 0).paragraphs[0].add_run()
            run.style.font.name = target_font
            blk: TextBlock = blklist[index]
            bubdict = vars(blk).copy()
            bubdict["imgkey"] = pagename
            bubdict["rich_text"] = ''
            bubdict["text"] = blk.get_text()
            write_jpg_metadata(cut_path, metadata=json.dumps(bubdict, ensure_ascii=False, cls=TextBlkEncoder))
            run.add_picture(cut_path, width=Inches(width / 96 * 0.85))
            table.cell(index, 1).text = bubdict["translation"]

        document.add_page_break()

        if fin_page_signal is not None:
            fin_page_signal.emit()

    doc_path = project.doc_path()
    document.save(doc_path)
    if delete_tmp_folder:
        shutil.rmtree(cuts_dir)


def export_project_txt(project, dump_target: str, suffix='.txt'):
    save_path = project.dump_txt_path(dump_target, suffix=suffix)
    text_all = []
    assert dump_target in {'source', 'translation'}
    assert suffix in {'.txt', '.md'}
    for page_name, blk_list in project.pages.items():
        text_in_page = ['### ' + page_name]
        for ii, blk in enumerate(blk_list):
            if dump_target == 'translation':
                text = blk.translation.strip()
            elif dump_target == 'source':
                text = blk.get_text().strip()
            text_in_page.append(f'{ii + 1}. {text}')
        text_all.append('\n\n'.join(text_in_page))
    with open(save_path, 'w', encoding='utf8') as f:
        f.write('\n\n\n'.join(text_all))


def import_project_doc(project, doc_path, delete_tmp_folder=True, fin_page_signal=None):
    tmp_bubble_folder = osp.join(project.directory, 'img_folder')
    os.makedirs(tmp_bubble_folder, exist_ok=True)
    docx2txt.process(doc_path, tmp_bubble_folder)

    doc = docx.Document(doc_path)
    body_xml_str = doc._body._element.xml

    pages = {}
    effect_notices = set()
    bub_index = 0
    for tbl in re.findall(r'<w:tbl>(.*?)</w:tbl>', body_xml_str, re.DOTALL):
        for tr in re.findall(r'<w:tr(.*?)>(.*?)</w:tr>', tbl, re.DOTALL):
            if re.findall(r'<pic:cNvPr id=\"(.*?)\" name=\"(.*?)\"(.*?)>', tr[1]):
                bub_index += 1
                translation = ""
                for paragraph in re.findall(r'<w:p(.*?)>(.*?)</w:p>', tr[1], re.DOTALL):
                    for wt in re.findall(r'<w:t>(.*?)</w:t>', paragraph[1], re.DOTALL):
                        translation += wt
                    translation += "\n"
                translation = translation[:-1]
                if len(translation) != 0 and translation[0] == "\n":
                    translation = translation[1:]

                bubpath = os.path.join(tmp_bubble_folder, "image" + str(bub_index))
                if osp.exists(bubpath + '.jpg'):
                    bubpath = bubpath + '.jpg'
                else:
                    bubpath = bubpath + '.jpeg'

                meta_dict = read_jpg_metadata(bubpath)
                meta_dict["translation"] = translation
                imgkey = meta_dict.pop("imgkey")
                if imgkey not in pages:
                    pages[imgkey] = []
                normalized, notices = normalize_textblock_effect_payload(meta_dict)
                effect_notices.update(notices)
                pages[imgkey].append(TextBlock(**normalized))

                if fin_page_signal is not None:
                    fin_page_signal.emit()

    warn_ignored_legacy_effects(effect_notices, 'document import')
    project.merge_from_proj_dict(pages)
    if delete_tmp_folder:
        shutil.rmtree(tmp_bubble_folder)


def import_project_txt(project, file_path: str, target_language=None) -> bool:
    page_list = parse_txt_translation(file_path)
    missing_pages = []
    unmatched_pages = []
    unexpected_pages = []
    matched_pages = []
    for page_dict in page_list:
        page_name = page_dict['page_name']
        if page_name in project.pages:
            matched_pages.append(page_name)
        else:
            unexpected_pages.append(page_name)
            continue
        blklist = project.pages[page_name]
        n_blk = len(blklist)
        src_blk_list = page_dict['blk_list']
        n_src_blk = len(src_blk_list)
        if n_src_blk != n_blk:
            LOGGER.warning(
                f'Unmatched text blocks in {page_name}, number of text blocks in this page vs source file: {n_blk}-{n_src_blk}'
            )
            unmatched_pages.append(page_name)
        for blkid in range(min(n_blk, n_src_blk)):
            blk = blklist[blkid]
            blk.rich_text = ''
            blk.translation = src_blk_list[blkid]

    matched_pages = set(matched_pages)
    if len(matched_pages) != project.num_pages:
        for page_name in project.pages:
            if page_name not in matched_pages:
                missing_pages.append(page_name)

    all_matched = (
        len(missing_pages) == 0
        and len(unmatched_pages) == 0
        and len(unexpected_pages) == 0
    )
    return all_matched
