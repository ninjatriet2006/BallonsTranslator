import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .config import RunStatus
from .proj_imgtrans import ProjImgTrans


@dataclass
class StageIssue:
    """Represents a stage execution failure, skip, or audit discrepancy."""

    page_key: str
    stage: str  # 'detect', 'ocr', 'translate', 'inpaint', 'verify'
    error_type: str  # 'safety_refusal', 'network_error', 'json_parse_error', 'stage_missing', 'stage_failed'
    message: str
    page_index: int = -1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'page_key': self.page_key,
            'stage': self.stage,
            'error_type': self.error_type,
            'message': self.message,
            'page_index': self.page_index,
            'timestamp': self.timestamp,
        }


def parse_page_ranges(range_str: str, max_pages: Optional[int] = None) -> List[int]:
    """Parse comma-separated page ranges (1-indexed) into sorted unique page numbers.

    Supported patterns:
        - "1-5, 8, 12-15"
        - "1,2,3"
        - " 4 - 7 , 10 "
    """
    if not range_str or not range_str.strip():
        return []

    pages: Set[int] = set()
    tokens = [t.strip() for t in range_str.split(',') if t.strip()]

    for token in tokens:
        match = re.match(r'^(\d+)(?:\s*-\s*(\d+))?$', token)
        if not match:
            continue
        start_p = int(match.group(1))
        end_p = int(match.group(2)) if match.group(2) else start_p

        if start_p > end_p:
            start_p, end_p = end_p, start_p

        if start_p < 1:
            start_p = 1
        if max_pages is not None:
            if start_p > max_pages:
                continue
            end_p = min(end_p, max_pages)

        for p in range(start_p, end_p + 1):
            pages.add(p)

    return sorted(pages)


def format_page_ranges(pages: List[int]) -> str:
    """Format sorted page numbers into compact range string.

    E.g. [1, 2, 3, 5, 8, 9, 10] -> "1-3, 5, 8-10"
    """
    if not pages:
        return ""

    sorted_pages = sorted(set(pages))
    ranges = []
    range_start = sorted_pages[0]
    range_end = range_start

    for p in sorted_pages[1:]:
        if p == range_end + 1:
            range_end = p
        else:
            if range_start == range_end:
                ranges.append(str(range_start))
            else:
                ranges.append(f"{range_start}-{range_end}")
            range_start = p
            range_end = p

    if range_start == range_end:
        ranges.append(str(range_start))
    else:
        ranges.append(f"{range_start}-{range_end}")

    return ", ".join(ranges)


def is_safety_refusal_error(error: Exception | str) -> bool:
    """Check if error indicates an LLM safety refusal or policy filter."""
    msg = str(error).lower()
    refusal_keywords = [
        "cannot assist with that request",
        "i'm sorry, but i cannot",
        "i cannot fulfill this request",
        "safety policy",
        "content policy",
        "refused",
        "sensitive content",
        "inappropriate content",
    ]
    return any(kw in msg for kw in refusal_keywords)


class StageAuditor:
    """Deterministic, zero-AI-token auditor for image translation project pages."""

    @classmethod
    def audit_project_pages(
        cls,
        project: ProjImgTrans,
        target_pages: Optional[List[str]] = None,
        enabled_stages: Optional[Dict[str, bool]] = None,
    ) -> List[StageIssue]:
        """Audit target pages against enabled stages to detect missing or dropped data."""
        if project is None or not hasattr(project, 'pages'):
            return []

        enabled = enabled_stages or {
            'detect': True,
            'ocr': True,
            'translate': True,
            'inpaint': True,
        }

        pages_to_check = target_pages if target_pages is not None else list(project.pages.keys())
        issues: List[StageIssue] = []

        for pagename in pages_to_check:
            if pagename not in project.pages:
                continue

            page_idx = project.pagename2idx(pagename) if hasattr(project, 'pagename2idx') else -1
            textblocks = project.pages[pagename]
            fin_code = 0
            if hasattr(project, 'get_page_finish_code'):
                fin_code = project.get_page_finish_code(pagename)
            elif hasattr(project, 'get_page_progress'):
                fin_code = project.get_page_progress(pagename)

            # 1. Detection Stage Audit
            if enabled.get('detect', False):
                has_det_flag = bool(fin_code & RunStatus.FIN_DET)
                if not has_det_flag and len(textblocks) == 0:
                    issues.append(StageIssue(
                        page_key=pagename,
                        stage='detect',
                        error_type='stage_missing',
                        message=f"Page {pagename} has no detected text blocks.",
                        page_index=page_idx,
                    ))

            # 2. OCR Stage Audit
            if enabled.get('ocr', False) and len(textblocks) > 0:
                has_ocr_flag = bool(fin_code & RunStatus.FIN_OCR)
                blocks_missing_ocr = sum(1 for b in textblocks if not getattr(b, 'text', '').strip())
                if not has_ocr_flag and blocks_missing_ocr == len(textblocks):
                    issues.append(StageIssue(
                        page_key=pagename,
                        stage='ocr',
                        error_type='stage_missing',
                        message=f"Page {pagename} text blocks are missing OCR textlines.",
                        page_index=page_idx,
                    ))

            # 3. Translation Stage Audit
            if enabled.get('translate', False) and len(textblocks) > 0:
                has_trans_flag = bool(fin_code & RunStatus.FIN_TRANSLATE)
                blocks_missing_trans = sum(
                    1 for b in textblocks
                    if not getattr(b, 'translation', '').strip()
                    and bool(getattr(b, 'text', '').strip())
                )
                if not has_trans_flag and blocks_missing_trans > 0:
                    issues.append(StageIssue(
                        page_key=pagename,
                        stage='translate',
                        error_type='stage_missing',
                        message=f"Page {pagename} has {blocks_missing_trans} block(s) without translation.",
                        page_index=page_idx,
                    ))

            # 4. Inpainting Stage Audit
            if enabled.get('inpaint', False) and len(textblocks) > 0:
                has_inpaint_flag = bool(fin_code & RunStatus.FIN_INPAINT)
                if not has_inpaint_flag:
                    issues.append(StageIssue(
                        page_key=pagename,
                        stage='inpaint',
                        error_type='stage_missing',
                        message=f"Page {pagename} inpainting was not completed.",
                        page_index=page_idx,
                    ))

        return issues
