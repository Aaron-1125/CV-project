#!/usr/bin/env python3
"""Export a Stage2 weekly Markdown report to PDF."""

from __future__ import annotations

import argparse
import html
import re
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import CondPageBreak, Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer


IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]
MONO_FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]


def register_font(name: str, candidates: list[str], fallback: Optional[str] = None) -> Optional[str]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(path)))
                return name
            except Exception:
                continue
    return fallback


def register_cjk_body_font() -> str:
    ttf_candidates = [item for item in FONT_CANDIDATES if Path(item).suffix.lower() in {".ttf", ".otf"}]
    font_name = register_font("Stage2WeeklyBody", ttf_candidates, None)
    if font_name:
        return font_name
    cid_name = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(cid_name))
    return cid_name


BODY_FONT = register_cjk_body_font()
MONO_FONT = register_font("Stage2WeeklyMono", MONO_FONT_CANDIDATES, "Courier")


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Stage2WeeklyTitle",
            parent=base["Title"],
            fontName=BODY_FONT,
            fontSize=20,
            leading=26,
            textColor=colors.HexColor("#111827"),
            spaceAfter=13,
            alignment=TA_CENTER,
        ),
        "h2": ParagraphStyle(
            "Stage2WeeklyH2",
            parent=base["Heading2"],
            fontName=BODY_FONT,
            fontSize=14.5,
            leading=19,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=13,
            spaceAfter=7,
        ),
        "h3": ParagraphStyle(
            "Stage2WeeklyH3",
            parent=base["Heading3"],
            fontName=BODY_FONT,
            fontSize=12.2,
            leading=16,
            textColor=colors.HexColor("#374151"),
            spaceBefore=9,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Stage2WeeklyBody",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=10.2,
            leading=14.8,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Stage2WeeklyBullet",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=10.0,
            leading=14.3,
            leftIndent=16,
            bulletIndent=4,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Stage2WeeklyCaption",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#4b5563"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "Stage2WeeklyCode",
            parent=base["Code"],
            fontName=MONO_FONT,
            fontSize=7.1,
            leading=9.1,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#f3f4f6"),
            borderColor=colors.HexColor("#d1d5db"),
            borderWidth=0.35,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "Stage2WeeklyTable",
            parent=base["Code"],
            fontName=BODY_FONT,
            fontSize=8.4,
            leading=10.6,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#f9fafb"),
            borderColor=colors.HexColor("#e5e7eb"),
            borderWidth=0.25,
            borderPadding=3.5,
            spaceBefore=2,
            spaceAfter=3,
        ),
    }


def markdown_inline(text: str) -> str:
    escaped = html.escape(text)
    return INLINE_CODE_RE.sub(
        lambda match: f'<font name="{MONO_FONT}" color="#0f766e">{html.escape(match.group(1))}</font>',
        escaped,
    )


def wrap_code(code: str, width: int = 96) -> str:
    wrapped: list[str] = []
    for line in code.rstrip().splitlines():
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            or [""]
        )
    return "\n".join(wrapped)


def image_flowables(source_dir: Path, alt: str, path_text: str, styles: dict[str, ParagraphStyle]):
    image_path = Path(path_text)
    if not image_path.is_absolute():
        image_path = (source_dir / image_path).resolve()
    if not image_path.exists():
        return [Paragraph(f"[missing image] {markdown_inline(path_text)}", styles["body"])]

    max_width = A4[0] - 3.4 * cm
    normalized_path = path_text.replace("\\", "/")
    if "/assets/weekly/" in normalized_path:
        max_height = 7.2 * cm
    elif "/evaluation/" in normalized_path:
        max_height = 8.2 * cm
    elif "/training/" in normalized_path:
        max_height = 7.6 * cm
    else:
        max_height = 10.0 * cm
    with PILImage.open(image_path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height, 1.0)
    return [
        KeepTogether(
            [
                Image(str(image_path), width=width * scale, height=height * scale),
                Paragraph(markdown_inline(alt or path_text), styles["caption"]),
            ]
        )
    ]


def parse_markdown(source: Path, styles: dict[str, ParagraphStyle]):
    lines = source.read_text(encoding="utf-8").splitlines()
    flowables = []
    paragraph_parts: list[str] = []
    in_code = False
    code_lines: list[str] = []
    source_dir = source.parent

    def flush_paragraph() -> None:
        if paragraph_parts:
            text = " ".join(part.strip() for part in paragraph_parts if part.strip())
            flowables.append(Paragraph(markdown_inline(text), styles["body"]))
            paragraph_parts.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                flowables.append(Preformatted(wrap_code("\n".join(code_lines)), styles["code"]))
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        image_match = IMAGE_RE.fullmatch(line.strip())
        if image_match:
            flush_paragraph()
            flowables.extend(image_flowables(source_dir, image_match.group(1), image_match.group(2), styles))
            continue
        if not line.strip():
            flush_paragraph()
            flowables.append(Spacer(1, 4))
            continue
        if line.startswith("# "):
            flush_paragraph()
            flowables.append(Paragraph(markdown_inline(line[2:].strip()), styles["title"]))
            continue
        if line.startswith("## "):
            flush_paragraph()
            if flowables:
                flowables.append(Spacer(1, 3))
            flowables.append(Paragraph(markdown_inline(line[3:].strip()), styles["h2"]))
            continue
        if line.startswith("### "):
            flush_paragraph()
            flowables.append(Paragraph(markdown_inline(line[4:].strip()), styles["h3"]))
            continue
        if line.startswith("- "):
            flush_paragraph()
            flowables.append(Paragraph(markdown_inline(line[2:].strip()), styles["bullet"], bulletText="-"))
            continue
        if line.startswith("|"):
            flush_paragraph()
            flowables.append(Preformatted(wrap_code(line, width=104), styles["table"]))
            continue
        if line == "---PAGEBREAK---":
            flush_paragraph()
            flowables.append(CondPageBreak(7.2 * cm))
            continue
        paragraph_parts.append(line)

    flush_paragraph()
    return flowables


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(BODY_FONT, 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 1.7 * cm, 1.05 * cm, f"Page {doc.page}")
    canvas.restoreState()


def export_pdf(source: Path, output: Path) -> None:
    styles = build_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.25 * cm,
        title=source.stem,
        author="CV Project",
    )
    doc.build(parse_markdown(source, styles), onFirstPage=add_page_number, onLaterPages=add_page_number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_pdf(args.source, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
