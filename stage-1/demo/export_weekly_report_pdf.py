#!/usr/bin/env python3
"""Export a weekly Markdown report to a polished PDF.

The parser intentionally supports the small Markdown subset used by the
weekly reports: headings, paragraphs, bullets, fenced code blocks and images.
"""

from __future__ import annotations

import argparse
import html
import re
import textwrap
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
MONO_FONT_CANDIDATES = [
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def register_font(name: str, candidates: list[str], fallback: str) -> str:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(path)))
                return name
            except Exception:
                continue
    return fallback


BODY_FONT = register_font("WeeklyBody", FONT_CANDIDATES, "Helvetica")
MONO_FONT = register_font("WeeklyMono", MONO_FONT_CANDIDATES, "Courier")


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "WeeklyTitle",
            parent=base["Title"],
            fontName=BODY_FONT,
            fontSize=21,
            leading=27,
            textColor=colors.HexColor("#111827"),
            spaceAfter=14,
            alignment=TA_CENTER,
        ),
        "h2": ParagraphStyle(
            "WeeklyH2",
            parent=base["Heading2"],
            fontName=BODY_FONT,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "WeeklyH3",
            parent=base["Heading3"],
            fontName=BODY_FONT,
            fontSize=12.5,
            leading=17,
            textColor=colors.HexColor("#374151"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "WeeklyBody",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=10.4,
            leading=15,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "WeeklyBullet",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=10.2,
            leading=14.5,
            leftIndent=16,
            firstLineIndent=0,
            bulletIndent=4,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "WeeklyCaption",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#4b5563"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=9,
        ),
        "code": ParagraphStyle(
            "WeeklyCode",
            parent=base["Code"],
            fontName=MONO_FONT,
            fontSize=7.2,
            leading=9.2,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#f3f4f6"),
            borderColor=colors.HexColor("#d1d5db"),
            borderWidth=0.35,
            borderPadding=5,
            leftIndent=0,
            rightIndent=0,
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def markdown_inline(text: str) -> str:
    escaped = html.escape(text)
    return INLINE_CODE_RE.sub(
        lambda match: f'<font name="{MONO_FONT}" color="#0f766e">{match.group(1)}</font>',
        escaped,
    )


def wrap_code(code: str, width: int = 96) -> str:
    wrapped: list[str] = []
    for line in code.rstrip().splitlines():
        if not line:
            wrapped.append("")
            continue
        chunks = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(chunks or [""])
    return "\n".join(wrapped)


def image_flowables(source_dir: Path, alt: str, path_text: str, styles: dict[str, ParagraphStyle]):
    image_path = (source_dir / path_text).resolve()
    if not image_path.exists():
        return [Paragraph(f"[missing image] {markdown_inline(path_text)}", styles["body"])]

    max_width = A4[0] - 3.4 * cm
    max_height = 12.5 * cm
    with PILImage.open(image_path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height, 1.0)
    flowables = [
        Image(str(image_path), width=width * scale, height=height * scale),
        Paragraph(markdown_inline(alt or path_text), styles["caption"]),
    ]
    return [KeepTogether(flowables)]


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
            flowables.append(Paragraph(markdown_inline(line[2:].strip()), styles["bullet"], bulletText="•"))
            continue

        if line == "---PAGEBREAK---":
            flush_paragraph()
            flowables.append(PageBreak())
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
        topMargin=1.55 * cm,
        bottomMargin=1.45 * cm,
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
