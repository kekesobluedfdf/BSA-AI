from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _text_of(element) -> str:
    return "".join(node.text or "" for node in element.iter(f"{W}t"))


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{W}body")
    if body is None:
        return ""
    lines = []
    for child in body:
        if child.tag == f"{W}p":
            value = _text_of(child).strip()
            if value:
                lines.append(value)
        elif child.tag == f"{W}tbl":
            lines.append("[TABLE]")
            for row in child.findall(f"{W}tr"):
                cells = []
                for cell in row.findall(f"{W}tc"):
                    value = " ".join(
                        _text_of(paragraph).strip()
                        for paragraph in cell.findall(f"{W}p")
                        if _text_of(paragraph).strip()
                    )
                    cells.append(value)
                lines.append(" | ".join(cells))
            lines.append("[/TABLE]")
    return "\n".join(lines)
