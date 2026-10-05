"""Page cleaning and legal-structure chunking. Pure functions, no I/O.

Tuned on the Indonesian ASN regulation corpus: old-OCR text layers with a mangled letterhead on every page,
`-3-` style page numbers, Pasal/BAB structure and a Penjelasan part full of "Cukup jelas." stubs.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

TARGET = 1200  # preferred chunk size in characters
HARD = 1500  # a segment up to this size stays in one piece; anything longer is split
OVERLAP = 200  # overlap between pieces of a split segment
MIN_CHUNK = 20  # chunks shorter than this carry no content
EDGE_LINES = 6  # header/footer candidates: first and last N lines of a page (old OCR scrambles the order)
LETTERHEAD_RATIO = 0.8
TITLE_BLOCK_MAX = 250  # a BAB/Bagian block up to this size is only titles and is glued to the next segment

# Letters-only forms of the stationery printed on every page of the corpus.
_LETTERHEAD = (
    "presiden",
    "republikindonesia",
    "presidenrepublikindonesia",
    "wwwperaturangoid",
    "salinan",
    "beritanegara",
)
_DASH = "-\N{EN DASH}\N{EM DASH}~_"
_PAGE_NUMBER = re.compile(
    rf"^(?:[{_DASH}]\s*[0-9A-Za-z]{{1,4}}\s*[{_DASH}]?"  # -3-  - 3 -  -t7-  -L2t  (old-OCR digits)
    rf"|(?=[^-]*\d)[0-9A-Za-z]{{1,4}}\s*[{_DASH}]"  # 3-
    rf"|[{_DASH}.]"  # a lone dash left over from "-" "3" "-" split across lines
    r"|(?:halaman|hal\.?)\s*\d+(?:\s*(?:dari|/)\s*\d+)?)$",  # Halaman 3 dari 10
    re.IGNORECASE,
)
_BARE_NUMBER = re.compile(r"^\d{1,4}$")  # a page number or year; trusted only on the very first/last line
_CATCHWORD = re.compile(r"^.{1,40}?[.\N{HORIZONTAL ELLIPSIS}](?:\s?[.\N{HORIZONTAL ELLIPSIS}])+$")  # "Pasal 5 . . ."
_RUNNING_HEADER = re.compile(  # Berita Negara / Lembaran Negara header, alone or fused with the page number
    rf"^(?:[{_DASH}]\s*\w{{1,4}}\s*[{_DASH}]\s*)?(?:\d{{4}})?,?\s*No\.?\s*\d+(?:\s*[{_DASH}]\s*\w{{1,4}}\s*[{_DASH}]?)?$",
    re.IGNORECASE,
)
_STAMP = re.compile(  # Setkab footer, browser-print footers ("1 of 3 3/5/2021, 10:20 AM") and JDIH download URLs
    r"^(?:SK\s*No\.?\s*[0-9OIl]+\s*[A-Z]?|\d+\s+of\s+\d+(?:\s+\d+/\d+/\d+.*)?|.*jdih\.[a-z.]+/api/download\S*"
    r'|.{0,6}[,"*]D)$',  # the last is the mangled SK signature stamp ("q,D", "#",D")
    re.IGNORECASE,
)
_SCAN_NOISE = re.compile(r"[<>{}#~^\\]")  # OCR garbage from stamps; only trusted on short edge lines
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # includes NUL, which Postgres text rejects
_SPACES = re.compile(r"[ \t\xa0]+")
# pdfium marks a line-end hyphen with U+FFFE and joins the next line. Every one in the corpus is a real
# compound (perundang-undangan, masing-masing), so it becomes "-". When it follows a page number like "-7t-"
# the line break is restored instead.
_PDFIUM_HYPHEN = chr(0xFFFE)
_PAGE_NUMBER_JOIN = re.compile(rf"(?m)^(\s*-\s*[0-9A-Za-z]{{1,4}}\s*){_PDFIUM_HYPHEN}")


def normalize(text: str) -> list[str]:
    """Non-empty lines with control characters removed, whitespace collapsed and pdfium hyphens resolved."""
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _PAGE_NUMBER_JOIN.sub(r"\1-\n", text).replace(_PDFIUM_HYPHEN, "-")
    return [s for line in text.split("\n") if (s := _SPACES.sub(" ", line).strip())]


def _is_edge_noise(line: str) -> bool:
    if _PAGE_NUMBER.match(line) or _RUNNING_HEADER.match(line) or _STAMP.match(line):
        return True
    if len(line) <= 30 and _SCAN_NOISE.search(line):
        return True
    key = re.sub(r"[^a-z]", "", line.lower())
    return any(
        abs(len(key) - len(ref)) <= 3 and SequenceMatcher(None, key, ref).ratio() >= LETTERHEAD_RATIO
        for ref in _LETTERHEAD
    )


def clean_page(text: str) -> str:
    """Normalise one page and drop its letterhead, page number and running header/footer lines.

    Only the first and last EDGE_LINES lines are candidates, so body text that merely resembles the
    letterhead is never touched. Line-end hyphens are kept.
    """
    lines = normalize(text)
    n = len(lines)
    edge = {*range(min(EDGE_LINES, n)), *range(max(0, n - EDGE_LINES), n)}

    def noise(i: int, line: str) -> bool:
        return (
            _is_edge_noise(line)
            or (i in (0, n - 1) and bool(_BARE_NUMBER.match(line)))
            or (i >= n - 2 and bool(_CATCHWORD.match(line)))  # the next page's first words, printed under this one
        )

    return "\n".join(line for i, line in enumerate(lines) if i not in edge or not noise(i, line))


# --- structure -------------------------------------------------------------------------------------------

_PASAL = re.compile(r"^Pasal\s+(\d+[A-Za-z]?|[IVX]{1,4}|[0-9lIO]{1,4}[A-Za-z]?)$")  # the line is only "Pasal N"
_BAB = re.compile(r"^BAB\s+[IVXLC1lI]{1,6}$")
_SUBSECTION = re.compile(r"^(Bagian\s+Ke\w+|Paragraf\s+\d+)$")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_PENJELASAN = re.compile(r"^PENJELASAN(\s+ATAS)?$")
_PEN_UMUM = re.compile(r"^(I\.\s*)?UMUM\.?$")
_PEN_PASAL_DEMI_PASAL = re.compile(r"^(II\.\s*)?PASAL\s+DEMI\s+PASAL\.?$")
_LAMPIRAN = re.compile(r"^LAMPIRAN\b")
_STUB_WORDS = re.compile(
    r"(?i)cukup\s*jelas|huruf\s+[a-z]\b|ayat\s*\(\d+[a-z]?\)|angka\s+\d+|butir\s+\d+|pasal\s*\w{1,4}\b|[^a-z]"
)


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    page_start: int | None
    page_end: int | None
    heading: str | None


@dataclass
class _Line:
    text: str
    page: int | None


@dataclass
class _Segment:
    heading: str | None
    pen: bool  # inside the Penjelasan part
    lines: list[_Line]
    title_block: bool = False  # BAB/Bagian/PENJELASAN title lines, merged into the next segment when small

    @property
    def size(self) -> int:
        return sum(len(line.text) + 1 for line in self.lines)


def _segments(lines: list[_Line], markdown: bool) -> list[_Segment]:
    segs: list[_Segment] = [_Segment(None, False, [])]
    pen = False
    for line in lines:
        t = line.text
        if not pen and _PENJELASAN.match(t):
            pen = True
            segs.append(_Segment(None, True, [line], title_block=True))
        elif pen and _LAMPIRAN.match(t):
            pen = False
            segs.append(_Segment(None, False, [line], title_block=True))
        elif m := _PASAL.match(t):
            segs.append(_Segment(f"{'Penjelasan ' if pen else ''}Pasal {m.group(1)}", pen, [line]))
        elif pen and _PEN_UMUM.match(t):
            segs.append(_Segment("Penjelasan Umum", True, [line]))
        elif _BAB.match(t) or _SUBSECTION.match(t) or (pen and _PEN_PASAL_DEMI_PASAL.match(t)):
            if segs[-1].title_block:  # BAB then Bagian: one title block
                segs[-1].lines.append(line)
            else:
                segs.append(_Segment(None, pen, [line], title_block=True))
        elif markdown and (m := _MD_HEADING.match(t)):
            segs.append(_Segment(m.group(1).strip()[:120], pen, [line]))
        else:
            segs[-1].lines.append(line)

    # A title-only block is glued to the segment that follows, so the Pasal chunk carries its chapter.
    # A block with real text under it (a BAB with no Pasal) stays a segment, headed by its title.
    merged: list[_Segment] = []
    carry: list[_Line] = []
    for seg in segs:
        if seg.title_block:
            if seg.size <= TITLE_BLOCK_MAX:
                carry += seg.lines
                continue
            first = seg.lines[0].text
            second = seg.lines[1].text if len(seg.lines) > 1 and seg.lines[1].text.isupper() else ""
            seg.heading = f"{first} {second}".strip() if first.startswith("BAB") else None
        if carry:
            seg.lines = carry + seg.lines
            carry = []
        if seg.lines:
            merged.append(seg)
    if carry:
        merged.append(_Segment(None, segs[-1].pen, carry))
    return merged


def _is_stub(seg: _Segment) -> bool:
    """A Penjelasan 'Pasal N / Cukup jelas.' entry that says nothing else."""
    if not (seg.pen and seg.heading and seg.heading.startswith("Penjelasan Pasal")):
        return False
    body = " ".join(line.text for line in seg.lines[1:])
    return "jelas" in body.lower() and len(_STUB_WORDS.sub("", body)) < 15  # leftovers are scan-stamp junk


def _break_long(line: _Line) -> list[_Line]:
    """Split a line longer than HARD at spaces so windows stay bounded."""
    if len(line.text) <= HARD:
        return [line]
    out, text = [], line.text
    while len(text) > HARD:
        cut = text.rfind(" ", TARGET // 2, TARGET)
        cut = cut if cut > 0 else TARGET
        out.append(_Line(text[:cut].rstrip(), line.page))
        text = text[cut:].lstrip()
    return [*out, _Line(text, line.page)] if text else out


def _split(lines: list[_Line]) -> list[list[_Line]]:
    """Windows of about TARGET characters at line boundaries, ending on a sentence where possible,
    overlapping by about OVERLAP characters."""
    lines = [piece for line in lines for piece in _break_long(line)]
    sizes = [len(line.text) + 1 for line in lines]
    out: list[list[_Line]] = []
    i, n = 0, len(lines)
    while i < n:
        if sum(sizes[i:]) <= HARD:  # no tiny tail made mostly of overlap
            out.append(lines[i:])
            break
        j, size = i, 0
        while j < n and (j == i or size + sizes[j] <= TARGET):
            size += sizes[j]
            j += 1
        for k in range(j, i + 1, -1):  # prefer ending on a sentence inside the last 40% of the window
            if sum(sizes[i:k]) < TARGET * 0.6:
                break
            if lines[k - 1].text.endswith((".", ";", ":")):
                j = k
                break
        out.append(lines[i:j])
        back, ov = j, 0
        while back > i + 1 and ov + sizes[back - 1] <= OVERLAP:
            back -= 1
            ov += sizes[back]
        i = back
    return out


def _family(seg: _Segment) -> str:
    """Segments are only packed into one chunk with others of the same family, so a chunk's heading stays true."""
    if seg.heading and seg.heading.startswith("Penjelasan Pasal"):
        return "pen-pasal"
    return "pen" if seg.pen else "body"


def _heading_for(segs: list[_Segment]) -> str | None:
    heads = [s.heading for s in segs if s.heading]
    if not heads:
        return None
    first, last = heads[0], heads[-1]
    if first == last:
        return first
    a, b = first.rsplit(" ", 1), last.rsplit(" ", 1)
    return f"{a[0]} {a[1]}\N{EN DASH}{b[1]}" if len(a) == 2 and len(b) == 2 and a[0] == b[0] else first


def chunk_pages(pages: list[str], paged: bool = True) -> list[Chunk]:
    """Chunk cleaned pages (lines separated by newlines).

    Chunks start on Pasal boundaries; short Pasals are packed together up to TARGET characters and a segment
    longer than HARD is split with overlap. The heading is the nearest one ("Pasal 87", "Pasal 2-4" for a pack,
    "Penjelasan Pasal 87"). Page numbers are 1-based; `paged=False` (docx, md, txt) leaves them None and turns
    on markdown headings.
    """
    lines = [
        _Line(text, i if paged else None)
        for i, page in enumerate(pages, 1)
        for text in page.split("\n")
        if text.strip()
    ]
    chunks: list[Chunk] = []

    def emit(parts: list[_Line], heading: str | None) -> None:
        content = "\n".join(line.text for line in parts)
        if len(content) < MIN_CHUNK:
            return
        nums = [line.page for line in parts if line.page is not None]
        chunks.append(Chunk(len(chunks), content, min(nums, default=None), max(nums, default=None), heading))

    pack: list[_Segment] = []

    def flush() -> None:
        if pack:
            emit([line for seg in pack for line in seg.lines], _heading_for(pack))
            pack.clear()

    for seg in _segments(lines, markdown=not paged):
        if _is_stub(seg):
            continue
        if seg.size > HARD:
            flush()
            for piece in _split(seg.lines):
                emit(piece, seg.heading)
            continue
        if pack and (sum(s.size for s in pack) + seg.size > TARGET or _family(pack[-1]) != _family(seg)):
            flush()
        pack.append(seg)
    flush()
    return chunks
