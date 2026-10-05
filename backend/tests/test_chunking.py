import itertools
import re

from app.chunking import HARD, OVERLAP, TARGET, chunk_pages, clean_page

HYPHEN_MARK = chr(0xFFFE)  # what pdfium emits for a line-end hyphen
BODY = "\n".join(f"baris isi nomor {i} yang cukup panjang untuk mengisi halaman ini" for i in range(8))


def page(*lines: str) -> str:
    return "\r\n".join(lines)


def test_letterhead_variants_are_removed():
    for head in (
        ("PRESIDEN", "REPUBLIK INDONESIA"),
        ("PRESIDEN", "REPUBUK INDONESIA"),  # old-OCR spellings seen in the corpus
        ("FRESIDEN", "REPUELIK INDONESIA"),
        ("trresiden", "republiktndonesia"),
        ("PRESIDEN REPUBLIK INDONESIA",),
        ("SALINAN",),
        ("BERITA NEGARA", "REPUBLIK INDONESIA"),
    ):
        assert clean_page(page(*head, BODY, "www.peraturan.go.id")) == BODY, head


def test_page_numbers_and_running_headers_are_removed():
    for line in (
        "-2-",
        "- 3 -",
        "-t7-",
        "-L2t-",
        "_4_",
        "2017, No.1907",
        "2019, No.118 -40-",
        ", No.1332",
        "Halaman 3 dari 10",
    ):
        assert clean_page(page(line, BODY)) == BODY, line
        assert clean_page(page(BODY, line)) == BODY, line
    assert clean_page(page(BODY, "SK No 106843 A", "www.peraturan.go.id")) == BODY
    assert clean_page(page(BODY, "Pasal 5 . . .")) == BODY  # catchword: the next page's first words


def test_scrambled_old_ocr_order_still_loses_its_header():
    # PP 11/2017: stamp junk and list markers come before the letterhead and page number
    text = page("#D", "(1)", "PRESIDEN", "REPUBLIK II.JDONESIA", "-90-", "Pasal 156", BODY)
    assert clean_page(text) == f"(1)\nPasal 156\n{BODY}"


def test_body_text_is_not_mistaken_for_letterhead():
    middle = "PRESIDEN REPUBLIK INDONESIA menetapkan peraturan ini"
    assert middle in clean_page(page(BODY, middle, BODY))
    assert "Pasal 87" in clean_page(page("Pasal 87", BODY))
    assert "(2)" in clean_page(page("(2)", BODY))  # an ayat marker is not a page number
    assert "12 orang pegawai" in clean_page(page("12 orang pegawai", BODY))
    assert "3" in clean_page(page(BODY, "3", BODY)).split("\n")  # a bare number inside the page stays


def test_compound_hyphens_are_kept():
    text = page(
        f"peraturan perundang{HYPHEN_MARK}undangan dan masing{HYPHEN_MARK}masing", "Undang-", "Undang Dasar", BODY
    )
    out = clean_page(text)
    assert "perundang-undangan" in out and "masing-masing" in out
    assert "Undang-\nUndang Dasar" in out  # a real line-end hyphen is not joined or removed
    assert HYPHEN_MARK not in out


def test_page_number_glued_to_next_line_by_pdfium_is_split_again():
    # "-7t-" followed by a line: pdfium sees a line-end hyphen and fuses them
    assert clean_page(page(f"-7t{HYPHEN_MARK}Sekretariat Jenderal", BODY)) == "Sekretariat Jenderal\n" + BODY


def test_nul_and_control_characters_are_stripped():
    assert clean_page(page("a\x00b\x0cc", BODY)).startswith("abc")


def pasal(n: int, size: int = 250) -> str:
    return f"Pasal {n}\n" + "\n".join(
        f"(1) Ketentuan pasal {n} mengatur hal yang penting nomor {i}." for i in range(size // 55)
    )


def test_chunks_start_on_pasal_boundaries_and_pack_short_pasals():
    pages = ["BAB I\nKETENTUAN UMUM\n" + "\n".join(pasal(n) for n in range(1, 13))]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1
    for c in chunks:
        assert re.match(r"(BAB I|Pasal \d+)\n", c.content), c.content[:40]
        assert len(c.content) <= HARD
    # every Pasal sits whole inside exactly one chunk
    for n in range(1, 13):
        holders = [c for c in chunks if f"Pasal {n}\n" in c.content]
        assert len(holders) == 1
        assert pasal(n) in holders[0].content
    assert chunks[0].heading.startswith("Pasal 1\N{EN DASH}")  # a pack of short Pasals is headed by its range
    assert chunks[0].content.startswith(
        "BAB I\nKETENTUAN UMUM\nPasal 1"
    )  # the chapter title rides with its first Pasal
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_a_long_pasal_is_split_with_overlap_and_keeps_its_heading():
    lines = [
        f"({i}) Ketentuan yang sangat panjang nomor {i} mengatur banyak hal berbeda dalam pasal ini."
        for i in range(1, 60)
    ]
    chunks = chunk_pages(["Pasal 87\n" + "\n".join(lines)])
    assert len(chunks) >= 3
    assert {c.heading for c in chunks} == {"Pasal 87"}
    for a, b in itertools.pairwise(chunks):
        assert len(a.content) <= HARD
        tail = a.content.split("\n")[-1]
        assert tail in b.content  # overlap: the last line of a chunk opens the next one
        assert len(a.content) <= TARGET + 100
    assert OVERLAP > 0
    assert "\n".join(c.content for c in chunks).count("(59)") >= 1  # nothing lost at the end


def test_cukup_jelas_stubs_are_dropped_and_penjelasan_headings_renamed():
    body = pasal(1) + "\n" + pasal(2)
    pen = (
        "PENJELASAN\nATAS\nPERATURAN PEMERINTAH NOMOR 94 TAHUN 2021\nI. UMUM\n"
        "Dalam rangka melaksanakan ketentuan undang-undang, perlu ditetapkan aturan disiplin yang jelas.\n"
        "II. PASAL DEMI PASAL\nPasal 1\nCukup jelas.\nPasal 2\nAyat (1)\nCukup jelas.\nHuruf a\nCukup jelas.\n"
        "Pasal 3\n"
        'Yang dimaksud dengan "hukuman disiplin" adalah hukuman yang dijatuhkan kepada PNS yang melanggar.'
    )
    chunks = chunk_pages([body, pen])
    heads = [c.heading for c in chunks]
    assert "Penjelasan Umum" in heads and "Penjelasan Pasal 3" in heads
    assert not any(h in ("Penjelasan Pasal 1", "Penjelasan Pasal 2") for h in heads)
    assert "Cukup jelas" not in "\n".join(c.content for c in chunks)
    assert any(h and h.startswith("Pasal ") and "Penjelasan" not in h for h in heads)  # the body keeps plain headings


def test_penjelasan_and_body_never_share_a_chunk():
    chunks = chunk_pages([pasal(1, 100), "PENJELASAN\nATAS\nPP\nPasal 1\nYang dimaksud dengan pegawai adalah orang."])
    for c in chunks:
        assert not ("Pasal 1\n(1)" in c.content and "PENJELASAN" in c.content)


def test_page_ranges_are_carried():
    p1 = "Pasal 1\n" + "\n".join(f"isi baris {i} pada halaman pertama yang cukup panjang" for i in range(5))
    p2 = (
        "\n".join(f"lanjutan baris {i} pada halaman kedua yang cukup panjang" for i in range(5))
        + "\nPasal 2\nisi pasal dua yang pendek."
    )
    p3 = "Pasal 3\nisi pasal tiga di halaman ketiga yang cukup panjang untuk dihitung."
    by_start = {c.page_start: c for c in chunk_pages([p1, p2, p3])}
    assert min(by_start) == 1
    first = by_start[1]
    assert first.page_end >= 2 and "lanjutan baris 0" in first.content  # Pasal 1 runs over the page break
    assert max(c.page_end for c in chunk_pages([p1, p2, p3])) == 3


def test_unpaged_documents_have_no_page_numbers_and_markdown_headings_split():
    text = "# Pengantar\nisi pengantar yang cukup panjang untuk dihitung.\n# Syarat\nisi syarat yang cukup panjang untuk dihitung."
    chunks = chunk_pages([text], paged=False)
    assert all(c.page_start is None and c.page_end is None for c in chunks)
    assert {c.heading for c in chunks} <= {"Pengantar", "Syarat", "Pengantar\N{EN DASH}Syarat"}


def test_text_without_structure_is_still_chunked():
    lines = [f"baris {i} dari dokumen tanpa struktur apa pun yang cukup panjang" for i in range(120)]
    chunks = chunk_pages(["\n".join(lines)])
    assert len(chunks) > 3 and all(c.heading is None for c in chunks)
    assert all(len(c.content) <= HARD for c in chunks)


def test_empty_pages_give_no_chunks():
    assert chunk_pages(["", "  \n "]) == []
