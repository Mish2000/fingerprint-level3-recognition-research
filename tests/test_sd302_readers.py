import pytest

from conftest import make_png
from fpl3.data.sd302 import parse_filename, read_checksum_csv, read_png_header


def test_parse_filename():
    fields = parse_filename("00002302_R_1000_slap_13.png")
    assert (fields.subject, fields.device, fields.ppi, fields.capture, fields.frgp) == ("00002302", "R", 1000, "slap", 13)


@pytest.mark.parametrize("name", ["00002302_A_roll_01.png", "2302_U_1000_roll_01.png", "00002302_U_1000_roll_1.png"])
def test_parse_filename_rejects_other_formats(name):
    with pytest.raises(ValueError):
        parse_filename(name)


def test_png_header_reads_dimensions_and_resolution(tmp_path):
    path = tmp_path / "a.png"
    make_png(path, width=7, height=5)
    header = read_png_header(path)
    assert (header.width, header.height, header.bit_depth, header.color_type) == (7, 5, 8, 0)
    assert (header.ppi_x, header.ppi_y) == (1000.0, 1000.0)


def test_png_header_without_phys_has_no_resolution(tmp_path):
    path = tmp_path / "a.png"
    make_png(path, per_metre=None)
    header = read_png_header(path)
    assert (header.ppi_x, header.ppi_y) == (None, None)


def test_png_header_rejects_non_png(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"not a png")
    with pytest.raises(ValueError):
        read_png_header(path)


@pytest.mark.parametrize("hash_column", ["sha256", "checksum"])
def test_checksum_csv_accepts_both_nist_header_spellings(tmp_path, hash_column):
    path = tmp_path / "checksum.csv"
    path.write_text(f"{hash_column},filename\nABCDEF,00002302_U_1000_roll_01.png\n", encoding="utf-8")
    assert read_checksum_csv(path) == {"00002302_U_1000_roll_01.png": "abcdef"}


def test_checksum_csv_column_order_does_not_matter(tmp_path):
    path = tmp_path / "checksum.csv"
    path.write_text("filename,sha256\nsub/dir/x.png,AA\n", encoding="utf-8")
    assert read_checksum_csv(path) == {"x.png": "aa"}
