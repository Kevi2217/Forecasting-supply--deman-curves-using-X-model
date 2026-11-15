from pathlib import Path

from electricity_forecasting.data.download import download_file, omie_download_url


class FakeResponse:
    def __init__(self, content: bytes, content_type: str = "application/octet-stream"):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return self.response


def test_download_file_writes_payload_and_provenance(tmp_path: Path):
    payload = b"MARGINALPDBC;\n2024;01;01;1;50.0;50.0;\n*\n"
    session = FakeSession(FakeResponse(payload))
    record = download_file(
        base_url="https://www.omie.es/en/file-download",
        filename="marginalpdbc_20240101.1",
        parent="marginalpdbc",
        destination_dir=tmp_path,
        dataset="prices",
        max_retries=0,
        session=session,
    )
    assert (tmp_path / record.filename).read_bytes() == payload
    assert (tmp_path / f"{record.filename}.provenance.json").exists()
    assert record.status == "downloaded"
    assert session.calls[0][0] == omie_download_url(
        "https://www.omie.es/en/file-download", "marginalpdbc_20240101.1", "marginalpdbc"
    )
