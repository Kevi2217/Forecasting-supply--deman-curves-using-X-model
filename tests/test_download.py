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


def test_latest_version_probe_skips_404_and_uses_next_version(tmp_path):
    from datetime import date

    from electricity_forecasting.data.download import download_latest_version, find_local_versions

    class Resp:
        def __init__(self, code, body=b""):
            self.status_code, self.content, self.headers = code, body, {"Content-Type": "application/octet-stream"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("should not be called for 404")

    class Sess:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout):
            self.urls.append(url)
            return Resp(200, b"MARGINALPDBC;\n2023;05;28;1;1;1;\n*\n") if url.endswith("=x") is False and ".2&" in url else Resp(404)

    sess = Sess()
    rec = download_latest_version(
        prefix="marginalpdbc", delivery_date=date(2023, 5, 28), base_url="https://example.invalid/f",
        parent="marginalpdbc", destination_dir=tmp_path, dataset="prices", session=sess, backoff_seconds=0.0,
    )
    assert rec.filename == "marginalpdbc_20230528.2"
    assert len(sess.urls) == 2  # one 404 for .1 (not retried), then .2
    assert [p.name for p in find_local_versions(tmp_path, "marginalpdbc", date(2023, 5, 28))] == ["marginalpdbc_20230528.2"]
