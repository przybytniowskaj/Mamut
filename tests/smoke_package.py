from importlib import metadata, resources

from mamut.wrapper import Mamut


def run_smoke() -> None:
    assert Mamut.__name__ == "Mamut"

    package_metadata = metadata.metadata("mamut")
    assert package_metadata["Name"] == "mamut"
    requires_python = package_metadata["Requires-Python"]
    assert ">=3.12" in requires_python
    assert "<3.13" in requires_python

    utils_files = resources.files("mamut.utils")
    for file_name in ("report_template.html", "mamut_header.png"):
        package_file = utils_files.joinpath(file_name)
        assert package_file.is_file(), f"Missing package data file: {file_name}"
        assert package_file.read_bytes(), f"Empty package data file: {file_name}"


if __name__ == "__main__":
    run_smoke()
