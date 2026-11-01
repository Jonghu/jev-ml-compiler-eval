import jev_compiler_eval


def test_package_import() -> None:
    assert jev_compiler_eval.__version__ == "0.1.0"
