"""Dashboard launcher path contracts."""
from scripts import start_dashboard


def test_dashboard_launcher_resolves_formal_package_app() -> None:
    app_path = start_dashboard.get_dashboard_app_path()

    assert app_path.is_file()
    assert app_path.as_posix().endswith(
        "src/modular_rag_mcp_server/observability/dashboard/app.py"
    )
