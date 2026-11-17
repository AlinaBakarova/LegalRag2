import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from filelock import FileLock
from sqlalchemy import URL, Connection, create_engine, event, text
from sqlalchemy.dialects.sqlite import insert

from legal_workspace.config import Settings
from legal_workspace.persistence.models import Workspace


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = create_engine(
            URL.create("sqlite", database=str(settings.database_path)),
            connect_args={"check_same_thread": False, "timeout": 5},
        )

        @event.listens_for(self.engine, "connect")
        def configure_connection(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    def initialize(self) -> None:
        self.settings.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Web and worker may start simultaneously, including against an empty directory.
        with FileLock(str(self.settings.data_dir / ".initialize.lock"), timeout=30):
            with self.engine.connect() as connection:
                mode = connection.exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one()
                if mode != "wal":
                    raise RuntimeError("SQLite WAL mode is required")
            config = Config()
            config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
            with self.engine.begin() as connection:
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
                connection.execute(
                    insert(Workspace)
                    .values(id=1, name="Legal AI Workspace", created_at=time.time())
                    .on_conflict_do_nothing(index_elements=["id"])
                )

        from legal_workspace.documents.service import DocumentService

        DocumentService(self).recover()
        from legal_workspace.retrieval.service import SearchService

        SearchService(self).recover()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[Connection]:
        with self.engine.connect() as connection:
            if immediate:
                connection.execute(text("BEGIN IMMEDIATE"))
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def check(self) -> bool:
        with self.engine.connect() as connection:
            return connection.exec_driver_sql("PRAGMA quick_check").scalars().all() == ["ok"]

    def close(self) -> None:
        self.engine.dispose()
