from concurrent.futures import ThreadPoolExecutor

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import select, update

from legal_workspace.persistence.database import Database
from legal_workspace.persistence.models import Base, Workspace


def test_migration_matches_models_and_enables_sqlite_features(database):
    with database.engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert (
            connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one()
            == "0002_documents"
        )
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []


def test_reinitialization_and_reopen_preserve_workspace(database, settings):
    with database.transaction() as connection:
        connection.execute(
            update(Workspace).where(Workspace.id == 1).values(name="Saved workspace")
        )
        created = connection.scalar(select(Workspace.created_at))
    database.close()
    reopened = Database(settings)
    try:
        reopened.initialize()
        reopened.initialize()
        with reopened.engine.connect() as connection:
            row = connection.execute(select(Workspace.name, Workspace.created_at)).one()
            assert row == ("Saved workspace", created)
        assert reopened.check()
    finally:
        reopened.close()


def test_concurrent_first_start_is_safe(settings):
    def initialize(_):
        db = Database(settings)
        try:
            db.initialize()
            return db.check()
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(initialize, range(2))) == [True, True]
