from alembic import context
from sqlalchemy import engine_from_config, pool
from src.db.models import Base
from src.core.config import get_settings

settings = get_settings()
config = context.config
config.set_main_option("sqlalchemy.url", settings.database.sync_url)
target_metadata = Base.metadata

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()
