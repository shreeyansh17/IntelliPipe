"""Seed initial data for local development."""
import asyncio
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from src.core.config import get_settings
from src.db.models import PipelineTable

settings = get_settings()

async def seed():
    engine = create_async_engine(settings.database.url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        tables = [
            PipelineTable(
                id=uuid.uuid4(),
                tenant_id="default",
                schema_name="raw",
                table_name="raw_orders",
                description="Raw e-commerce order events from Kafka",
                owner_team="data-platform",
                sla_freshness_minutes=30,
                kafka_topic="raw_events",
                dbt_model_ref="stg_raw_orders",
            ),
            PipelineTable(
                id=uuid.uuid4(),
                tenant_id="default",
                schema_name="staging",
                table_name="stg_raw_orders",
                description="Staged and validated order events",
                owner_team="data-platform",
                sla_freshness_minutes=45,
                dbt_model_ref="mart_order_summary",
            ),
        ]
        session.add_all(tables)
        await session.commit()
        print(f"Seeded {len(tables)} tables")

if __name__ == "__main__":
    asyncio.run(seed())
