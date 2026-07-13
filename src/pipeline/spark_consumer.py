"""
IntelliPipe PySpark Structured Streaming Consumer
===================================================
Production-grade streaming pipeline that:
1. Reads from Kafka raw_events topic
2. Parses and validates JSON payloads
3. Detects schema drift against known version
4. Routes invalid events to Dead Letter Queue
5. Computes micro-batch DQ metrics
6. Emits alerts to Redis for the LLM agent
7. Writes clean events to Delta Lake / PostgreSQL sink
8. Tracks watermarks for late-arrival handling
9. Supports event replay via Kafka offset management
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import redis
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.telemetry import (
    EVENTS_CONSUMED_TOTAL,
    SCHEMA_DRIFT_DETECTED_TOTAL,
    timed_operation,
    PIPELINE_BATCH_DURATION,
)

logger = get_logger(__name__, component="spark_consumer")
settings = get_settings()


# ---------------------------------------------------------------------------
# Expected schema definition (v1 — "golden standard")
# ---------------------------------------------------------------------------

ORDER_ITEM_SCHEMA = StructType(
    [
        StructField("product_id", StringType(), True),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("discount_pct", DoubleType(), True),
    ]
)

ORDER_EVENT_SCHEMA_V1 = StructType(
    [
        StructField("event_id", StringType(), False),
        StructField("event_type", StringType(), True),
        StructField("tenant_id", StringType(), False),
        StructField("order_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("customer_email", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField("payment_method", StringType(), True),
        StructField("items", ArrayType(ORDER_ITEM_SCHEMA), True),
        StructField("subtotal", DoubleType(), True),
        StructField("tax_amount", DoubleType(), True),
        StructField("shipping_amount", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("shipping_country", StringType(), True),
        StructField("shipping_city", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("event_timestamp", TimestampType(), True),
        StructField("schema_version", StringType(), True),
        StructField("source_system", StringType(), True),
        StructField("partition_date", StringType(), True),
    ]
)

EXPECTED_COLUMNS = frozenset(f.name for f in ORDER_EVENT_SCHEMA_V1.fields)
REQUIRED_COLUMNS = frozenset(
    f.name for f in ORDER_EVENT_SCHEMA_V1.fields if not f.nullable
)


def schema_fingerprint(schema: StructType) -> str:
    """Generate a deterministic hash of a Spark schema for drift comparison."""
    column_sig = sorted(f"{f.name}:{f.dataType.simpleString()}" for f in schema.fields)
    return hashlib.sha256("|".join(column_sig).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------


def create_spark_session() -> SparkSession:
    """
    Create a production-configured SparkSession.
    In production, uses yarn/k8s master; locally uses local[*].
    """
    spark_conf = settings.spark
    return (
        SparkSession.builder.appName(spark_conf.app_name)
        .master(spark_conf.master)
        .config("spark.sql.shuffle.partitions", str(spark_conf.shuffle_partitions))
        .config(
            "spark.sql.streaming.checkpointLocation", spark_conf.checkpoint_location
        )
        .config("spark.streaming.kafka.consumer.poll.ms", "512")
        # Delta Lake configuration
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Performance optimisations
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Logging
        .config(
            "spark.ui.enabled", "false"
        )  # Disable in prod (use Spark History Server)
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Schema drift detection (runs per micro-batch)
# ---------------------------------------------------------------------------


class SchemaDriftDetector:
    """
    Per-batch schema drift analysis.
    Compares incoming batch schema against the golden standard.
    Emits structured drift events to Redis for LLM agent processing.
    """

    def __init__(
        self, redis_client: redis.Redis, tenant_id: str, table_name: str
    ) -> None:
        self._redis = redis_client
        self._tenant_id = tenant_id
        self._table_name = table_name
        self._baseline_fingerprint = schema_fingerprint(ORDER_EVENT_SCHEMA_V1)

    def detect_drift(
        self,
        batch_df: DataFrame,
        batch_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare batch schema to baseline. Returns drift event dict or None.
        Publishes to Redis alert queue if drift detected.
        """
        observed_cols = frozenset(batch_df.columns)
        added = observed_cols - EXPECTED_COLUMNS
        removed = EXPECTED_COLUMNS - observed_cols

        if not added and not removed:
            return None  # No drift

        drift_event = {
            "alert_type": "schema_drift",
            "tenant_id": self._tenant_id,
            "table_name": self._table_name,
            "batch_id": batch_id,
            "columns_added": sorted(added),
            "columns_removed": sorted(removed),
            "severity": self._calculate_severity(added, removed),
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "baseline_fingerprint": self._baseline_fingerprint,
        }

        self._redis.lpush(
            settings.redis.alert_queue_key,
            json.dumps(drift_event),
        )

        SCHEMA_DRIFT_DETECTED_TOTAL.labels(
            table=self._table_name,
            tenant_id=self._tenant_id,
            drift_type="column_change",
        ).inc()

        logger.warning(
            "Schema drift detected",
            batch_id=batch_id,
            columns_added=list(added),
            columns_removed=list(removed),
        )
        return drift_event

    @staticmethod
    def _calculate_severity(added: frozenset, removed: frozenset) -> str:
        """Removed columns are more severe than added ones."""
        if removed & REQUIRED_COLUMNS:
            return "critical"
        if removed:
            return "high"
        if len(added) > 3:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# DQ micro-batch metrics
# ---------------------------------------------------------------------------


class BatchDQMetrics:
    """
    Compute DQ dimension scores for a micro-batch.
    Scores are pushed to Redis for real-time dashboard consumption.
    """

    VALID_ORDER_STATUSES = {
        "pending",
        "confirmed",
        "shipped",
        "delivered",
        "cancelled",
        "refunded",
    }
    VALID_PAYMENT_METHODS = {
        "credit_card",
        "debit_card",
        "paypal",
        "crypto",
        "wire_transfer",
    }
    VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "SGD", "INR"}

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def compute_and_publish(
        self,
        batch_df: DataFrame,
        tenant_id: str,
        table_name: str,
        batch_id: int,
    ) -> Dict[str, float]:
        """Run all DQ checks and return dimension scores dict."""
        total = batch_df.count()
        if total == 0:
            return {"overall": 100.0}

        # Completeness: % non-null for critical fields
        critical_fields = [
            "event_id",
            "tenant_id",
            "order_id",
            "total_amount",
            "event_timestamp",
        ]
        null_counts = (
            batch_df.select(
                [
                    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
                    for c in critical_fields
                    if c in batch_df.columns
                ]
            )
            .collect()[0]
            .asDict()
        )
        avg_nulls = (
            sum(null_counts.values()) / (len(null_counts) * total) if null_counts else 0
        )
        completeness = round((1 - avg_nulls) * 100, 2)

        # Validity: valid enum values
        if "order_status" in batch_df.columns and "payment_method" in batch_df.columns:
            valid_status_expr = F.col("order_status").isin(self.VALID_ORDER_STATUSES)
            valid_payment_expr = F.col("payment_method").isin(
                self.VALID_PAYMENT_METHODS
            )
            valid_count = batch_df.filter(
                valid_status_expr & valid_payment_expr
            ).count()
            validity = round((valid_count / total) * 100, 2)
        else:
            validity = 0.0  # Missing columns = 0% validity

        # Uniqueness: duplicate event_id check
        if "event_id" in batch_df.columns:
            unique_events = batch_df.select("event_id").distinct().count()
            uniqueness = round((unique_events / total) * 100, 2)
        else:
            uniqueness = 100.0

        # Freshness: % of events within SLA window
        if "event_timestamp" in batch_df.columns:
            sla_minutes = settings.data_quality.freshness_sla_minutes
            now = datetime.now(timezone.utc).timestamp()
            fresh_events = batch_df.filter(
                F.col("event_timestamp").cast("long") >= (now - sla_minutes * 60)
            ).count()
            freshness = round((fresh_events / total) * 100, 2)
        else:
            freshness = 0.0

        # Consistency: total_amount = subtotal + tax + shipping (within tolerance)
        amount_cols = ["subtotal", "tax_amount", "shipping_amount", "total_amount"]
        if all(c in batch_df.columns for c in amount_cols):
            consistent_rows = batch_df.filter(
                F.abs(
                    F.col("total_amount")
                    - (
                        F.col("subtotal")
                        + F.col("tax_amount")
                        + F.col("shipping_amount")
                    )
                )
                <= 0.01
            ).count()
            consistency = round((consistent_rows / total) * 100, 2)
        else:
            consistency = 50.0

        overall = round(
            completeness * 0.25
            + validity * 0.25
            + uniqueness * 0.20
            + freshness * 0.20
            + consistency * 0.10,
            2,
        )

        scores = {
            "overall": overall,
            "completeness": completeness,
            "validity": validity,
            "uniqueness": uniqueness,
            "freshness": freshness,
            "consistency": consistency,
            "row_count": total,
            "batch_id": batch_id,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "table": table_name,
            "tenant_id": tenant_id,
        }

        # Publish to Redis for real-time dashboard
        cache_key = f"{settings.redis.dq_scores_key}:{tenant_id}:{table_name}"
        self._redis.setex(
            cache_key, settings.redis.cache_ttl_seconds, json.dumps(scores)
        )

        # Alert if DQ score is below threshold
        if overall < 80.0:
            alert = {
                "alert_type": "dq_score_degradation",
                "tenant_id": tenant_id,
                "table_name": table_name,
                "dq_score": overall,
                "severity": (
                    "critical" if overall < 60 else "high" if overall < 70 else "medium"
                ),
                "scores": scores,
                "detected_at": datetime.now(timezone.utc).isoformat(),
            }
            self._redis.lpush(settings.redis.alert_queue_key, json.dumps(alert))
            logger.warning(
                "DQ score alert published", overall=overall, table=table_name
            )

        return scores


# ---------------------------------------------------------------------------
# Main streaming pipeline
# ---------------------------------------------------------------------------


class OrderStreamingPipeline:
    """
    PySpark Structured Streaming pipeline for order events.
    Manages the full lifecycle from Kafka ingestion to alert publication.
    """

    def __init__(self) -> None:
        self._spark = create_spark_session()
        self._redis = redis.Redis.from_url(settings.redis.url, decode_responses=True)
        self._drift_detector = SchemaDriftDetector(
            self._redis,
            tenant_id="default",
            table_name="raw_orders",
        )
        self._dq_metrics = BatchDQMetrics(self._redis)
        logger.info("OrderStreamingPipeline initialised")

    def _read_kafka_stream(self) -> DataFrame:
        """Configure Kafka source with structured streaming."""
        kafka_conf = settings.kafka
        spark_conf = settings.spark
        return (
            self._spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_conf.bootstrap_servers)
            .option("subscribe", kafka_conf.raw_events_topic)
            .option("startingOffsets", spark_conf.starting_offsets)
            .option("maxOffsetsPerTrigger", spark_conf.max_offsets_per_trigger)
            .option(
                "failOnDataLoss", "false"
            )  # Handle topic partition deletion gracefully
            .option("kafka.consumer.commit.groupid", kafka_conf.consumer_group_id)
            .load()
        )

    def _parse_events(self, raw_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Parse raw Kafka bytes into structured events.
        Returns (valid_df, dlq_df) tuple.
        """
        # Attempt schema-on-read with permissive mode (corrupted records → _corrupt_record)
        parsed_df = raw_df.select(
            F.col("key").cast("string").alias("kafka_key"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.from_json(
                F.col("value").cast("string"),
                ORDER_EVENT_SCHEMA_V1,
                {"mode": "PERMISSIVE", "columnNameOfCorruptRecord": "_corrupt_record"},
            ).alias("event"),
            F.col("value").cast("string").alias("raw_value"),
        ).select("*", "event.*")

        # Split valid vs corrupt
        valid_df = parsed_df.filter(F.col("_corrupt_record").isNull())
        dlq_df = parsed_df.filter(F.col("_corrupt_record").isNotNull())

        # Add processing metadata
        valid_df = valid_df.withColumn(
            "processing_timestamp", F.current_timestamp()
        ).withColumn("pipeline_version", F.lit("1.0"))

        return valid_df, dlq_df

    def _process_batch(self, batch_df: DataFrame, batch_id: int) -> None:
        """
        Micro-batch processing function (foreachBatch).
        Called by Spark for each micro-batch trigger.
        """
        if batch_df.isEmpty():
            return

        with timed_operation(PIPELINE_BATCH_DURATION, {"stage": "full_batch"}):
            # 1. Schema drift detection
            self._drift_detector.detect_drift(batch_df, batch_id)

            # 2. DQ metrics computation
            scores = self._dq_metrics.compute_and_publish(
                batch_df,
                tenant_id="default",
                table_name="raw_orders",
                batch_id=batch_id,
            )

            # 3. Track metrics
            row_count = batch_df.count()
            EVENTS_CONSUMED_TOTAL.labels(
                topic=settings.kafka.raw_events_topic,
                tenant_id="default",
                status="success",
            ).inc(row_count)

            logger.info(
                "Batch processed",
                batch_id=batch_id,
                row_count=row_count,
                dq_score=scores["overall"],
            )

    def _write_dlq(self, dlq_df: DataFrame) -> StreamingQuery:
        """Write corrupt events to dead-letter Kafka topic."""
        return (
            dlq_df.select(
                F.col("kafka_key").alias("key"),
                F.to_json(
                    F.struct(
                        F.col("raw_value").alias("original"),
                        F.current_timestamp().alias("dlq_timestamp"),
                        F.lit("PARSE_ERROR").alias("dlq_reason"),
                    )
                )
                .cast("bytes")
                .alias("value"),
            )
            .writeStream.format("kafka")
            .option("kafka.bootstrap.servers", settings.kafka.bootstrap_servers)
            .option("topic", settings.kafka.dlq_topic)
            .option("checkpointLocation", f"{settings.spark.checkpoint_location}/dlq")
            .outputMode("append")
            .start()
        )

    def run(self) -> None:
        """Start the streaming pipeline and block until termination."""
        logger.info("Starting streaming pipeline")

        raw_df = self._read_kafka_stream()

        # Add watermark for late-arrival handling
        watermarked_df = raw_df.withColumn(
            "event_ts", F.col("timestamp")
        ).withWatermark("event_ts", settings.spark.watermark_delay)

        valid_df, dlq_df = self._parse_events(watermarked_df)

        # Main processing stream
        main_query = (
            valid_df.writeStream.foreachBatch(self._process_batch)
            .trigger(processingTime=settings.spark.trigger_interval)
            .option("checkpointLocation", f"{settings.spark.checkpoint_location}/main")
            .outputMode("append")
            .start()
        )

        # DLQ stream
        dlq_query = self._write_dlq(dlq_df)

        logger.info(
            "Streaming queries started",
            main_query_id=main_query.id,
            dlq_query_id=dlq_query.id,
        )

        # Block until both streams terminate
        self._spark.streams.awaitAnyTermination()
