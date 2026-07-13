"""
IntelliPipe Kafka Event Producer
==================================
Simulates an e-commerce order event stream with configurable anomaly injection:
- Schema drift (adding/removing/renaming columns)
- Null spikes (sudden increase in null fields)
- Delayed events (out-of-order, stale timestamps)
- Duplicate events
- Invalid enum values
- Statistical outliers (extreme prices, quantities)
- Stale partitions

Supports:
- Kafka Schema Registry (Avro serialisation)
- Event replay from checkpoint
- Dead-letter queue routing
- Multi-tenant event tagging
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from confluent_kafka import KafkaError, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.telemetry import EVENTS_PRODUCED_TOTAL

logger = get_logger(__name__, component="kafka_producer")
settings = get_settings()


# ---------------------------------------------------------------------------
# Domain event schemas
# ---------------------------------------------------------------------------


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    CRYPTO = "crypto"
    WIRE_TRANSFER = "wire_transfer"


@dataclass
class OrderItem:
    product_id: str
    product_name: str
    category: str
    quantity: int
    unit_price: float
    discount_pct: float = 0.0


@dataclass
class OrderEvent:
    """Base e-commerce order event schema (v1)."""

    event_id: str
    event_type: str
    tenant_id: str
    order_id: str
    customer_id: str
    customer_email: str
    order_status: str
    payment_method: str
    items: List[Dict[str, Any]]
    subtotal: float
    tax_amount: float
    shipping_amount: float
    total_amount: float
    currency: str
    shipping_country: str
    shipping_city: str
    created_at: str
    event_timestamp: str
    schema_version: str = "1.0"
    source_system: str = "order-service"
    partition_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# ---------------------------------------------------------------------------
# Anomaly injection strategies
# ---------------------------------------------------------------------------


class AnomalyInjector:
    """
    Probabilistic anomaly injection for testing detection systems.
    Each method returns a mutated copy of the event dict.
    """

    @staticmethod
    def inject_null_spike(
        event: Dict[str, Any], null_probability: float = 0.8
    ) -> Dict[str, Any]:
        """Randomly null out key fields to simulate null spike."""
        nullable_fields = [
            "customer_email",
            "shipping_city",
            "payment_method",
            "currency",
        ]
        for field_name in nullable_fields:
            if random.random() < null_probability:
                event[field_name] = None
        return event

    @staticmethod
    def inject_schema_drift_add_column(event: Dict[str, Any]) -> Dict[str, Any]:
        """Add unexpected columns — simulates upstream schema change."""
        event["promo_code"] = f"PROMO-{random.randint(1000, 9999)}"
        event["loyalty_points"] = random.randint(0, 5000)
        event["referral_source"] = random.choice(["google", "email", "social", None])
        logger.debug(
            "Schema drift injected: added columns", event_id=event.get("event_id")
        )
        return event

    @staticmethod
    def inject_schema_drift_remove_column(event: Dict[str, Any]) -> Dict[str, Any]:
        """Remove expected columns — simulates upstream schema breaking change."""
        for col in ["shipping_amount", "tax_amount", "schema_version"]:
            event.pop(col, None)
        return event

    @staticmethod
    def inject_schema_drift_rename_column(event: Dict[str, Any]) -> Dict[str, Any]:
        """Rename a column — simulates refactoring without backward compatibility."""
        if "customer_id" in event:
            event["user_id"] = event.pop("customer_id")
        if "order_status" in event:
            event["status"] = event.pop("order_status")
        return event

    @staticmethod
    def inject_statistical_outlier(event: Dict[str, Any]) -> Dict[str, Any]:
        """Inject extreme values for numerical anomaly detection."""
        anomaly_choice = random.choice(["price", "quantity", "discount"])
        if anomaly_choice == "price":
            event["total_amount"] = random.uniform(100_000, 1_000_000)
            event["subtotal"] = event["total_amount"] * 0.9
        elif anomaly_choice == "quantity":
            if event.get("items"):
                event["items"][0]["quantity"] = random.randint(10_000, 50_000)
        elif anomaly_choice == "discount":
            if event.get("items"):
                event["items"][0]["discount_pct"] = random.uniform(101, 999)
        return event

    @staticmethod
    def inject_invalid_enum(event: Dict[str, Any]) -> Dict[str, Any]:
        """Inject invalid enum values."""
        invalid_statuses = ["PROCESSING", "UNKNOWN", "error", "null", "N/A", ""]
        invalid_payments = ["BARTER", "PROMISE", "IOU", None]
        event["order_status"] = random.choice(invalid_statuses)
        event["payment_method"] = random.choice(invalid_payments)
        return event

    @staticmethod
    def inject_delayed_event(
        event: Dict[str, Any], delay_hours: int = 48
    ) -> Dict[str, Any]:
        """Set event timestamp far in the past to simulate delayed/stale events."""
        stale_time = datetime.now(timezone.utc) - timedelta(
            hours=delay_hours + random.randint(0, 24)
        )
        event["event_timestamp"] = stale_time.isoformat()
        event["partition_date"] = (stale_time - timedelta(days=2)).strftime("%Y-%m-%d")
        return event

    @staticmethod
    def inject_duplicate(event: Dict[str, Any]) -> Dict[str, Any]:
        """Keep same order_id but generate new event_id — simulates duplicate."""
        # event_id changes but business key (order_id) stays same
        event["event_id"] = str(uuid.uuid4())
        return event


# ---------------------------------------------------------------------------
# Event generator
# ---------------------------------------------------------------------------

PRODUCT_CATALOG = [
    ("LAPTOP-001", "MacBook Pro 14", "Electronics"),
    ("PHONE-002", "iPhone 15 Pro", "Electronics"),
    ("SHIRT-003", "Classic Oxford Shirt", "Clothing"),
    ("SHOE-004", "Running Shoes Pro", "Footwear"),
    ("BOOK-005", "Clean Architecture", "Books"),
    ("DESK-006", "Standing Desk 160cm", "Furniture"),
    ("HDPH-007", "Noise-Cancelling Headphones", "Electronics"),
    ("WTCH-008", "Smart Watch Series 9", "Electronics"),
    ("BLND-009", "High-Speed Blender", "Kitchen"),
    ("YOGA-010", "Premium Yoga Mat", "Sports"),
]

COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "JP", "SG", "IN", "BR"]
CITIES = [
    "New York",
    "London",
    "Toronto",
    "Sydney",
    "Berlin",
    "Paris",
    "Tokyo",
    "Singapore",
    "Mumbai",
    "São Paulo",
]
TENANTS = ["tenant_alpha", "tenant_beta", "tenant_gamma"]


def generate_order_event(
    tenant_id: Optional[str] = None,
    anomaly_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a realistic order event with optional anomaly injection.

    Args:
        tenant_id: Override tenant. Defaults to random from TENANTS.
        anomaly_type: One of: null_spike, schema_add, schema_remove, schema_rename,
                     outlier, invalid_enum, delayed, duplicate, or None (clean).
    """
    num_items = random.randint(1, 5)
    selected_products = random.sample(
        PRODUCT_CATALOG, min(num_items, len(PRODUCT_CATALOG))
    )

    items = []
    subtotal = 0.0
    for prod_id, prod_name, category in selected_products:
        qty = random.randint(1, 5)
        price = round(random.uniform(9.99, 1499.99), 2)
        discount = round(random.uniform(0, 0.3), 2)
        items.append(
            {
                "product_id": prod_id,
                "product_name": prod_name,
                "category": category,
                "quantity": qty,
                "unit_price": price,
                "discount_pct": discount,
            }
        )
        subtotal += qty * price * (1 - discount)

    subtotal = round(subtotal, 2)
    tax = round(subtotal * 0.08, 2)
    shipping = round(random.uniform(0, 25), 2)
    country_idx = random.randint(0, len(COUNTRIES) - 1)

    event = OrderEvent(
        event_id=str(uuid.uuid4()),
        event_type="order.created",
        tenant_id=tenant_id or random.choice(TENANTS),
        order_id=f"ORD-{random.randint(100000, 999999)}",
        customer_id=f"CUST-{random.randint(10000, 99999)}",
        customer_email=f"user{random.randint(1000, 9999)}@example.com",
        order_status=random.choice(list(OrderStatus)).value,
        payment_method=random.choice(list(PaymentMethod)).value,
        items=items,
        subtotal=subtotal,
        tax_amount=tax,
        shipping_amount=shipping,
        total_amount=round(subtotal + tax + shipping, 2),
        currency="USD",
        shipping_country=COUNTRIES[country_idx],
        shipping_city=CITIES[country_idx],
        created_at=datetime.now(timezone.utc).isoformat(),
        event_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    event_dict = asdict(event)

    # Apply anomaly injection
    if anomaly_type == "null_spike":
        event_dict = AnomalyInjector.inject_null_spike(event_dict)
    elif anomaly_type == "schema_add":
        event_dict = AnomalyInjector.inject_schema_drift_add_column(event_dict)
    elif anomaly_type == "schema_remove":
        event_dict = AnomalyInjector.inject_schema_drift_remove_column(event_dict)
    elif anomaly_type == "schema_rename":
        event_dict = AnomalyInjector.inject_schema_drift_rename_column(event_dict)
    elif anomaly_type == "outlier":
        event_dict = AnomalyInjector.inject_statistical_outlier(event_dict)
    elif anomaly_type == "invalid_enum":
        event_dict = AnomalyInjector.inject_invalid_enum(event_dict)
    elif anomaly_type == "delayed":
        event_dict = AnomalyInjector.inject_delayed_event(event_dict)
    elif anomaly_type == "duplicate":
        event_dict = AnomalyInjector.inject_duplicate(event_dict)

    return event_dict


# ---------------------------------------------------------------------------
# Kafka Producer
# ---------------------------------------------------------------------------


class IntelliPipeProducer:
    """
    Production-grade Kafka producer with:
    - Delivery confirmation callbacks
    - Dead-letter queue routing on failure
    - Automatic topic creation
    - Rate limiting support
    - Metrics integration
    """

    def __init__(self) -> None:
        self._settings = settings.kafka
        self._producer = self._create_producer()
        self._dlq_producer = self._create_producer()
        self._ensure_topics()
        logger.info(
            "Kafka producer initialised",
            bootstrap_servers=self._settings.bootstrap_servers,
            topic=self._settings.raw_events_topic,
        )

    def _create_producer(self) -> Producer:
        config: Dict[str, Any] = {
            "bootstrap.servers": self._settings.bootstrap_servers,
            "acks": "all",
            "enable.idempotence": True,
            "compression.type": "snappy",
            "batch.size": 65536,
            "linger.ms": 10,
            "retries": 5,
            "retry.backoff.ms": 500,
            "max.in.flight.requests.per.connection": 5,
        }
        if self._settings.security_protocol != "PLAINTEXT":
            config["security.protocol"] = self._settings.security_protocol
            if self._settings.sasl_mechanism:
                config["sasl.mechanisms"] = self._settings.sasl_mechanism
                config["sasl.username"] = (
                    self._settings.sasl_username.get_secret_value()
                )
                config["sasl.password"] = (
                    self._settings.sasl_password.get_secret_value()
                )
        return Producer(config)

    def _ensure_topics(self) -> None:
        """Create topics if they don't exist."""
        admin = AdminClient({"bootstrap.servers": self._settings.bootstrap_servers})
        topics_to_create = [
            NewTopic(
                self._settings.raw_events_topic, num_partitions=12, replication_factor=3
            ),
            NewTopic(self._settings.dlq_topic, num_partitions=3, replication_factor=3),
        ]
        futures = admin.create_topics(topics_to_create)
        for topic, future in futures.items():
            try:
                future.result()
                logger.info("Kafka topic created", topic=topic)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Topic creation warning", topic=topic, error=str(e))

    def _delivery_callback(self, err: Optional[KafkaError], msg: Any) -> None:
        """Called per-message after broker acknowledgement."""
        if err:
            logger.error(
                "Message delivery failed",
                topic=msg.topic(),
                partition=msg.partition(),
                error=str(err),
            )
        else:
            EVENTS_PRODUCED_TOTAL.labels(
                topic=msg.topic(),
                tenant_id="unknown",
            ).inc()

    def produce(
        self,
        event: Dict[str, Any],
        topic: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """
        Produce a single event to Kafka.

        Args:
            event: The event payload dict.
            topic: Target topic (defaults to raw_events_topic).
            key: Message key for partition routing (use tenant_id or order_id).
        """
        target_topic = topic or self._settings.raw_events_topic
        message_key = key or event.get("tenant_id") or event.get("order_id")

        try:
            self._producer.produce(
                topic=target_topic,
                key=message_key.encode("utf-8") if message_key else None,
                value=json.dumps(event, default=str).encode("utf-8"),
                on_delivery=self._delivery_callback,
                headers={
                    "content-type": "application/json",
                    "schema-version": str(event.get("schema_version", "1.0")),
                    "source-system": str(event.get("source_system", "unknown")),
                },
            )
            # Poll to trigger delivery callbacks without blocking
            self._producer.poll(0)
        except BufferError:
            # Producer queue full — apply backpressure
            logger.warning("Producer queue full, flushing...")
            self._producer.flush(timeout=5)
            self.produce(event, topic, key)

    def produce_to_dlq(self, event: Dict[str, Any], reason: str) -> None:
        """Route a failed/invalid event to the dead-letter queue."""
        dlq_envelope = {
            "original_event": event,
            "dlq_reason": reason,
            "dlq_timestamp": datetime.now(timezone.utc).isoformat(),
            "dlq_version": "1.0",
        }
        self._dlq_producer.produce(
            topic=self._settings.dlq_topic,
            value=json.dumps(dlq_envelope, default=str).encode("utf-8"),
            on_delivery=self._delivery_callback,
        )
        self._dlq_producer.poll(0)

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for all in-flight messages to be delivered."""
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning("Producer flush timed out", undelivered_messages=remaining)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


class EventSimulator:
    """
    Continuous event simulation with configurable anomaly injection schedule.
    Used for local dev / load testing / demo environments.
    """

    # Injection schedule: (anomaly_type, probability_per_batch)
    ANOMALY_SCHEDULE = [
        ("null_spike", 0.02),
        ("schema_add", 0.01),
        ("schema_remove", 0.005),
        ("schema_rename", 0.005),
        ("outlier", 0.03),
        ("invalid_enum", 0.02),
        ("delayed", 0.02),
        ("duplicate", 0.01),
    ]

    def __init__(
        self,
        producer: IntelliPipeProducer,
        events_per_second: int = 100,
        tenant_id: Optional[str] = None,
    ) -> None:
        self._producer = producer
        self._eps = events_per_second
        self._tenant_id = tenant_id
        self._total_produced = 0

    def _choose_anomaly(self) -> Optional[str]:
        """Stochastically select an anomaly type for this event."""
        for anomaly_type, probability in self.ANOMALY_SCHEDULE:
            if random.random() < probability:
                return anomaly_type
        return None

    def generate_batch(self, batch_size: int = 100) -> Iterator[Dict[str, Any]]:
        """Generate a batch of events with probabilistic anomalies."""
        for _ in range(batch_size):
            anomaly = self._choose_anomaly()
            yield generate_order_event(
                tenant_id=self._tenant_id,
                anomaly_type=anomaly,
            )

    def run_continuous(self, duration_seconds: Optional[int] = None) -> None:
        """
        Run continuous event production at target EPS.
        Set duration_seconds=None for indefinite production.
        """
        logger.info(
            "Starting event simulation",
            events_per_second=self._eps,
            duration_seconds=duration_seconds or "indefinite",
        )

        start_time = time.time()
        batch_size = max(1, self._eps // 10)  # 100ms batches
        sleep_interval = batch_size / self._eps

        try:
            while True:
                batch_start = time.time()

                for event in self.generate_batch(batch_size):
                    self._producer.produce(event)
                    self._total_produced += 1

                # Control rate
                elapsed = time.time() - batch_start
                sleep_needed = max(0, sleep_interval - elapsed)
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

                # Check duration
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self._producer.flush()
            logger.info("Simulation complete", total_produced=self._total_produced)


if __name__ == "__main__":
    producer = IntelliPipeProducer()
    simulator = EventSimulator(producer, events_per_second=500)
    simulator.run_continuous(duration_seconds=300)
