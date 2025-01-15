from opentelemetry import (  # type: ignore
    metrics,
    trace
)
from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
from opentelemetry.sdk.metrics.export import (  # type: ignore
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider  # type: ignore
from typing import Optional

import logging


logger = logging.getLogger(__name__)


def configure_metrics():
    '''
    Configure the metric provider implementation.

    Currently implements a simple in-memory provider that can be used to log,
    store and read metrics.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        exporter = ConsoleMetricExporter()
        reader = PeriodicExportingMetricReader(exporter)
        provider = MeterProvider(metric_readers=[reader])

        metrics.set_meter_provider(provider)
    except Exception as e:
        logger.exception(e)
        raise e


def configure_tracer():
    '''
    Configure the tracer provider implementation.

    Currently implements a default in-memory provider that can be used to log,
    store and read traces.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        trace.set_tracer_provider(TracerProvider())
    except Exception as e:
        logger.exception(e)
        raise e


def get_counter(name: str, meter: Optional[metrics.Meter] = None):
    '''
    Fetch a counter metric using the currently configured metrics provider.

    Args:
        name: The name of the counter metric to create.
        meter: The meter object under which to create the metric counter.

    Returns:
        counter: An Opentelemetry counter.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        if not meter:
            meter = metrics.get_meter()
        return meter.create_counter(name)
    except Exception as e:
        logger.exception(e)
        raise e


def get_meter():
    '''
    Fetch a meter using the currently configured metrics provider.

    Args:
        None

    Returns:
        meter: An Opentelemetry meter.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        provider = metrics.get_meter_provider()

        return metrics.get_meter(provider)
    except Exception as e:
        logger.exception(e)
        raise e
