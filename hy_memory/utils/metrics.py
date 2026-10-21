"""
Agent Memory - Performance monitoring

Provides performance metric collection and statistics.
"""

import time
import functools
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading


@dataclass
class MetricRecord:
    """Metrics record for a single call."""
    method_name: str
    duration_ms: float
    timestamp: str
    success: bool = True
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class MemoryMetrics:
    """
    Memory system metrics collector (singleton pattern)
    
    Collects and aggregates the execution time of each method.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._records: Dict[str, List[MetricRecord]] = defaultdict(list)
        self._enabled = True
        self._print_enabled = False
        self._max_records = 1000  # keep at most 1000 records per method
        self._initialized = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    @property
    def print_enabled(self) -> bool:
        return self._print_enabled
    
    @print_enabled.setter
    def print_enabled(self, value: bool):
        self._print_enabled = value
    
    def record(self, record: MetricRecord) -> None:
        """Record one metric."""
        if not self._enabled:
            return
        
        with self._lock:
            records = self._records[record.method_name]
            records.append(record)
            
            # Cap the number of retained records
            if len(records) > self._max_records:
                self._records[record.method_name] = records[-self._max_records:]
        
        if self._print_enabled:
            status = "✓" if record.success else "✗"
            print(f"[Memory] {status} {record.method_name}: {record.duration_ms:.2f}ms")
    
    def get_stats(self, method_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics
        
        Args:
            method_name: method name; None returns statistics for all methods
        
        Returns:
            dictionary of statistics
        """
        if method_name:
            return self._calc_stats(method_name, self._records.get(method_name, []))
        
        return {
            name: self._calc_stats(name, records)
            for name, records in self._records.items()
        }
    
    def _calc_stats(self, name: str, records: List[MetricRecord]) -> Dict[str, Any]:
        """Compute statistics for a single method."""
        if not records:
            return {"method": name, "count": 0}
        
        durations = [r.duration_ms for r in records]
        success_count = sum(1 for r in records if r.success)
        
        return {
            "method": name,
            "count": len(records),
            "success_count": success_count,
            "error_count": len(records) - success_count,
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": sorted(durations)[len(durations) // 2],
            "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else max(durations),
        }
    
    def print_summary(self) -> None:
        """Print a summary of the statistics."""
        stats = self.get_stats()
        
        if not stats:
            print("[Memory Metrics] no records")
            return
        
        print("\n" + "=" * 80)
        print("Agent Memory Metrics Summary")
        print("=" * 80)
        print(f"{'Method':<30} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10}")
        print("-" * 80)
        
        total_time = 0
        total_count = 0
        
        for name, s in sorted(stats.items()):
            if s["count"] == 0:
                continue
            print(
                f"{s['method']:<30} {s['count']:>8} {s['total_ms']:>12.2f} "
                f"{s['avg_ms']:>10.2f} {s.get('p50_ms', 0):>10.2f} {s.get('p95_ms', 0):>10.2f}"
            )
            total_time += s["total_ms"]
            total_count += s["count"]
        
        print("-" * 80)
        print(f"{'TOTAL':<30} {total_count:>8} {total_time:>12.2f}")
        print("=" * 80 + "\n")
    
    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()
    
    def export(self) -> Dict[str, Any]:
        """Export all metrics."""
        return {
            "stats": self.get_stats(),
            "enabled": self._enabled,
            "record_count": sum(len(r) for r in self._records.values()),
        }


# Global singleton
metrics = MemoryMetrics()


def timed(method_name: Optional[str] = None):
    """
    Method timing decorator
    
    Args:
        method_name: custom method name; defaults to the function name
    
    Usage:
        @timed()
        def my_method(self, ...):
            ...
        
        @timed("custom_name")
        def another_method(self, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not metrics.enabled:
                return func(*args, **kwargs)
            
            name = method_name or func.__name__
            
            # For class methods, prefix the class name
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                name = f"{class_name}.{name}"
            
            start_time = time.perf_counter()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                record = MetricRecord(
                    method_name=name,
                    duration_ms=duration_ms,
                    timestamp=datetime.now().isoformat(),
                    success=success,
                    error=error,
                )
                metrics.record(record)
        
        return wrapper
    return decorator


def timed_async(method_name: Optional[str] = None):
    """Async method timing decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not metrics.enabled:
                return await func(*args, **kwargs)
            
            name = method_name or func.__name__
            
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                name = f"{class_name}.{name}"
            
            start_time = time.perf_counter()
            success = True
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                record = MetricRecord(
                    method_name=name,
                    duration_ms=duration_ms,
                    timestamp=datetime.now().isoformat(),
                    success=success,
                    error=error,
                )
                metrics.record(record)
        
        return wrapper
    return decorator
