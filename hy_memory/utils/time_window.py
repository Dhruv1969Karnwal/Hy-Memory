"""
Agent Memory - Time window management

Implements rolling time window deletion
"""

import threading
import time
from typing import Optional, Callable
from datetime import datetime, timedelta
import logging

from ..models import MemoryLayer
from ..config import TimeWindowConfig

logger = logging.getLogger(__name__)


class TimeWindowManager:
    """
    Time window manager
    
    Provides rolling time window deletion:
    - deletes expired memories automatically
    - supports a different window per layer
    - a background thread checks periodically
    """
    
    def __init__(
        self,
        config: Optional[TimeWindowConfig] = None,
        delete_callback: Optional[Callable] = None
    ):
        """
        Initialise the time window manager
        
        Args:
            config: time window configuration
            delete_callback: deletion callback function
        """
        self.config = config or TimeWindowConfig()
        self.delete_callback = delete_callback
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the background check thread."""
        if not self.config.enable_rolling_delete:
            logger.info("Rolling delete is disabled")
            return
        
        if self._running:
            logger.warning("TimeWindowManager is already running")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._check_loop,
            daemon=True,
            name="TimeWindowManager"
        )
        self._thread.start()
        logger.info("TimeWindowManager started")
    
    def stop(self):
        """Stop the background check thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("TimeWindowManager stopped")
    
    def _check_loop(self):
        """Background check loop."""
        while self._running:
            try:
                self._check_and_delete()
            except Exception as e:
                logger.error(f"Error in time window check: {e}")
            
            # wait for the next check
            time.sleep(self.config.rolling_check_interval)
    
    def _check_and_delete(self):
        """Check for and delete expired memories."""
        if not self.delete_callback:
            return
        
        now = datetime.now()
        
        # check layer by layer
        for layer_value, window_days in self.config.layer_windows.items():
            try:
                layer = MemoryLayer.from_string(layer_value)
            except ValueError:
                continue
            
            cutoff_time = now - timedelta(days=window_days)
            
            logger.debug(
                f"Checking layer {layer_value} for memories before {cutoff_time}"
            )
            
            # invoke the deletion callback
            try:
                deleted_count = self.delete_callback(
                    layer=layer,
                    before_time=cutoff_time
                )
                if deleted_count > 0:
                    logger.info(
                        f"Deleted {deleted_count} expired memories "
                        f"from layer {layer_value}"
                    )
            except Exception as e:
                logger.error(f"Failed to delete expired memories: {e}")
    
    def get_window_for_layer(self, layer: MemoryLayer) -> int:
        """
        Get the time window (in days) for a given layer
        
        Args:
            layer: memory layer
        
        Returns:
            time window (in days)
        """
        return self.config.layer_windows.get(
            layer.value,
            self.config.rolling_window_days
        )
    
    def set_window_for_layer(self, layer: MemoryLayer, days: int):
        """
        Set the time window for a given layer
        
        Args:
            layer: memory layer
            days: time window (in days)
        """
        with self._lock:
            self.config.layer_windows[layer.value] = days
    
    def is_expired(
        self,
        layer: MemoryLayer,
        event_time: datetime,
        reference_time: Optional[datetime] = None
    ) -> bool:
        """
        Check whether a memory has expired
        
        Args:
            layer: memory layer
            event_time: event time
            reference_time: reference time
        
        Returns:
            whether it has expired
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        window_days = self.get_window_for_layer(layer)
        cutoff_time = reference_time - timedelta(days=window_days)
        
        return event_time < cutoff_time
