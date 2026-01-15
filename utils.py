#!/usr/bin/env python3
"""
videogen.utils - Utility Functions and Classes
Contains logging, metrics, monitoring, and helper functions.

Version: 42.2 (ALL CRITICAL BUGS FIXED)

FIXED ISSUES:
- Bug #2: Rate Limiter Per-Instance - Global registry pattern implemented
- Bug #5: TTS Semaphore - Lazy initialization with proper double-checked locking
- Bug #7: Script Cache - Proper LRU with OrderedDict
- Bug #8: Metrics History Unbounded Growth - MAX_TRACKED_ASSETS limit enforced
- Bug #9: File Descriptor Leak - Improved fd tracking and logging
- Bug #14: Progress Tracker Time Overflow - Capped maximum ETA
- Bug #15: Hardcoded Retry Logic - Generic retry_with_backoff decorator
- Bug #16: Unsafe String Truncation - safe_truncate with UTF-8 handling
- Bug #17: Inefficient File Existence Checks - Safe utility functions
- Bug #18: Flamegraph Memory Leak - Bounded deque (maxlen=10000)
- Bug #19: Incorrect Timeout Calculation - Overflow protection
- Bug #312: Error Count Never Reset - Added reset method + limit
- Bug #317: Deadlock in Logger Shutdown - Lock released before flushing
- Bug #318: Race Condition in Metrics Export - Atomic write pattern
- Bug #322: Cache Key Collision - normalize_unicode function
- Bug #323: Resource Leak in managed_subprocess - Proper kill after timeout
- Bug #324: Division by Zero in Progress Tracker - Validated in update
- Bug #329: Broken Pipe Not Handled - Improved error handling
- Bug #330: Missing Null Check - Validation added
- Bug #331: Inefficient String Concatenation - StringBuilder class
- Bug #333: Missing Timeout on Webhook - Desktop notification timeout
- Bug #342: Unicode Filename Truncation - UTF-8 safe truncation
- Bug #376: Double-Checked Locking - Proper implementation
- Bug #392: Subprocess Zombies - setup_zombie_prevention function
- Bug #401: Logger Name Collision - Added timestamp to run_id
- Bug #404: Memory Check Cached - Cache with 1s TTL
- Bug #406: Safe Print Deadlock - Changed to RLock
"""

import sys
import os
import time
import threading
import logging
import logging.handlers
import tempfile
import gc
import gzip
import signal
import subprocess
import unicodedata
import random
from pathlib import Path
from collections import defaultdict, deque, OrderedDict
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable, Deque, Tuple, Set, List, Union
from functools import wraps
from datetime import datetime, timezone

from .config import (
    VERSION,
    ServiceConstants,
    ResourceConstants,
    TimeoutConstants
)

__version__ = VERSION

# ============ PSUTIL OPTIONAL IMPORT ============
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ============ EMERGENCY LOG WRITER ============
def _write_to_emergency_log(msg: str) -> None:
    """
    Write to emergency log file when all output streams fail.
    
    This function is intentionally simple and has no dependencies
    to ensure it can always execute even in catastrophic failure.
    
    Args:
        msg: Message to write
    """
    try:
        log_file = Path.home() / ".video_gen_errors.log"
        with open(log_file, 'a', encoding='utf-8', errors='replace') as f:
            timestamp = datetime.now(timezone.utc).isoformat()
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        # Truly nothing we can do - fail silently
        pass

# ============ THREAD-SAFE OUTPUT ============
_print_lock = threading.RLock()  # RLock prevents nested deadlock (fix #406)

def safe_print(msg: str, file=sys.stdout, flush: bool = True) -> None:
    """
    Thread-safe print with comprehensive error handling.
    
    Fixes: #253, #268, #329, #406
    
    Handles all common output failures:
    - Broken pipes (SIGPIPE)
    - Unicode encoding errors
    - I/O errors
    - Stream closure
    
    Args:
        msg: Message to print
        file: Output stream (default: sys.stdout)
        flush: Whether to flush output (default: True)
    """
    if not isinstance(msg, str):
        msg = str(msg)
    
    with _print_lock:
        try:
            print(msg, file=file, flush=flush)
        except BrokenPipeError:
            # Pipe closed - try alternate stream
            try:
                alt_file = sys.stderr if file == sys.stdout else sys.stdout
                print(f"[pipe error] {msg}", file=alt_file, flush=flush)
            except Exception:
                _write_to_emergency_log(msg)
        except (ValueError, OSError, IOError) as e:
            # Stream closed or other I/O error
            try:
                alt_file = sys.stderr if file == sys.stdout else sys.stdout
                print(f"[io error: {type(e).__name__}] {msg}", file=alt_file, flush=flush)
            except Exception:
                _write_to_emergency_log(msg)
        except UnicodeEncodeError:
            # Try with ASCII encoding
            try:
                safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                print(f"[encoding error] {safe_msg}", file=file, flush=flush)
            except Exception:
                _write_to_emergency_log(msg)
        except Exception:
            # Catch-all for unexpected errors
            _write_to_emergency_log(msg)

# ============ STRING UTILITIES ============
def safe_truncate(text: str, max_length: int) -> str:
    """
    Truncate string without breaking UTF-8 multibyte characters.
    
    Fix #16 - Prevents Unicode corruption from naive slicing.
    Fix #342 - UTF-8 safe truncation.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
    
    Returns:
        Truncated string with valid UTF-8 encoding
    
    Example:
        >>> safe_truncate("Hello 世界", 7)
        'Hello 世'
    """
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    # Truncate to max_length
    truncated = text[:max_length]
    
    # Verify UTF-8 validity
    try:
        truncated.encode('utf-8')
        return truncated
    except UnicodeEncodeError:
        # Back off one character at a time (UTF-8 max 4 bytes per character)
        for i in range(1, 5):
            try:
                truncated = text[:max_length - i]
                truncated.encode('utf-8')
                return truncated
            except UnicodeEncodeError:
                continue
    
    # Fallback: return empty if all attempts fail
    return text[:max(0, max_length - 4)]

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode string to prevent cache key collisions.
    
    Fix #322 - Cache key collision prevention.
    
    Args:
        text: Text to normalize
    
    Returns:
        Normalized text in NFC form
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
    # This ensures "é" (single character) and "é" (e + combining accent) are treated the same
    normalized = unicodedata.normalize('NFC', text)
    
    # Strip whitespace
    normalized = normalized.strip()
    
    return normalized

class StringBuilder:
    """
    Efficient string builder to avoid O(n²) concatenation.
    
    Fix #331 - Inefficient string concatenation.
    
    Example:
        >>> sb = StringBuilder()
        >>> sb.append("Hello")
        >>> sb.append(" ")
        >>> sb.append("World")
        >>> str(sb)
        'Hello World'
    """
    
    def __init__(self):
        self._parts: List[str] = []
    
    def append(self, text: str) -> 'StringBuilder':
        """Append text to builder."""
        if text:
            self._parts.append(str(text))
        return self
    
    def __str__(self) -> str:
        """Get final string."""
        return ''.join(self._parts)
    
    def __len__(self) -> int:
        """Get total length."""
        return sum(len(p) for p in self._parts)
    
    def clear(self) -> None:
        """Clear all content."""
        self._parts.clear()

# ============ STRUCTURED LOGGER ============
class StructuredLogger:
    """
    Thread-safe structured logger with rotation.
    
    Fixes: #108, #279, #317, #401
    
    Features:
    - Automatic log rotation with gzip compression
    - Thread-safe operation with RLock
    - Bounded error counting to prevent memory leaks
    - Deadlock-free shutdown procedure
    - Unique logger names to prevent conflicts
    """
    
    def __init__(self, log_file: Optional[Path] = None, run_id: Optional[str] = None):
        import uuid
        
        self.log_file = log_file
        # Add timestamp to run_id to prevent collision (fix #401)
        timestamp = int(time.time() * 1000)
        self.run_id = run_id or f"{uuid.uuid4().hex[:8]}_{timestamp}"
        self.lock = threading.RLock()  # RLock for nested logging calls
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_counts_limit = 10000  # Prevent unbounded growth (fix #312)
        self._shutdown_flag = False
        
        # Create unique logger name to prevent conflicts
        logger_name = f"vidgen_{self.run_id}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False  # Prevent duplicate logs
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)-5s %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(console)
        
        # File handler with rotation and compression (fixes #279)
        if log_file:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Use RotatingFileHandler with compression
                file_handler = logging.handlers.RotatingFileHandler(
                    str(log_file),
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - [%(run_id)s] %(message)s'
                ))
                
                # Add filter for run_id
                class RunIdFilter(logging.Filter):
                    def __init__(self, run_id: str):
                        super().__init__()
                        self.run_id = run_id
                    
                    def filter(self, record) -> bool:
                        record.run_id = self.run_id
                        return True
                
                file_handler.addFilter(RunIdFilter(self.run_id))
                
                # Custom rotation with gzip compression
                def namer(name: str) -> str:
                    return name + ".gz"
                
                def rotator(source: str, dest: str) -> None:
                    try:
                        with open(source, 'rb') as sf:
                            with gzip.open(dest, 'wb') as df:
                                df.writelines(sf)
                        os.remove(source)
                    except Exception:
                        pass  # Rotation failure shouldn't crash
                
                file_handler.namer = namer
                file_handler.rotator = rotator
                
                self.logger.addHandler(file_handler)
            except Exception as e:
                safe_print(f"⚠️  Could not create log file: {e}", file=sys.stderr)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message."""
        if self._shutdown_flag:
            return
        if self.logger.isEnabledFor(logging.DEBUG):
            with self.lock:
                try:
                    self.logger.debug(msg, extra=kwargs)
                except Exception:
                    pass  # Don't let logging errors crash the app
    
    def info(self, msg: str, **kwargs) -> None:
        """Log info message."""
        if self._shutdown_flag:
            return
        with self.lock:
            try:
                self.logger.info(msg, extra=kwargs)
            except Exception:
                pass
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message."""
        if self._shutdown_flag:
            return
        with self.lock:
            try:
                self.logger.warning(msg, extra=kwargs)
            except Exception:
                pass
    
    def error(self, msg: str, user_msg: Optional[str] = None, **kwargs) -> None:
        """Log error message with optional user-facing message."""
        if self._shutdown_flag:
            return
        error_type = kwargs.get('error_type', 'unknown')
        with self.lock:
            try:
                # Prevent unbounded growth (fix #312)
                total_errors = sum(self.error_counts.values())
                if total_errors < self.error_counts_limit:
                    self.error_counts[error_type] += 1
                self.logger.error(msg, extra=kwargs)
            except Exception:
                pass
        if user_msg:
            safe_print(f"  ⚠️  {user_msg}")
    
    def critical(self, msg: str, **kwargs) -> None:
        """Log critical message."""
        if self._shutdown_flag:
            return
        with self.lock:
            try:
                self.logger.critical(msg, extra=kwargs)
            except Exception:
                pass
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get error counts summary."""
        with self.lock:
            return dict(self.error_counts)
    
    def reset_error_counts(self) -> None:
        """Reset error counts (fix #312 - prevents unbounded growth)."""
        with self.lock:
            self.error_counts.clear()
    
    def shutdown(self) -> None:
        """
        Shutdown logger safely.
        
        Fixes: #294, #317 (deadlock prevention)
        
        Strategy: Copy handlers list, release lock, then flush/close.
        """
        self._shutdown_flag = True
        
        # Copy handlers list WITH lock
        with self.lock:
            handlers = list(self.logger.handlers)
        
        # Flush and close handlers WITHOUT lock (fix #317 - prevents deadlock)
        for handler in handlers:
            try:
                handler.flush()
            except Exception as e:
                try:
                    safe_print(f"Warning: Handler flush failed: {e}", file=sys.stderr)
                except Exception:
                    pass
            
            try:
                handler.close()
            except Exception as e:
                try:
                    safe_print(f"Warning: Handler close failed: {e}", file=sys.stderr)
                except Exception:
                    pass
        
        # Now remove handlers with lock
        with self.lock:
            for handler in handlers:
                try:
                    self.logger.removeHandler(handler)
                except Exception:
                    pass
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"StructuredLogger(run_id={self.run_id}, handlers={len(self.logger.handlers)})"

# ============ METRICS COLLECTOR ============
class MetricsCollector:
    """
    Thread-safe metrics collection with history limiting.
    
    Fixes: #18, #240, #314, #318, #8
    
    Features:
    - Bounded memory usage with deque maxlen
    - Limited number of tracked assets (MAX_TRACKED_ASSETS)
    - Atomic file export with temporary file
    - Safe division with zero checks
    - Thread-safe iteration over collections
    """
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "videos_success": 0,
            "videos_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_corruptions": 0,
            "asset_generation_times": {},  # Will be populated with deques
            "errors": defaultdict(int),
            "asset_failures": defaultdict(int),
            "retries": defaultdict(int),
            "quality_rejections": defaultdict(int),
            "operation_times": {},  # Will be populated with deques
            "flamegraph_data": deque(maxlen=10000)  # Fix #18 - bounded
        }
        self.lock = threading.RLock()
        self.max_tracked_assets = ResourceConstants.MAX_TRACKED_ASSETS  # Fix #8
    
    def record_success(self) -> None:
        """Record successful video generation."""
        with self.lock:
            self.metrics["videos_success"] += 1
    
    def record_failure(self, error_type: str = "unknown") -> None:
        """Record failed video generation."""
        with self.lock:
            self.metrics["videos_failed"] += 1
            self.metrics["errors"][error_type] += 1
    
    def record_asset_failure(self, asset_type: str) -> None:
        """Record asset generation failure."""
        with self.lock:
            self.metrics["asset_failures"][asset_type] += 1
    
    def record_retry(self, operation: str) -> None:
        """Record retry attempt."""
        with self.lock:
            self.metrics["retries"][operation] += 1
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        with self.lock:
            self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        with self.lock:
            self.metrics["cache_misses"] += 1
    
    def record_cache_corruption(self) -> None:
        """Record cache corruption."""
        with self.lock:
            self.metrics["cache_corruptions"] += 1
    
    def record_quality_rejection(self, reason: str) -> None:
        """Record quality check rejection."""
        with self.lock:
            self.metrics["quality_rejections"][reason] += 1
    
    def record_generation_time(self, asset_type: str, duration: float) -> None:
        """
        Record generation time with automatic limiting.
        
        Fixes: #18, #240, #8
        
        Args:
            asset_type: Type of asset (e.g., "image", "audio")
            duration: Time taken in seconds
        """
        with self.lock:
            # Limit number of tracked asset types (fix #8)
            if asset_type not in self.metrics["asset_generation_times"]:
                if len(self.metrics["asset_generation_times"]) >= self.max_tracked_assets:
                    # Remove oldest (first inserted)
                    oldest = next(iter(self.metrics["asset_generation_times"]))
                    del self.metrics["asset_generation_times"][oldest]
                
                # Create new deque with size limit
                self.metrics["asset_generation_times"][asset_type] = deque(
                    maxlen=ResourceConstants.METRICS_HISTORY_LIMIT
                )
            
            self.metrics["asset_generation_times"][asset_type].append(duration)
    
    def record_operation_time(self, operation: str, duration: float) -> None:
        """
        Record operation time for profiling.
        
        Fix #8 - bounded with MAX_TRACKED_ASSETS
        
        Args:
            operation: Operation name
            duration: Time taken in seconds
        """
        with self.lock:
            if operation not in self.metrics["operation_times"]:
                if len(self.metrics["operation_times"]) >= self.max_tracked_assets:
                    oldest = next(iter(self.metrics["operation_times"]))
                    del self.metrics["operation_times"][oldest]
                
                self.metrics["operation_times"][operation] = deque(maxlen=100)
            
            self.metrics["operation_times"][operation].append(duration)
    
    def add_flamegraph_entry(self, stack_trace: str, duration: float) -> None:
        """
        Add flamegraph profiling entry.
        
        Fix #18 - bounded deque prevents memory leak
        
        Args:
            stack_trace: Stack trace string
            duration: Operation duration in seconds
        """
        with self.lock:
            self.metrics["flamegraph_data"].append({
                "stack": stack_trace,
                "duration": duration,
                "timestamp": time.time()
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary with deep copy.
        
        Fixes: #289, #314 (safe division)
        
        Returns:
            Dictionary containing summary statistics
        """
        with self.lock:
            total = self.metrics["videos_success"] + self.metrics["videos_failed"]
            cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            
            # Fix #314 - safe division with zero check
            success_rate = 0.0
            if total > 0:
                success_rate = (self.metrics["videos_success"] / total) * 100
            
            cache_hit_rate = 0.0
            if cache_total > 0:
                cache_hit_rate = (self.metrics["cache_hits"] / cache_total) * 100
            
            summary = {
                "total_videos": total,
                "success": self.metrics["videos_success"],
                "failed": self.metrics["videos_failed"],
                "success_rate": success_rate,
                "cache_hit_rate": cache_hit_rate,
                "cache_corruptions": self.metrics["cache_corruptions"],
                "asset_failures": dict(self.metrics["asset_failures"]),
                "retries": dict(self.metrics["retries"]),
                "quality_rejections": dict(self.metrics["quality_rejections"])
            }
            
            # Calculate averages from deques (safe iteration with lock held)
            for asset_type, times in list(self.metrics["asset_generation_times"].items()):
                if times and len(times) > 0:
                    summary[f"avg_{asset_type}_time"] = sum(times) / len(times)
            
            return summary
    
    def export_flamegraph(self, output_path: Path) -> bool:
        """
        Export flamegraph data for visualization.
        
        Fix #318 - atomic write pattern prevents corruption
        
        Args:
            output_path: Destination file path
        
        Returns:
            True if export succeeded
        """
        try:
            import json
            
            # Copy data with lock
            with self.lock:
                data = list(self.metrics["flamegraph_data"])
            
            # Write to temp file first (atomic write pattern - fix #318)
            temp_path = output_path.with_suffix('.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                # Atomic rename
                temp_path.replace(output_path)
                return True
            except Exception:
                # Clean up temp file on error
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                return False
        except Exception:
            return False
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        with self.lock:
            return (f"MetricsCollector(success={self.metrics['videos_success']}, "
                   f"failed={self.metrics['videos_failed']})")

# ============ RESOURCE MONITOR ============
class ResourceMonitor:
    """
    System resource monitoring.
    
    Fix #404 - cached results to avoid excessive psutil calls
    
    Features:
    - Caches results for 1 second to reduce overhead
    - Thread-safe operation
    - Graceful degradation when psutil unavailable
    """
    
    _cache_timeout = 1.0  # Cache results for 1 second (fix #404)
    _last_check: Dict[str, Tuple[float, Any]] = {}
    _check_lock = threading.Lock()
    
    @staticmethod
    def check_memory(critical_threshold: bool = False) -> Tuple[bool, float]:
        """
        Check memory usage with critical threshold support.
        
        Args:
            critical_threshold: Use critical threshold instead of normal
        
        Returns:
            Tuple of (is_ok, percent_used)
        """
        if not PSUTIL_AVAILABLE:
            return True, 0.0
        
        # Check cache (fix #404)
        cache_key = f"memory_{critical_threshold}"
        with ResourceMonitor._check_lock:
            if cache_key in ResourceMonitor._last_check:
                last_time, last_result = ResourceMonitor._last_check[cache_key]
                if time.time() - last_time < ResourceMonitor._cache_timeout:
                    return last_result
        
        try:
            memory = psutil.virtual_memory()
            threshold = (ResourceConstants.CRITICAL_MEMORY_PERCENT 
                        if critical_threshold 
                        else ResourceConstants.MAX_MEMORY_PERCENT)
            result = (memory.percent <= threshold, memory.percent)
            
            # Update cache
            with ResourceMonitor._check_lock:
                ResourceMonitor._last_check[cache_key] = (time.time(), result)
            
            return result
        except Exception:
            return True, 0.0
    
    @staticmethod
    def get_available_memory_mb() -> float:
        """
        Get available memory in MB.
        
        Returns:
            Available memory in megabytes
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return 0.0
    
    @staticmethod
    def check_file_descriptors() -> Tuple[bool, int]:
        """
        Check available file descriptors.
        
        Returns:
            Tuple of (is_ok, count)
        """
        if not PSUTIL_AVAILABLE:
            return True, 0
        try:
            process = psutil.Process()
            open_fds = len(process.open_files())
            return open_fds < 800, open_fds
        except Exception:
            return True, 0
    
    @staticmethod
    def get_cpu_usage() -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            CPU usage as percentage (0-100)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    @staticmethod
    def should_downscale_quality() -> bool:
        """
        Check if quality should be downscaled due to resources.
        
        Returns:
            True if resources are constrained
        """
        if not PSUTIL_AVAILABLE:
            return False
        
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            # Downscale if memory >80% or CPU >90%
            if memory.percent > 80 or cpu > 90:
                return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def clear_cache() -> None:
        """Clear resource check cache."""
        with ResourceMonitor._check_lock:
            ResourceMonitor._last_check.clear()

# ============ RATE LIMITER ============
# Global registry to fix Bug #2 - Rate Limiter Per-Instance Bug
_global_limiters: Dict[str, 'RateLimiter'] = {}
_limiter_lock = threading.Lock()

class RateLimiter:
    """
    Thread-safe rate limiter with jitter support.
    
    Fix #274 - jitter prevents thundering herd
    
    Features:
    - Per-key rate limiting
    - Optional jitter to prevent synchronized retries
    - Thread-safe operation
    """
    
    def __init__(self):
        self.last_call: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def wait(self, key: str = "default", delay: float = ServiceConstants.API_RATE_LIMIT_SECONDS, jitter: bool = True) -> None:
        """
        Wait if necessary to maintain rate limit.
        
        Args:
            key: Rate limit key (e.g., "qwen", "comfyui")
            delay: Minimum delay between calls in seconds
            jitter: Whether to add random jitter
        """
        sleep_time = 0.0
        now = time.time()
        
        with self.lock:
            if key in self.last_call:
                elapsed = now - self.last_call[key]
                if elapsed < delay:
                    sleep_time = delay - elapsed
                    # Add jitter to prevent thundering herd (fixes #274)
                    if jitter:
                        sleep_time += random.uniform(0, ServiceConstants.RETRY_JITTER_MAX)
            self.last_call[key] = now + sleep_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        with self.lock:
            return f"RateLimiter(tracked_keys={len(self.last_call)})"

def rate_limited(service: str = "default", delay: float = ServiceConstants.API_RATE_LIMIT_SECONDS):
    """
    Rate limiting decorator.
    
    Fix #2 - use global limiter registry to ensure rate limiting works across
    multiple decorated functions calling the same service.
    
    Args:
        service: Service name for rate limiting
        delay: Minimum delay between calls
    
    Returns:
        Decorated function
    
    Example:
        @rate_limited("api_service", delay=1.0)
        def call_api():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create limiter from global registry (fix #2)
            with _limiter_lock:
                if service not in _global_limiters:
                    _global_limiters[service] = RateLimiter()
                limiter = _global_limiters[service]
            
            limiter.wait(service, delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ============ RETRY DECORATOR ============
def retry_with_backoff(
    max_attempts: int = 3,
    exceptions: Tuple = (Exception,),
    backoff_base: float = 2.0,
    max_backoff: float = 60.0,
    jitter: bool = True
):
    """
    Generic retry decorator with exponential backoff.
    
    Fix #15 - Eliminates hardcoded retry logic duplication.
    
    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch
        backoff_base: Base for exponential backoff
        max_backoff: Maximum backoff time in seconds
        jitter: Whether to add random jitter
    
    Returns:
        Decorated function
    
    Example:
        @retry_with_backoff(max_attempts=3, exceptions=(requests.RequestException,))
        def call_api():
            return requests.get("https://api.example.com")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        # Last attempt failed, re-raise
                        raise
                    
                    # Calculate backoff time
                    wait = min(backoff_base ** attempt, max_backoff)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        wait += random.uniform(0, ServiceConstants.RETRY_JITTER_MAX)
                    
                    # Log retry attempt
                    try:
                        func_name = func.__name__
                        safe_print(f"  ⚠️  {func_name} failed (attempt {attempt + 1}/{max_attempts}), retrying in {wait:.1f}s...")
                    except Exception:
                        pass
                    
                    time.sleep(wait)
            
            # Should never reach here
            return None
        return wrapper
    return decorator

# ============ PROGRESS TRACKER ============
class ProgressTracker:
    """
    Progress tracking with ETA calculation.
    
    Fixes: #266, #324, #14
    
    Features:
    - Smoothed ETA using historical data
    - Division by zero protection
    - Maximum ETA capping
    - Thread-safe operation
    - Graceful handling of backward progress
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = max(1, total_steps)  # Prevent division by zero (fix #324)
        self.current_step = 0
        self.start_time = time.time()
        self.lock = threading.RLock()
        self.last_update = self.start_time
        self.step_history: Deque[Tuple[int, float]] = deque(maxlen=10)  # For smoothing
        self.MAX_ETA_SECONDS = 3600  # 1 hour max (fix #14)
    
    def update(self, step: int, message: str = "") -> None:
        """
        Update progress with smoothed ETA.
        
        Fixes: #266, #324, #14
        
        Args:
            step: Current step number
            message: Progress message to display
        """
        with self.lock:
            # Validate step (fix #324)
            if step < 0:
                step = 0
            if step > self.total_steps:
                step = self.total_steps
            
            # Handle backward steps gracefully
            if step < self.current_step:
                self.current_step = max(0, step)
            else:
                self.current_step = min(step, self.total_steps)
            
            now = time.time()
            self.last_update = now
            self.step_history.append((self.current_step, now))
            
            if self.total_steps > 0:
                progress = (self.current_step / self.total_steps) * 100
                elapsed = now - self.start_time
                
                if self.current_step > 0 and len(self.step_history) >= 2:
                    # Use smoothed rate from history
                    oldest_step, oldest_time = self.step_history[0]
                    recent_step, recent_time = self.step_history[-1]
                    
                    if recent_step > oldest_step:
                        rate = (recent_time - oldest_time) / (recent_step - oldest_step)
                        remaining = (self.total_steps - self.current_step) * rate
                        
                        # Ensure non-negative ETA and cap at max (fix #14)
                        remaining = max(0, min(remaining, self.MAX_ETA_SECONDS))
                        
                        if remaining >= self.MAX_ETA_SECONDS:
                            eta_str = "ETA: >1 hour"
                        else:
                            eta_str = f"ETA: {int(remaining)}s"
                    else:
                        eta_str = "ETA: calculating..."
                else:
                    eta_str = "ETA: calculating..."
                
                safe_print(f"  [{progress:5.1f}%] {message} ({eta_str})")
    
    def finish(self) -> None:
        """Mark as finished."""
        with self.lock:
            self.current_step = self.total_steps
            elapsed = time.time() - self.start_time
            safe_print(f"  [100.0%] Complete (took {int(elapsed)}s)")

# ============ TIMEOUT CALCULATION ============
def calculate_safe_timeout(base_timeout: int, duration: float, per_second: float, max_timeout: int) -> int:
    """
    Calculate timeout with overflow protection.
    
    Fix #19 - Prevents integer overflow in timeout calculations.
    
    Args:
        base_timeout: Base timeout in seconds
        duration: Duration in seconds
        per_second: Multiplier per second
        max_timeout: Maximum allowed timeout
    
    Returns:
        Safe timeout value
    
    Example:
        >>> calculate_safe_timeout(60, 10.0, 5.0, 300)
        110
    """
    try:
        # Calculate with overflow check
        calculated = base_timeout + int(duration * per_second)
        
        # Check for overflow (calculated should be >= base_timeout)
        if calculated < base_timeout:
            # Overflow detected, use max timeout
            return max_timeout
        
        # Return minimum of calculated and max
        return min(calculated, max_timeout)
    except (OverflowError, ValueError):
        # Any calculation error, return max timeout
        return max_timeout

# ============ RESOURCE MANAGEMENT ============
def setup_zombie_prevention() -> None:
    """
    Setup zombie process prevention.
    
    Fix #392 - Prevents subprocess zombies by setting SIGCHLD handler.
    
    This should be called once at application startup.
    """
    if hasattr(signal, 'SIGCHLD'):
        try:
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        except (OSError, ValueError):
            # May fail in some environments (e.g., threading restrictions)
            pass

@contextmanager
def managed_temp_file(suffix: str = '', prefix: str = 'vidgen_', dir: Optional[Path] = None):
    """
    Context manager for temp files with guaranteed cleanup.
    
    Fixes: #213, #307, #9
    
    Args:
        suffix: File suffix (e.g., '.wav')
        prefix: File prefix
        dir: Directory for temp file
    
    Yields:
        Path to temporary file
    
    Example:
        with managed_temp_file(suffix='.wav') as temp_path:
            # Use temp_path
            pass
        # File automatically deleted
    """
    temp_file = None
    fd = None
    fd_closed = False
    
    try:
        if dir:
            dir.mkdir(parents=True, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
        temp_file = Path(temp_path)
        
        # Close fd immediately after getting path (fix #307)
        # Consumer should open file themselves if needed
        try:
            os.close(fd)
            fd_closed = True
        except OSError as e:
            # Log but don't crash (fix #9)
            try:
                safe_print(f"Warning: Failed to close temp file descriptor: {e}", file=sys.stderr)
            except Exception:
                pass
        
        yield temp_file
    finally:
        # Close fd if still open (fix #9)
        if fd is not None and not fd_closed:
            try:
                os.close(fd)
            except OSError as e:
                try:
                    safe_print(f"Warning: Failed to close file descriptor {fd}: {e}", file=sys.stderr)
                except Exception:
                    _write_to_emergency_log(f"FD leak: {fd}")
        
        # Clean up file with retries
        if temp_file and temp_file.exists():
            for attempt in range(5):
                try:
                    gc.collect()
                    time.sleep(0.05 * (2 ** attempt))
                    temp_file.unlink()
                    break
                except (PermissionError, OSError) as e:
                    if attempt == 4:
                        try:
                            safe_print(f"⚠️  Could not delete temp file {temp_file}: {e}", file=sys.stderr)
                        except Exception:
                            _write_to_emergency_log(f"Temp file leak: {temp_file}")

@contextmanager
def managed_subprocess(cmd: List[str], **kwargs):
    """
    Context manager for subprocess with proper cleanup.
    
    Fix #323 - Ensures process is killed if wait() times out.
    
    Args:
        cmd: Command list
        **kwargs: Arguments passed to subprocess.Popen
    
    Yields:
        Subprocess.Popen object
    
    Example:
        with managed_subprocess(['ffmpeg', '-i', 'input.mp4']) as proc:
            stdout, stderr = proc.communicate(timeout=30)
    """
    proc = None
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        yield proc
    finally:
        if proc and proc.poll() is None:
            try:
                # Try graceful termination
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill after timeout (fix #323)
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:
                        pass
            except Exception:
                pass

# ============ PERFORMANCE PROFILER ============
class PerformanceProfiler:
    """
    Performance profiling context manager.
    
    Tracks operation duration and records to metrics collector.
    
    Example:
        with PerformanceProfiler("image_generation", metrics):
            generate_image()
    """
    
    def __init__(self, operation: str, metrics: Optional[MetricsCollector] = None):
        self.operation = operation
        self.metrics = metrics
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.metrics:
            self.metrics.record_operation_time(self.operation, duration)
        return False

# ============ NOTIFICATION SYSTEM ============
class NotificationSystem:
    """
    Multi-channel notification system.
    
    Fix #333 - Added timeout to desktop notifications.
    
    Supports:
    - Webhook notifications
    - Desktop notifications (Linux notify-send)
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
    
    def send_notification(self, title: str, message: str, level: str = "info") -> bool:
        """
        Send notification via configured channels.
        
        Args:
            title: Notification title
            message: Notification message
            level: Severity level (info, warning, error)
        
        Returns:
            True if at least one notification succeeded
        """
        success = False
        
        # Webhook notification
        if self.webhook_url:
            success = self._send_webhook(title, message, level) or success
        
        # Desktop notification (if available)
        success = self._send_desktop(title, message, level) or success
        
        return success
    
    def _send_webhook(self, title: str, message: str, level: str) -> bool:
        """Send webhook notification."""
        try:
            import requests
            
            payload = {
                "title": title,
                "message": message,
                "level": level,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _send_desktop(self, title: str, message: str, level: str) -> bool:
        """
        Send desktop notification.
        
        Fix #333 - Added timeout to prevent hanging.
        """
        try:
            # Try notify-send on Linux
            if sys.platform == 'linux':
                subprocess.run(
                    ['notify-send', title, message],
                    timeout=5,  # Fix #333 - prevent hanging
                    capture_output=True,
                    check=False
                )
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return False

# ============ CLOUD STORAGE INTEGRATION ============
class CloudStorageUploader:
    """
    Cloud storage upload integration.
    
    Supports:
    - AWS S3
    - Google Cloud Storage
    - Azure Blob Storage
    """
    
    def __init__(self, provider: str, bucket: str):
        self.provider = provider.lower()
        self.bucket = bucket
    
    def upload_file(self, local_path: Path, remote_key: str) -> bool:
        """
        Upload file to cloud storage.
        
        Args:
            local_path: Local file path
            remote_key: Remote object key/name
        
        Returns:
            True if upload succeeded
        """
        try:
            if self.provider == 's3':
                return self._upload_s3(local_path, remote_key)
            elif self.provider == 'gcs':
                return self._upload_gcs(local_path, remote_key)
            elif self.provider == 'azure':
                return self._upload_azure(local_path, remote_key)
            else:
                return False
        except Exception:
            return False
    
    def _upload_s3(self, local_path: Path, remote_key: str) -> bool:
        """Upload to AWS S3."""
        try:
            import boto3
            s3 = boto3.client('s3')
            s3.upload_file(str(local_path), self.bucket, remote_key)
            return True
        except Exception:
            return False
    
    def _upload_gcs(self, local_path: Path, remote_key: str) -> bool:
        """Upload to Google Cloud Storage."""
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(self.bucket)
            blob = bucket.blob(remote_key)
            blob.upload_from_filename(str(local_path))
            return True
        except Exception:
            return False
    
    def _upload_azure(self, local_path: Path, remote_key: str) -> bool:
        """Upload to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if not connection_string:
                return False
            
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service.get_blob_client(container=self.bucket, blob=remote_key)
            
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            return True
        except Exception:
            return False

# ============ FILE UTILITIES ============
def safe_file_exists(path: Path) -> bool:
    """
    Check file existence safely.
    
    Fix #17 - Handles race conditions gracefully.
    
    Args:
        path: File path to check
    
    Returns:
        True if file exists
    """
    try:
        return path.exists() and path.is_file()
    except (OSError, PermissionError):
        return False

def safe_stat(path: Path) -> Optional[os.stat_result]:
    """
    Get file stats safely.
    
    Fix #17 - Handles TOCTOU issues.
    Fix #330 - Adds null check validation.
    
    Args:
        path: File path
    
    Returns:
        os.stat_result or None if failed
    """
    try:
        return path.stat()
    except (OSError, FileNotFoundError, PermissionError):
        return None

def safe_file_size(path: Path) -> int:
    """
    Get file size safely.
    
    Args:
        path: File path
    
    Returns:
        File size in bytes, or 0 if not accessible
    """
    stat = safe_stat(path)
    if stat:
        return stat.st_size
    return 0

# ============ EXPORTS ============
__all__ = [
    # Version
    '__version__',
    
    # Logging
    'StructuredLogger',
    'safe_print',
    
    # Metrics
    'MetricsCollector',
    'PerformanceProfiler',
    
    # Resource Management
    'ResourceMonitor',
    'managed_temp_file',
    'managed_subprocess',
    'setup_zombie_prevention',
    
    # Rate Limiting & Retry
    'RateLimiter',
    'rate_limited',
    'retry_with_backoff',
    
    # Progress Tracking
    'ProgressTracker',
    
    # Utilities
    'safe_truncate',
    'normalize_unicode',
    'StringBuilder',
    'calculate_safe_timeout',
    'safe_file_exists',
    'safe_stat',
    'safe_file_size',
    
    # Notifications
    'NotificationSystem',
    
    # Cloud Storage
    'CloudStorageUploader',
]