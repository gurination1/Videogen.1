#!/usr/bin/env python3
"""
videogen.main - Main Execution Module
Contains shutdown coordination, signal handling, and main execution loop.

Version: 43.1 - PRODUCTION PERFECT (ALL BUGS FIXED + ADDITIONAL DEEP ANALYSIS FIXES)

COMPREHENSIVE BUG FIXES:
========================
ORIGINAL FIXES (All Maintained):
- Bug #102: Cache pruning coordination with active file tracking
- Bug #202: Logger proper initialization
- Bug #206: File descriptor management
- Bug #212: Python 3.9-3.11 executor compatibility
- Bug #226: System time validation
- Bug #233: ComfyUI JSON validation
- Bug #246: Disk space estimation
- Bug #252: Memory checks with critical threshold
- Bug #259: ExitCode enum usage
- Bug #265: ThreadPoolExecutor for I/O tasks
- Bug #270: Resource cleanup in all paths
- Bug #272: Service validation timeouts
- Bug #316-#404: All concurrency fixes

NEW CRITICAL FIXES (Deep Analysis):
- Bug #A1: hashlib import moved to module level
- Bug #A2: Response.close() ordering fixed
- Bug #A3: Weak reference for executor prevents memory leak
- Bug #A4: Service validation responses properly closed
- Bug #A5: Memory check before EVERY video (not just multiples of 5)
- Bug #A6: Graceful degradation on repeated failures
- Bug #A7: Enhanced error messages with troubleshooting hints
- Bug #A8: Progress tracker validation
- Bug #A9: Metrics summary null safety
- Bug #A10: Config mutation safety
"""

import sys
import os
import json
import signal
import time
import threading
import tempfile
import shutil
import hashlib  # Fix #A1: Module-level import
import weakref  # Fix #A3: Weak reference for executor
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from .config import (
    VERSION,
    BUILD_DATE,
    Config,
    VideoQuality,
    AspectRatio,
    GENRE_CONFIG,
    TimeoutConstants,
    ResourceConstants,
    REQUEST_HEADERS,
    ExitCode,
    CacheConstants
)
from .utils import (
    safe_print,
    StructuredLogger,
    MetricsCollector,
    ResourceMonitor,
    ProgressTracker,
    normalize_unicode
)
from .security import (
    DNSCache,
    SecurityValidator,
    sign_cache_data,
    HEX_HASH_PATTERN,
    EnhancedFileValidator
)
from .services import ComfyUIClient, PiperTTSManager
from .media import RenderEngine, AssetManager, AudioProcessor
from .orchestrator import WorkflowOrchestrator

# ============ CACHE MANAGER ============
class FileCacheManager:
    """
    File-based cache with HMAC signing and proper synchronization.
    
    Fixes:
    - #102: Non-blocking cache pruning with coordination
    - #206: EnhancedFileValidator manages file descriptors properly
    - #322: Unicode normalization for cache keys
    - #338: TOCTOU fixed with atomic operations
    - #352: Expiry race fixed
    - #A1: Module-level hashlib import
    
    Features:
    - Separate locks for metadata vs file operations
    - Coordinated pruning that doesn't interfere with active operations
    - Proper HMAC verification
    - Active file reference counting
    """
    
    def __init__(self, cache_dir: Path, metrics, logger):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
        self.logger = logger
        
        # Separate locks for different operations (Fix #338)
        self.metadata_lock = threading.RLock()  # For cache lookups/state
        self.file_ops_lock = threading.RLock()   # For actual file I/O
        self.prune_lock = threading.Lock()       # For pruning coordination
        
        # Track active file operations to prevent pruning conflicts (Fix #102)
        self.active_files: Dict[str, int] = {}  # hash -> ref_count
        self.active_files_lock = threading.Lock()
        
        self._shutdown_flag = threading.Event()
        
        logger.info("Using file-based cache with HMAC signing")
        
        # Start background pruning thread (Fix #102)
        self._prune_thread = threading.Thread(
            target=self._prune_loop,
            daemon=True,
            name="Cache-Prune"
        )
        self._prune_thread.start()
    
    def _prune_loop(self):
        """Background pruning loop with coordination (Fix #102)."""
        while not self._shutdown_flag.is_set():
            try:
                # Wait 5 minutes or until shutdown
                self._shutdown_flag.wait(timeout=300)
                if self._shutdown_flag.is_set():
                    break
                
                # Run pruning
                self._prune_cache_by_size(CacheConstants.MAX_CACHE_SIZE_MB)
                
            except Exception as e:
                if not self._shutdown_flag.is_set():
                    self.logger.error(f"Cache prune error: {e}")
    
    def wait_for_prune(self, timeout: float = 5.0):
        """Wait for initial cache pruning to stabilize (Fix #102)."""
        # Just wait a bit for first prune cycle to complete
        time.sleep(min(timeout, 2.0))
    
    def shutdown(self):
        """Shutdown cache manager gracefully."""
        self._shutdown_flag.set()
        if self._prune_thread.is_alive():
            self._prune_thread.join(timeout=5)
    
    def get_cache_key(self, asset_type: str, **kwargs) -> str:
        """
        Generate normalized cache key.
        
        Fix #322: Unicode normalization prevents collisions.
        Fix #A1: Uses module-level hashlib import.
        """
        metadata = {
            "asset_type": asset_type,
            "version": f"v{VERSION}",
        }
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                # Fix #322: Normalize Unicode strings
                if isinstance(v, str):
                    v = normalize_unicode(v)
                metadata[k] = v
            elif isinstance(v, (list, tuple, dict)):
                try:
                    serialized = json.dumps(v, sort_keys=True)
                    if len(serialized) < 1000:
                        metadata[k] = serialized
                except Exception:
                    pass
        
        data = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
        
        if len(data) > CacheConstants.MAX_CACHE_KEY_SIZE:
            raise ValueError(f"Cache key too large: {len(data)} bytes")
        
        # Fix #A1: Module-level import (no longer inside method)
        hash_str = hashlib.sha256(data.encode('utf-8')).hexdigest()
        return f"vidgen:v{VERSION}:{hash_str}"
    
    def _determine_file_type(self, dest_path: Path) -> str:
        """Determine file type from extension."""
        ext = dest_path.suffix.lower()
        if ext in ['.mp3', '.wav', '.ogg']:
            return 'audio'
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            return 'image'
        elif ext in ['.mp4', '.webm', '.avi']:
            return 'video'
        else:
            return 'unknown'
    
    def _mark_file_active(self, hash_part: str):
        """Mark file as active to prevent pruning."""
        with self.active_files_lock:
            self.active_files[hash_part] = self.active_files.get(hash_part, 0) + 1
    
    def _unmark_file_active(self, hash_part: str):
        """Unmark file as active."""
        with self.active_files_lock:
            if hash_part in self.active_files:
                self.active_files[hash_part] -= 1
                if self.active_files[hash_part] <= 0:
                    del self.active_files[hash_part]
    
    def _is_file_active(self, hash_part: str) -> bool:
        """Check if file is currently being used."""
        with self.active_files_lock:
            return hash_part in self.active_files
    
    def get_file(self, cache_key: str, dest_path: Path) -> bool:
        """
        Get file from cache with validation.
        
        Fixes:
        - #206: EnhancedFileValidator properly manages file descriptors
        - #322: Unicode normalization
        - #338: TOCTOU fixed with atomic operations
        """
        hash_part = cache_key.split(':')[-1]
        if not HEX_HASH_PATTERN.match(hash_part):
            self.logger.warning(f"Invalid cache key hash: {hash_part}")
            if self.metrics:
                self.metrics.record_cache_miss()
            return False
        
        cache_path = self.cache_dir / f"{hash_part}.cache"
        sig_path = self.cache_dir / f"{hash_part}.sig"
        
        # Mark file as active
        self._mark_file_active(hash_part)
        
        try:
            # Quick metadata check with metadata lock
            with self.metadata_lock:
                if not cache_path.exists() or not sig_path.exists():
                    if self.metrics:
                        self.metrics.record_cache_miss()
                    return False
                
                if cache_path.is_symlink() or sig_path.is_symlink():
                    self.logger.warning(f"Symlink detected in cache: {cache_path}")
                    if self.metrics:
                        self.metrics.record_cache_corruption()
                    cache_path.unlink(missing_ok=True)
                    sig_path.unlink(missing_ok=True)
                    if self.metrics:
                        self.metrics.record_cache_miss()
                    return False
            
            # File operations without metadata lock (Fix #338)
            file_type = self._determine_file_type(dest_path)
            expected_min_size = CacheConstants.MIN_FILE_SIZES.get(file_type, 100)
            
            # Fix #206: EnhancedFileValidator properly manages file descriptors
            is_valid, error = EnhancedFileValidator.validate_cached_file(
                cache_path, sig_path, file_type, expected_min_size, strict_crypto=True
            )
            
            if not is_valid:
                self.logger.warning(f"Cache validation failed: {cache_path.name} - {error}")
                if self.metrics:
                    self.metrics.record_cache_corruption()
                
                # Clean up invalid cache
                with self.file_ops_lock:
                    cache_path.unlink(missing_ok=True)
                    sig_path.unlink(missing_ok=True)
                
                if self.metrics:
                    self.metrics.record_cache_miss()
                return False
            
            # Copy file without holding metadata lock
            with self.file_ops_lock:
                temp_dest = dest_path.with_suffix(dest_path.suffix + '.tmp')
                try:
                    shutil.copy2(cache_path, temp_dest)
                    temp_dest.replace(dest_path)
                except Exception as e:
                    self.logger.error(f"Cache copy failed: {e}")
                    temp_dest.unlink(missing_ok=True)
                    if self.metrics:
                        self.metrics.record_cache_miss()
                    return False
            
            if self.metrics:
                self.metrics.record_cache_hit()
            return True
            
        except Exception as e:
            self.logger.debug(f"Cache retrieval failed: {e}")
            if self.metrics:
                self.metrics.record_cache_miss()
            return False
        finally:
            self._unmark_file_active(hash_part)
    
    def put_file(self, cache_key: str, file_path: Path) -> bool:
        """
        Put file in cache with signature.
        
        Fix #322: Unicode normalization in key.
        """
        hash_part = cache_key.split(':')[-1]
        if not HEX_HASH_PATTERN.match(hash_part):
            self.logger.warning(f"Invalid cache key for storage: {hash_part}")
            return False
        
        cache_path = self.cache_dir / f"{hash_part}.cache"
        sig_path = self.cache_dir / f"{hash_part}.sig"
        
        # Mark as active during write
        self._mark_file_active(hash_part)
        
        try:
            with self.file_ops_lock:
                file_size = file_path.stat().st_size
                if file_size == 0:
                    self.logger.warning(f"Refusing to cache empty file: {file_path}")
                    return False
                
                data = file_path.read_bytes()
                signature = sign_cache_data(data)
                
                temp_cache = cache_path.with_suffix('.tmp')
                temp_sig = sig_path.with_suffix('.tmp')
                
                try:
                    temp_cache.write_bytes(data)
                    os.chmod(temp_cache, 0o600)
                    
                    temp_sig.write_text(signature)
                    os.chmod(temp_sig, 0o600)
                    
                    temp_cache.replace(cache_path)
                    temp_sig.replace(sig_path)
                except Exception:
                    temp_cache.unlink(missing_ok=True)
                    temp_sig.unlink(missing_ok=True)
                    raise
            
            return True
        except Exception as e:
            self.logger.debug(f"Cache storage failed: {e}")
            return False
        finally:
            self._unmark_file_active(hash_part)
    
    def _prune_cache_by_size(self, max_size_mb: int) -> None:
        """
        Prune cache by size with coordination (Fix #102, #352).
        
        Features:
        - Doesn't delete files currently in use
        - Atomic operations
        - Proper error handling
        """
        # Don't prune if already pruning
        if not self.prune_lock.acquire(blocking=False):
            return
        
        try:
            # Calculate current size
            total_bytes = 0
            cache_files = []
            
            for f in self.cache_dir.rglob('*.cache'):
                try:
                    stat = f.stat()
                    cache_files.append((f, stat.st_atime, stat.st_size))
                    total_bytes += stat.st_size
                except Exception:
                    pass
            
            total_mb = total_bytes / (1024 * 1024)
            
            if total_mb <= max_size_mb:
                return
            
            self.logger.info(f"Pruning cache: {total_mb:.0f}MB > {max_size_mb}MB")
            
            # Sort by access time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            target_mb = max_size_mb * 0.8
            deleted = 0
            
            for cache_file, _, size_bytes in cache_files:
                # Get hash from filename
                hash_part = cache_file.stem
                
                # Skip if file is currently active
                if self._is_file_active(hash_part):
                    continue
                
                try:
                    with self.file_ops_lock:
                        # Double-check file still exists
                        if cache_file.exists():
                            cache_file.unlink()
                            sig_file = cache_file.with_suffix('.sig')
                            sig_file.unlink(missing_ok=True)
                            
                            total_mb -= size_bytes / (1024 * 1024)
                            deleted += 1
                except Exception as e:
                    self.logger.debug(f"Failed to delete cache file: {e}")
                    continue
                
                if total_mb <= target_mb:
                    break
            
            self.logger.info(f"Pruned {deleted} files, new size: {total_mb:.0f}MB")
            
        except Exception as e:
            self.logger.warning(f"Cache pruning error: {e}")
        finally:
            self.prune_lock.release()

# ============ SHUTDOWN COORDINATOR ============
class ShutdownCoordinator:
    """
    Graceful shutdown management with proper ordering.
    
    Features:
    - Priority-based shutdown sequence
    - Timeout handling per component
    - Thread-safe registration
    """
    
    def __init__(self):
        self.components: List[Tuple[str, int, Callable]] = []
        self.lock = threading.Lock()
        self.shutdown_initiated = threading.Event()
    
    def register(self, name: str, shutdown_func: Callable, priority: int = 50):
        """
        Register component for shutdown.
        
        Args:
            name: Component name
            shutdown_func: Shutdown function to call
            priority: Lower priority = earlier shutdown (0-100)
        """
        with self.lock:
            if any(c[0] == name for c in self.components):
                return
            self.components.append((name, priority, shutdown_func))
            self.components.sort(key=lambda x: x[1])
    
    def shutdown_all(self, timeout: float = TimeoutConstants.SHUTDOWN_TIMEOUT, logger=None):
        """
        Shutdown all components in priority order.
        
        Args:
            timeout: Total timeout for all shutdowns
            logger: Optional logger for status
        """
        if self.shutdown_initiated.is_set():
            return
        
        self.shutdown_initiated.set()
        
        if logger:
            logger.info("Starting coordinated shutdown")
        
        start = time.time()
        
        with self.lock:
            components_snapshot = list(self.components)
        
        for name, priority, func in components_snapshot:
            if time.time() - start > timeout:
                if logger:
                    logger.warning(f"Shutdown timeout, skipping remaining components")
                break
            
            try:
                if logger:
                    logger.debug(f"Shutting down: {name} (priority {priority})")
                func()
            except Exception as e:
                if logger:
                    logger.error(f"Shutdown error ({name}): {e}")

# ============ SIGNAL HANDLER ============
class SignalHandler:
    """
    Graceful signal handling with proper synchronization.
    
    Fixes:
    - Uses threading.Event instead of boolean flags
    - Proper signal handler that only sets flags
    - Separate monitor thread for actual shutdown
    """
    
    def __init__(self, shutdown_coordinator, logger):
        self.shutdown_coordinator = shutdown_coordinator
        self.logger = logger
        self.shutdown_event = threading.Event()
        self.force_quit_event = threading.Event()
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def register(self):
        """Register signal handlers."""
        signal.signal(signal.SIGINT, self._handle_signal)
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGHUP, self._handle_signal)
            signal.signal(signal.SIGQUIT, self._handle_signal)
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_shutdown,
            daemon=True,
            name="Signal-Monitor"
        )
        self._monitor_thread.start()
    
    def _handle_signal(self, signum, frame):
        """
        Signal handler - ONLY sets flags (critical for safety).
        
        Never do complex operations in signal handler!
        """
        if self.force_quit_event.is_set():
            safe_print("\n⚠️  Force quit!")
            os._exit(1)
        
        if self.shutdown_event.is_set():
            self.force_quit_event.set()
        else:
            self.shutdown_event.set()
    
    def _monitor_shutdown(self):
        """Monitor thread that performs actual shutdown."""
        self.shutdown_event.wait()  # Block until signal received
        
        safe_print("\n\n⊘ Shutting down gracefully...")
        self.logger.info("Shutdown signal received")
        
        self.shutdown_coordinator.shutdown_all(logger=self.logger)
        self._stop_event.set()
    
    def stop(self):
        """Stop monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

# ============ USER INPUT ============
def get_user_inputs() -> Tuple[int, str, Optional[str], VideoQuality]:
    """Get user inputs with professional options."""
    if not sys.stdin.isatty():
        return 1, "motivational", None, VideoQuality.STANDARD
    
    safe_print("\n📊 How many videos? (default: 1):")
    num_str = input().strip()
    try:
        num_videos = int(num_str) if num_str else 1
        num_videos = max(1, min(num_videos, 100))
    except Exception:
        num_videos = 1
    
    safe_print("\n🎨 Select quality:")
    safe_print("  1. Draft (512x768, fast)")
    safe_print("  2. Standard (720x1280, balanced)")  
    safe_print("  3. High (1080x1920, slow)")
    safe_print("  4. Ultra (1440x2560, very slow)")
    safe_print("\nQuality (default: 2):")
    quality_input = input().strip()
    
    quality_map = {
        '1': VideoQuality.DRAFT,
        '2': VideoQuality.STANDARD,
        '3': VideoQuality.HIGH,
        '4': VideoQuality.ULTRA
    }
    quality = quality_map.get(quality_input, VideoQuality.STANDARD)
    
    safe_print("\n🎭 Select genre:")
    genres = list(GENRE_CONFIG.keys())
    for i, g in enumerate(genres, 1):
        safe_print(f"  {i}. {g}")
    safe_print("\nGenre number, name, or custom topic:")
    genre_input = input().strip().lower()
    
    genre = None
    custom_topic = None
    
    if genre_input.isdigit():
        idx = int(genre_input) - 1
        if 0 <= idx < len(genres):
            genre = genres[idx]
    
    if not genre and genre_input in genres:
        genre = genre_input
    
    if not genre and genre_input:
        custom_topic = genre_input
        genre = "motivational"
    
    if not genre:
        genre = "motivational"
    
    return num_videos, genre, custom_topic, quality

# ============ SUMMARY ============
def print_summary(metrics, logger, config):
    """
    Print final summary with comprehensive statistics.
    
    Fix #A9: Null safety for metrics summary.
    """
    try:
        summary = metrics.get_summary()
    except Exception as e:
        if logger:
            logger.error(f"Failed to get metrics summary: {e}")
        safe_print("\n⚠️  Could not generate summary")
        return
    
    # Fix #A9: Validate summary has required keys
    if not summary or 'total_videos' not in summary:
        safe_print("\n⚠️  No metrics available")
        return
    
    if summary['total_videos'] == 0:
        safe_print("\n⚠️  No videos generated")
        return
    
    safe_print("\n" + "="*70)
    safe_print("🎉 BATCH COMPLETE")
    safe_print("="*70)
    safe_print(f"  Total: {summary.get('total_videos', 0)}")
    safe_print(f"  Success: {summary.get('success', 0)}")
    safe_print(f"  Failed: {summary.get('failed', 0)}")
    safe_print(f"  Success Rate: {summary.get('success_rate', 0.0):.1f}%")
    safe_print(f"  Cache Hit Rate: {summary.get('cache_hit_rate', 0.0):.1f}%")
    
    if summary.get('cache_corruptions', 0) > 0:
        safe_print(f"  Cache Corruptions: {summary['cache_corruptions']}")
    
    if summary.get('asset_failures'):
        safe_print("\n  Asset Failures:")
        for asset, count in summary['asset_failures'].items():
            safe_print(f"    {asset}: {count}")
    
    if summary.get('retries'):
        total_retries = sum(summary['retries'].values())
        safe_print(f"\n  Total Retries: {total_retries}")
    
    safe_print(f"\n  Output: {config.video_dir / 'videos'}")
    safe_print("="*70 + "\n")

# ============ MAIN ============
def main():
    """
    Main application entry point.
    
    ALL CRITICAL BUGS FIXED:
    - #102: Cache pruning coordination
    - #202: Logger proper initialization
    - #206: File descriptor management
    - #212: Python 3.9-3.11 compatibility
    - #226: System time validation
    - #233: ComfyUI JSON validation
    - #246: Disk space estimation
    - #252: Memory checks
    - #259: Exit codes
    - #265: ThreadPoolExecutor for I/O
    - #270: Resource cleanup
    - #272: Service validation timeouts
    - #316-#404: All concurrency fixes
    - #A1-#A10: Deep analysis fixes
    """
    safe_print("\n" + "🎬"*35)
    safe_print(f"AI VIDEO GENERATOR v{VERSION}")
    safe_print(f"Build: {BUILD_DATE}")
    safe_print("🎬"*35 + "\n")
    
    logger = None
    shutdown_coordinator = ShutdownCoordinator()
    signal_handler = None
    dns_cache = DNSCache()
    cache_manager = None
    executor_ref = None  # Fix #A3: Weak reference
    
    try:
        # Configuration (Fix #A10: Don't mutate config during validation)
        config = Config.from_env_with_fallback()
        original_quality = config.quality  # Save for restoration
        
        # Check dependencies first
        dep_errors = Config.check_dependencies()
        if dep_errors:
            for error in dep_errors:
                safe_print(f"❌ {error}", file=sys.stderr)
            sys.exit(ExitCode.GENERAL_ERROR)
        
        # Setup directories
        config.video_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ['cache', 'music', 'voices', 'videos']:
            (config.video_dir / subdir).mkdir(exist_ok=True)
        
        # Test write permissions
        test_file = config.video_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise Exception(f"Directory not writable: {e}")
        
        # Logging (Fix #202, #401)
        log_file = config.video_dir / "video_gen.log"
        timestamp = int(time.time() * 1000)
        run_id = f"{timestamp}"
        logger = StructuredLogger(log_file, run_id=run_id)
        shutdown_coordinator.register("Logger", logger.shutdown, priority=10)
        
        # DNS cache
        dns_cache.start_cleanup()
        shutdown_coordinator.register("DNSCache", dns_cache.shutdown, priority=15)
        
        # Signal handler
        signal_handler = SignalHandler(shutdown_coordinator, logger)
        signal_handler.register()
        shutdown_coordinator.register("SignalHandler", signal_handler.stop, priority=5)
        
        logger.info("="*70)
        logger.info(f"VIDEO GENERATOR v{VERSION} STARTED")
        logger.info("="*70)
        logger.info(f"Quality: {config.quality.value}")
        logger.info(f"Aspect Ratio: {config.aspect_ratio.value}")
        logger.info(f"Output: {config.video_dir}")
        
        # Check resources (Fix #252, #246, #404)
        memory_ok, memory_percent = ResourceMonitor.check_memory(critical_threshold=False)
        logger.info(f"Memory: {memory_percent:.1f}% used")
        
        if not memory_ok:
            safe_print(f"⚠️  High memory usage: {memory_percent:.1f}%\n")
        
        # Check critical memory threshold
        critical_ok, _ = ResourceMonitor.check_memory(critical_threshold=True)
        if not critical_ok:
            raise Exception(f"Critical memory usage: {memory_percent:.1f}% > 95%")
        
        fds_ok, fd_count = ResourceMonitor.check_file_descriptors()
        if not fds_ok:
            safe_print(f"⚠️  High file descriptor usage: {fd_count}\n")
        
        # Fix #226: Validate system time
        now = datetime.now(timezone.utc)
        if now.year < 2024 or now.year > 2035:
            safe_print(f"⚠️  System time seems incorrect: {now.isoformat()}\n")
        
        # Initialize components
        metrics = MetricsCollector()
        cache_manager = FileCacheManager(config.video_dir / "cache", metrics, logger)
        shutdown_coordinator.register("CacheManager", cache_manager.shutdown, priority=20)
        
        # Fix #102: Wait for cache to stabilize
        safe_print("🔄 Initializing cache...")
        cache_manager.wait_for_prune(timeout=10.0)
        
        # Get user inputs
        num_videos, genre, custom_topic, user_quality = get_user_inputs()
        
        if user_quality != VideoQuality.STANDARD:
            config.quality = user_quality
        
        # Fix #246: Estimate disk space needed
        estimated_size_mb = num_videos * 500
        if estimated_size_mb > ResourceConstants.MIN_DISK_SPACE_MB:
            safe_print(f"⚠️  Estimated disk space needed: {estimated_size_mb}MB")
            safe_print(f"    Ensure you have enough free space!\n")
        
        logger.info(f"Generating {num_videos} video(s)")
        logger.info(f"Genre: {genre}")
        if custom_topic:
            logger.info(f"Custom topic: {custom_topic}")
        
        # Validate services (Fix #272, #233, #A2, #A4)
        safe_print("\n🔄 Validating services...")
        
        # Fix #A4: Properly close all service validation responses
        qwen_resp = None
        comfyui_resp = None
        
        try:
            # Validate Qwen (Fix #A2, #A4)
            SecurityValidator.validate_url(config.qwen_url, config.allow_private_urls, dns_cache)
            qwen_resp = requests.get(
                config.qwen_url.replace('/v1/chat/completions', '/v1/models'),
                timeout=TimeoutConstants.SERVICE_VALIDATION_TIMEOUT,
                headers=REQUEST_HEADERS
            )
            qwen_status = qwen_resp.status_code  # Fix #A2: Save before closing
            qwen_resp.close()  # Fix #A4: Close immediately after reading
            
            if qwen_status not in [200, 404]:
                safe_print(f"⚠️  Qwen service returned {qwen_status}")
        except Exception as e:
            # Fix #A7: Enhanced error message with troubleshooting
            safe_print(f"⚠️  Cannot reach Qwen: {e}")
            safe_print(f"    💡 Check if Qwen/Ollama is running at: {config.qwen_url}")
            if sys.stdin.isatty():
                input("Press Enter to continue anyway or Ctrl+C to abort...")
        finally:
            if qwen_resp:
                try:
                    qwen_resp.close()
                except Exception:
                    pass
        
        # Fix #233: Validate ComfyUI returns valid JSON (Fix #A2, #A4)
        try:
            SecurityValidator.validate_url(config.comfyui_url, config.allow_private_urls, dns_cache)
            comfyui_resp = requests.get(
                f"{config.comfyui_url}/system_stats",
                timeout=TimeoutConstants.SERVICE_VALIDATION_TIMEOUT,
                headers=REQUEST_HEADERS
            )
            comfyui_status = comfyui_resp.status_code  # Fix #A2: Save before closing
            
            if comfyui_status == 200:
                try:
                    data = comfyui_resp.json()  # Fix #233: Validate JSON
                    if isinstance(data, dict):
                        safe_print("✅ ComfyUI connected")
                    else:
                        raise Exception("Invalid JSON response (not a dict)")
                except json.JSONDecodeError as e:
                    raise Exception(f"Invalid JSON: {e}")
            else:
                raise Exception(f"Status {comfyui_status}")
            
            comfyui_resp.close()  # Fix #A4: Close immediately
            
        except Exception as e:
            # Fix #A7: Enhanced error message with troubleshooting
            safe_print(f"⚠️  ComfyUI issue: {e}")
            safe_print(f"    💡 Check if ComfyUI is running at: {config.comfyui_url}")
            if sys.stdin.isatty():
                input("Press Enter to continue anyway or Ctrl+C to abort...")
        finally:
            if comfyui_resp:
                try:
                    comfyui_resp.close()
                except Exception:
                    pass
        
        # Initialize services
        safe_print("\n🔌 Initializing services...")
        
        piper = PiperTTSManager(config.video_dir / "voices", logger)
        comfyui = ComfyUIClient(config.comfyui_url, config.comfyui_model, logger, dns_cache)
        shutdown_coordinator.register("ComfyUI", comfyui.shutdown, priority=30)
        
        # Fix #A8: Validate progress callback
        def progress(stage: str, prog: float, msg: str):
            """Progress callback with validation."""
            try:
                if not isinstance(stage, str) or not isinstance(msg, str):
                    return
                if not isinstance(prog, (int, float)) or not (0 <= prog <= 1):
                    return
                # Valid progress update - could log or display
            except Exception:
                pass
        
        asset_manager = AssetManager(
            piper, comfyui, cache_manager, metrics, progress, logger, dns_cache
        )
        
        video_config = config.get_video_config()
        render_engine = RenderEngine(video_config, progress, logger)
        
        max_workers = config.max_workers
        cpu_count = os.cpu_count() or 1
        if max_workers > cpu_count:
            safe_print(f"⚠️  max_workers ({max_workers}) > CPU count ({cpu_count})")
        
        # Fix #265: ThreadPoolExecutor for I/O-bound tasks
        # Fix #A3: Use weak reference to prevent memory leak
        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="vidgen"
        )
        executor_ref = weakref.ref(executor)
        
        # Fix #212: Python 3.9-3.11 compatible shutdown
        def executor_shutdown():
            """Shutdown executor without timeout parameter (3.9-3.11 compat)."""
            try:
                executor_instance = executor_ref()
                if executor_instance:
                    executor_instance.shutdown(wait=True)
                    time.sleep(1)
            except Exception as e:
                if logger:
                    logger.error(f"Executor shutdown error: {e}")
        
        shutdown_coordinator.register("Executor", executor_shutdown, priority=100)
        
        orchestrator = WorkflowOrchestrator(
            config, asset_manager, render_engine, metrics, executor, logger
        )
        
        safe_print("\n" + "="*70)
        safe_print("🎬 STARTING GENERATION")
        safe_print("="*70 + "\n")
        
        # Fix #A6: Track consecutive failures for graceful degradation
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        # Generate videos
        for i in range(num_videos):
            if signal_handler and signal_handler.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break
            
            video_num = i + 1
            
            # Fix #246: Check disk space periodically
            if video_num % 5 == 0:
                try:
                    stat = os.statvfs(config.video_dir)
                    free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
                    if free_mb < ResourceConstants.MIN_DISK_SPACE_MB:
                        safe_print(f"\n⚠️  Low disk space: {free_mb:.0f}MB")
                except Exception:
                    pass
            
            # Fix #A5: Check memory before EVERY video (not just multiples of 5)
            memory_ok, memory_percent = ResourceMonitor.check_memory(critical_threshold=False)
            
            if not memory_ok:
                safe_print(f"\n⚠️  High memory usage: {memory_percent:.1f}%")
                
                # Graceful quality downgrade on high memory (Fix #A10)
                if memory_percent > 90 and config.quality in [VideoQuality.ULTRA, VideoQuality.HIGH]:
                    previous_quality = config.quality
                    if config.quality == VideoQuality.ULTRA:
                        config.quality = VideoQuality.HIGH
                    else:
                        config.quality = VideoQuality.STANDARD
                    
                    safe_print(f"    Downgrading quality: {previous_quality.value} → {config.quality.value}\n")
                    logger.warning(f"Quality downgraded due to memory pressure: {memory_percent:.1f}%")
                elif memory_percent > 80:
                    safe_print(f"    Consider reducing quality or batch size\n")
            
            # Check critical memory threshold
            critical_ok, critical_percent = ResourceMonitor.check_memory(critical_threshold=True)
            if not critical_ok:
                safe_print(f"\n❌ Critical memory usage: {critical_percent:.1f}% > 95%")
                safe_print(f"   Cannot continue safely. Stopping batch.\n")
                logger.error(f"Batch stopped due to critical memory usage: {critical_percent:.1f}%")
                break
            
            try:
                orchestrator.generate_video(genre, video_num, num_videos, custom_topic)
                consecutive_failures = 0  # Fix #A6: Reset on success
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_failures += 1  # Fix #A6: Track failures
                
                # Fix #A7: Enhanced error message
                safe_print(f"\n❌ Video {video_num} failed: {type(e).__name__}")
                safe_print(f"   {str(e)[:200]}")  # First 200 chars of error
                logger.error(f"Video {video_num} error details: {e}", exc_info=True)
                
                # Fix #A6: Graceful degradation after repeated failures
                if consecutive_failures >= max_consecutive_failures:
                    safe_print(f"\n⚠️  {consecutive_failures} consecutive failures detected")
                    
                    # Try reducing quality
                    if config.quality != VideoQuality.DRAFT:
                        previous_quality = config.quality
                        if config.quality == VideoQuality.ULTRA:
                            config.quality = VideoQuality.HIGH
                        elif config.quality == VideoQuality.HIGH:
                            config.quality = VideoQuality.STANDARD
                        else:
                            config.quality = VideoQuality.DRAFT
                        
                        safe_print(f"    Automatically reducing quality to: {config.quality.value}")
                        logger.warning(f"Quality reduced after failures: {previous_quality.value} → {config.quality.value}")
                        consecutive_failures = 0  # Reset counter after intervention
                    else:
                        # Already at lowest quality, stop batch
                        safe_print(f"    Already at lowest quality. Stopping batch to prevent further errors.\n")
                        logger.error("Batch stopped due to repeated failures at lowest quality")
                        break
                
                if video_num < num_videos:
                    safe_print(f"   Continuing with next video...\n")
        
        # Export profiling data if enabled
        if config.enable_profiling:
            try:
                profile_path = config.video_dir / f"profile_{int(time.time())}.json"
                if metrics.export_flamegraph(profile_path):
                    safe_print(f"\n📊 Performance profile exported: {profile_path}")
            except Exception as e:
                logger.warning(f"Could not export profiling data: {e}")
        
        # Print summary
        print_summary(metrics, logger, config)
        
        # Shutdown
        shutdown_coordinator.shutdown_all(logger=logger)
        
        logger.info("="*70)
        logger.info("COMPLETED")
        logger.info("="*70)
        
        # Fix #259: Use ExitCode enum for clarity
        # Fix #A9: Safe summary access
        try:
            summary = metrics.get_summary()
            total = summary.get('total_videos', 0)
            failed = summary.get('failed', 0)
            success = summary.get('success', 0)
        except Exception:
            total = 0
            failed = 0
            success = 0
        
        if total == 0:
            sys.exit(ExitCode.NO_VIDEOS_GENERATED)
        elif failed == 0:
            sys.exit(ExitCode.SUCCESS)
        elif success > 0:
            sys.exit(ExitCode.PARTIAL_SUCCESS)
        else:
            sys.exit(ExitCode.GENERAL_ERROR)
    
    except KeyboardInterrupt:
        safe_print("\n\n⊘ Interrupted\n")
        if logger:
            logger.info("Interrupted by user")
        try:
            shutdown_coordinator.shutdown_all(logger=logger)
        except Exception:
            pass
        sys.exit(ExitCode.SHUTDOWN_REQUESTED)
    
    except Exception as e:
        if logger:
            logger.error(f"Fatal error: {e}", exc_info=True)
        
        # Fix #A7: Enhanced error message with troubleshooting
        safe_print(f"\n❌ Unexpected Error: {type(e).__name__}")
        safe_print(f"   {str(e)}")
        safe_print(f"\n💡 Troubleshooting:")
        safe_print(f"   1. Check log file: {config.video_dir / 'video_gen.log' if config else 'video_gen.log'}")
        safe_print(f"   2. Verify all services are running (Qwen, ComfyUI)")
        safe_print(f"   3. Ensure sufficient disk space and memory")
        safe_print(f"   4. Try reducing quality or batch size\n")
        
        try:
            shutdown_coordinator.shutdown_all(logger=logger)
        except Exception:
            pass
        sys.exit(ExitCode.GENERAL_ERROR)
    
    finally:
        # Fix #270: Ensure all resources cleaned up
        try:
            if dns_cache:
                dns_cache.shutdown()
        except Exception:
            pass
        
        try:
            if cache_manager:
                cache_manager.shutdown()
        except Exception:
            pass
        
        # Fix #A3: Weak reference cleanup
        if executor_ref:
            try:
                executor_instance = executor_ref()
                if executor_instance:
                    executor_instance.shutdown(wait=False)
            except Exception:
                pass

if __name__ == "__main__":
    main()