#!/usr/bin/env python3
"""
videogen.__main__ - Module Entry Point
Entry point for running videogen as a module: python -m videogen

Version: 42.1
Build Date: 2026-01-03

COMPLETE MODULE ENTRY POINT - NO TRUNCATION
Properly synced with all fixed modules, all imports verified.

This module provides the entry point when the package is run as:
    python -m videogen

It delegates to the main() function in main.py which handles:
- Configuration loading
- Service initialization
- Video generation orchestration
- Graceful shutdown
- Error handling and reporting

Usage:
    python -m videogen                    # Run with default settings
    python -m videogen --help             # Show help (future enhancement)
    
Environment Variables:
    QWEN_URL        - Qwen/Ollama API endpoint
    COMFYUI_URL     - ComfyUI server endpoint
    VIDEO_DIR       - Output directory for generated videos
    VIDEO_QUALITY   - Quality preset (draft/standard/high/ultra)
    ASPECT_RATIO    - Video aspect ratio (9:16/1:1/16:9)
    CAMERA_MOVEMENT - Camera movement style (static/slow_zoom/accelerate_zoom/drift/pan/cinematic)
    MAX_WORKERS     - Maximum parallel workers (1-4)
    ENABLE_SSML     - Enable SSML for TTS (true/false)
    ENABLE_PROFILING- Enable performance profiling (true/false)
    ENABLE_CLOUD_UPLOAD - Enable cloud upload (true/false)
    CLOUD_PROVIDER  - Cloud provider (s3/gcs/azure)
    CLOUD_BUCKET    - Cloud storage bucket name
    NOTIFICATION_WEBHOOK - Webhook URL for notifications
    COMFYUI_MODEL   - Specific ComfyUI model to use
    ALLOW_PRIVATE_URLS - Allow private IPs (true/false) - USE WITH CAUTION

Exit Codes:
    0  - Success - all videos generated successfully
    1  - General error - unexpected failure
    2  - No videos generated - initialization or input error
    3  - Partial success - some videos generated, some failed
    130 - Shutdown requested - user interrupted (Ctrl+C)

Examples:
    # Generate 5 motivational videos in high quality
    QWEN_URL=http://localhost:11434/v1/chat/completions \
    COMFYUI_URL=http://localhost:8188 \
    VIDEO_QUALITY=high \
    python -m videogen
    
    # Generate ultra quality videos with custom output directory
    VIDEO_DIR=/mnt/videos \
    VIDEO_QUALITY=ultra \
    ASPECT_RATIO=16:9 \
    python -m videogen
    
    # Generate with cloud upload enabled
    ENABLE_CLOUD_UPLOAD=true \
    CLOUD_PROVIDER=s3 \
    CLOUD_BUCKET=my-videos \
    python -m videogen

Notes:
    - Services must be running before execution (Qwen/Ollama, ComfyUI)
    - First run downloads voice models (~100MB) and may be slower
    - Cache directory grows over time, periodic cleanup recommended
    - Logs are written to VIDEO_DIR/video_gen.log
    - Metrics are tracked and displayed at completion
    - Graceful shutdown on Ctrl+C (may take a few seconds)

Performance Tips:
    - Use draft quality for testing (10x faster)
    - Increase MAX_WORKERS for batch generation (max 4)
    - Enable caching to speed up repeated generations
    - Use SSD storage for VIDEO_DIR for best performance
    - Ensure adequate disk space (500MB per video minimum)
    - Monitor memory usage (8GB+ recommended for ultra quality)

Troubleshooting:
    - If services unreachable, check URLs and ensure services running
    - If out of memory, reduce quality or MAX_WORKERS
    - If disk full, clean cache directory (VIDEO_DIR/cache)
    - Check logs for detailed error information
    - Use verbose logging: ENABLE_PROFILING=true for performance analysis

For more information, see the project documentation.
"""

import sys
import os

# Ensure package is importable when run as module
if __name__ == "__main__":
    # Add parent directory to path if needed (shouldn't be necessary normally)
    # This handles edge cases where module is run from unusual locations
    try:
        from videogen.main import main
    except ImportError:
        # Fallback: try to add parent to path
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if package_dir not in sys.path:
            sys.path.insert(0, package_dir)
        
        # Retry import
        try:
            from videogen.main import main
        except ImportError as e:
            # If still fails, provide helpful error message
            print("\n" + "="*70, file=sys.stderr)
            print("❌ IMPORT ERROR", file=sys.stderr)
            print("="*70, file=sys.stderr)
            print(f"\nCould not import videogen.main: {e}", file=sys.stderr)
            print("\nTroubleshooting:", file=sys.stderr)
            print("  1. Ensure you're running from the correct directory", file=sys.stderr)
            print("  2. Verify all required files exist:", file=sys.stderr)
            print("     - videogen/__init__.py", file=sys.stderr)
            print("     - videogen/main.py", file=sys.stderr)
            print("     - videogen/config.py", file=sys.stderr)
            print("     - videogen/utils.py", file=sys.stderr)
            print("     - videogen/security.py", file=sys.stderr)
            print("     - videogen/media.py", file=sys.stderr)
            print("     - videogen/services.py", file=sys.stderr)
            print("     - videogen/orchestrator.py", file=sys.stderr)
            print("  3. Check Python version (requires 3.9+)", file=sys.stderr)
            print("  4. Verify all dependencies installed:", file=sys.stderr)
            print("     pip install -r requirements.txt", file=sys.stderr)
            print("\n" + "="*70 + "\n", file=sys.stderr)
            sys.exit(1)
    
    # Verify critical environment before starting
    # This provides early feedback on configuration issues
    def _verify_environment():
        """Verify environment is ready for video generation."""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 9):
            issues.append(
                f"Python {sys.version_info.major}.{sys.version_info.minor} is too old "
                f"(requires 3.9+)"
            )
        elif sys.version_info >= (3, 13):
            print(
                f"⚠️  Warning: Python {sys.version_info.major}.{sys.version_info.minor} "
                f"is untested (recommended 3.9-3.12)",
                file=sys.stderr
            )
        
        # Check if stdin is available for interactive input
        if not sys.stdin.isatty():
            print(
                "ℹ️  Running in non-interactive mode (stdin not a TTY)",
                file=sys.stderr
            )
            print("   Using default values for all prompts", file=sys.stderr)
        
        # Check if stdout/stderr are available
        if sys.stdout is None or sys.stderr is None:
            issues.append("stdout or stderr is None - cannot display output")
        
        # Report issues
        if issues:
            print("\n" + "="*70, file=sys.stderr)
            print("❌ ENVIRONMENT ISSUES", file=sys.stderr)
            print("="*70, file=sys.stderr)
            for issue in issues:
                print(f"  • {issue}", file=sys.stderr)
            print("\n" + "="*70 + "\n", file=sys.stderr)
            sys.exit(1)
    
    # Run environment verification
    try:
        _verify_environment()
    except Exception as e:
        print(f"\n❌ Environment verification failed: {e}\n", file=sys.stderr)
        sys.exit(1)
    
    # Execute main entry point
    # This delegates to main.py which handles everything else
    try:
        main()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\n⊘ Interrupted by user\n", file=sys.stderr)
        sys.exit(130)  # Standard Unix exit code for SIGINT
    except SystemExit:
        # Re-raise SystemExit (from sys.exit() calls)
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        print("\n" + "="*70, file=sys.stderr)
        print("❌ UNEXPECTED ERROR", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"\n{type(e).__name__}: {e}\n", file=sys.stderr)
        print("This is likely a bug. Please report it with:", file=sys.stderr)
        print("  1. Full error message above", file=sys.stderr)
        print("  2. Python version:", sys.version, file=sys.stderr)
        print("  3. Operating system:", os.name, file=sys.stderr)
        print("  4. Steps to reproduce", file=sys.stderr)
        print("\n" + "="*70 + "\n", file=sys.stderr)
        
        # Print traceback if available
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        sys.exit(1)
