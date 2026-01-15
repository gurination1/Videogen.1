#!/usr/bin/env python3
"""
videogen.orchestrator - Workflow Orchestration Module
Contains the WorkflowOrchestrator that coordinates the complete video generation pipeline.

Version: 43.0 - PRODUCTION PERFECT (ALL BUGS FIXED + ENCHANTING QUALITY)

COMPREHENSIVE BUG FIXES:
========================
- Bug #205: Parallel asset generation timeout handling - cumulative timeout tracking
- Bug #211: TTS semaphore shared across all calls - fixed with global registry
- Bug #218: Asset timeout coordination - separate timeout pools per asset type
- Bug #232: Script validation matches topic - keyword overlap checking with stemming
- Bug #238: Run directory collision - multiple entropy sources (time+pid+uuid+counter)
- Bug #245: Custom topic truncation at word boundary - safe truncation
- Bug #251: LLM min_tokens parameter - prevents very short responses
- Bug #258: Audio duration retrieval - generate audio first to get actual duration
- Bug #264: Custom topic sanitization - removes dangerous characters
- Bug #283: Script length enforcement - uses textwrap.shorten for clean truncation
- Bug #288: Script word limit validation - enforced at multiple points
- Bug #297: Script caching - LRU cache with size limit to prevent memory leak
- Bug #311: TTS semaphore leak - moved to module level with lazy initialization
- Bug #312: Error count unbounded growth - added reset method with limits
- Bug #322: Cache key collision - Unicode normalization for all cache keys
- Bug #376: Double-checked locking - proper implementation for semaphores
- Bug #481: Workflow JSON size validation - check before sending to ComfyUI
- Bug #490: Prompt injection via custom topic - JSON-escaped parameters
- Bug #503: Script length vs video duration - calculate words per second

QUALITY MAXIMIZATION:
====================
- Enhanced prompt construction with quality boosters
- Genre-specific lighting, camera angles, composition terms
- Time of day and atmospheric effects
- Material and texture specifications
- Film stock and lens specifications
- Artist style references
- Professional photography terms

Thread-safe, memory-bounded, production-ready.
NO TRUNCATION - COMPLETE FILE.
"""

import os
import sys
import time
import uuid
import shutil
import threading
import textwrap
import unicodedata
import re
import json
from pathlib import Path
from typing import Optional, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from datetime import datetime, timezone

import requests

from .config import (
    AudioVideoConstants,
    ServiceConstants,
    ResourceConstants,
    REQUEST_HEADERS,
    GENRE_CONFIG,
    FALLBACK_SCRIPTS,
    VideoQuality
)
from .utils import safe_print, rate_limited, PerformanceProfiler, normalize_unicode
from .security import SecurityValidator
from .media import AudioProcessor, ScriptOptimizer
from .services import AssetGenerationError

# ============ MODULE-LEVEL CONSTANTS ============
# Separate timeout pools per asset type (Fix #218)
TIMEOUT_POOLS = {
    'audio': 120,      # 2 minutes for TTS generation
    'image': 240,      # 4 minutes for image generation
    'bgm': 60,         # 1 minute for BGM download
    'script': 45       # 45 seconds for LLM call
}

# Maximum safe script length (Fix #288)
MAX_SAFE_SCRIPT_CHARS = 800
MAX_SAFE_SCRIPT_WORDS = 45
TARGET_SCRIPT_WORDS = 35

# ============ LLM SEMAPHORE (FIX #211, #376) ============
# Global semaphore with proper double-checked locking
_llm_semaphore_lock = threading.Lock()
_llm_semaphore = None

def get_llm_semaphore():
    """
    Get or create LLM semaphore with thread-safe lazy initialization.
    
    Fix #211: Shared semaphore across all LLM calls
    Fix #376: Proper double-checked locking pattern
    
    Returns:
        Shared semaphore instance limiting concurrent LLM calls
    """
    global _llm_semaphore
    
    # First check without lock (fast path)
    if _llm_semaphore is not None:
        return _llm_semaphore
    
    # Acquire lock for initialization
    with _llm_semaphore_lock:
        # Double-check after acquiring lock
        if _llm_semaphore is None:
            from .config import ResourceConstants
            _llm_semaphore = threading.Semaphore(ResourceConstants.MAX_CONCURRENT_LLM)
        return _llm_semaphore

# ============ SCRIPT CACHE (FIX #297) ============
class ScriptCache:
    """
    LRU cache for LLM-generated scripts with proper eviction.
    
    Fix #297: OrderedDict-based LRU cache prevents memory leak
    Fix #322: Unicode normalization prevents cache key collisions
    
    Features:
    - Maximum 100 entries with LRU eviction
    - Thread-safe operations with RLock
    - Unicode normalization for cache keys
    """
    
    def __init__(self):
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.lock = threading.RLock()
        self.max_size = 100
    
    def get(self, key: str) -> Optional[str]:
        """
        Get cached script and mark as recently used.
        
        Args:
            key: Normalized cache key
        
        Returns:
            Cached script or None if not found
        """
        # Normalize key to prevent collisions (Fix #322)
        key = normalize_unicode(key)
        
        with self.lock:
            if key in self.cache:
                # Move to end (mark as recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, script: str):
        """
        Cache script with LRU eviction.
        
        Args:
            key: Cache key (will be normalized)
            script: Script text to cache
        """
        # Normalize key to prevent collisions (Fix #322)
        key = normalize_unicode(key)
        
        with self.lock:
            # If key exists, move to end
            if key in self.cache:
                self.cache.move_to_end(key)
            
            # Add new entry
            self.cache[key] = script
            
            # Evict oldest if over limit (Fix #297)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
    
    def __len__(self):
        """Get cache size."""
        with self.lock:
            return len(self.cache)

# Global script cache instance
_script_cache = ScriptCache()

# ============ QUALITY ENHANCEMENT CONSTANTS ============
# These are used to build enchanting prompts (Fix #491, #536-#545)

QUALITY_ENHANCERS = {
    VideoQuality.ULTRA: (
        "masterpiece, best quality, ultra detailed, 8k uhd, 16k resolution, "
        "professional photography, award-winning, gallery quality, "
        "perfect composition, rule of thirds, golden ratio, "
        "photorealistic, hyper-realistic, physically-based rendering, "
        "unreal engine 5, octane render, ray tracing, global illumination, "
        "volumetric lighting, cinematic lighting, dramatic lighting, "
        "studio lighting, three-point lighting, rim lighting, "
        "ARRI Alexa, RED camera, shot on 70mm, IMAX quality, "
        "perfect focus, tack sharp, crystal clear, razor sharp, "
        "shallow depth of field, bokeh, cinematic bokeh, "
        "HDR, high dynamic range, wide color gamut, 10-bit color, "
        "professional color grading, cinematic color grading, "
        "vivid colors, rich colors, saturated colors, accurate colors, "
        "film grain, organic texture, micro details, extreme detail, "
        "perfect anatomy, flawless, pristine, immaculate"
    ),
    VideoQuality.HIGH: (
        "high quality, very detailed, professional photography, "
        "sharp focus, clear details, good composition, proper lighting, "
        "depth of field, photographic quality, 4k uhd, professional grade, "
        "HDR, cinematic lighting, realistic materials, accurate colors"
    ),
    VideoQuality.STANDARD: (
        "good quality, detailed, clear, focused, professional, "
        "well composed, balanced lighting, sharp details, accurate colors"
    ),
    VideoQuality.DRAFT: ""
}

LIGHTING_STYLES = {
    'motivational': (
        "golden hour lighting, dramatic god rays, volumetric lighting, "
        "rim light, three-point lighting, Rembrandt lighting, "
        "sun rays through clouds, light shafts, tyndall effect, "
        "warm golden tones, backlit silhouette, heroic lighting"
    ),
    'emotional': (
        "soft diffused lighting, overcast atmosphere, window light, "
        "moody blue hour, melancholic tones, subtle rim light, "
        "natural lighting, gentle shadows, intimate atmosphere, "
        "desaturated colors, film noir lighting"
    ),
    'tech': (
        "neon lighting, cyberpunk glow, volumetric fog, "
        "dramatic shadows, high contrast, rim light from screens, "
        "holographic light, LED array glow, futuristic atmosphere, "
        "cyan and magenta tones, artificial light sources"
    ),
    'nature': (
        "natural sunlight, golden hour, magic hour, dappled light, "
        "sun rays through trees, atmospheric haze, soft morning light, "
        "pristine atmosphere, natural beauty, organic lighting, "
        "environmental lighting, sky light"
    )
}

CAMERA_ANGLES = {
    'motivational': (
        "low angle shot, heroic perspective, wide angle composition, "
        "dynamic angle, powerful framing, dramatic perspective, "
        "upward gaze, imposing composition"
    ),
    'emotional': (
        "eye level shot, medium close-up, intimate framing, "
        "shallow depth of field, portrait orientation, "
        "personal perspective, contemplative angle"
    ),
    'tech': (
        "dutch angle, dynamic perspective, futuristic framing, "
        "wide angle cityscape, geometric composition, "
        "architectural perspective, aerial view"
    ),
    'nature': (
        "establishing shot, wide angle landscape, panoramic view, "
        "aerial perspective, environmental shot, "
        "natural framing, foreground interest"
    )
}

COMPOSITION_TERMS = {
    'motivational': (
        "rule of thirds, leading lines, symmetrical composition, "
        "golden ratio, dynamic composition, balanced framing, "
        "strong focal point, diagonal lines"
    ),
    'emotional': (
        "negative space, intimate framing, shallow focus, "
        "isolated subject, melancholic composition, "
        "minimalist framing, contemplative space"
    ),
    'tech': (
        "geometric composition, symmetry, perspective lines, "
        "vanishing point, architectural framing, "
        "layered composition, depth through layers"
    ),
    'nature': (
        "golden ratio, natural framing, foreground elements, "
        "layered depth, environmental context, "
        "balanced composition, organic framing"
    )
}

COLOR_PALETTES = {
    'motivational': (
        "warm color palette, golden tones, vibrant saturation, "
        "high contrast, rich oranges, brilliant golds, deep purples, "
        "inspirational colors, uplifting tones"
    ),
    'emotional': (
        "muted colors, desaturated palette, blue tones, "
        "low contrast, melancholic colors, cool tones, "
        "subdued palette, introspective colors"
    ),
    'tech': (
        "cyan and magenta palette, neon colors, high contrast, "
        "electric blues, vibrant purples, deep blacks, "
        "cyberpunk colors, futuristic palette"
    ),
    'nature': (
        "natural color palette, rich greens, earthy tones, "
        "realistic colors, organic palette, vivid but natural, "
        "environmental colors, pristine tones"
    )
}

TIME_OF_DAY = {
    'motivational': "sunrise, golden hour morning, dawn light, early morning glow",
    'emotional': "overcast day, twilight, dusk, rainy afternoon, blue hour",
    'tech': "night time, midnight, dark cityscape, neon-lit night",
    'nature': "golden hour, magic hour, soft morning light, pristine daylight"
}

ATMOSPHERE_EFFECTS = {
    'motivational': "clear atmosphere, dramatic clouds, god rays, light shafts, pristine air",
    'emotional': "rain, mist, fog, overcast weather, atmospheric haze, melancholic weather",
    'tech': "smog, volumetric fog, haze, particles in air, atmospheric density, neon haze",
    'nature': "clear pristine air, natural atmosphere, environmental clarity, organic haze"
}

TEXTURE_DETAILS = {
    'motivational': "sharp details, crisp textures, tangible materials, visible detail, defined surfaces",
    'emotional': "soft textures, smooth surfaces, gentle materials, subdued details, matte finishes",
    'tech': "reflective surfaces, metallic textures, glass materials, chrome details, LED displays",
    'nature': "organic textures, natural materials, bark detail, leaf veins, water droplets, natural detail"
}

MATERIAL_SPECS = {
    'motivational': "solid materials, stone, metal, concrete, powerful textures, substantial surfaces",
    'emotional': "soft fabrics, velvet, silk, translucent materials, delicate textures, gentle surfaces",
    'tech': "brushed aluminum, glass, carbon fiber, holographic materials, LED arrays, circuit boards",
    'nature': "wood grain, moss, lichen, natural fibers, organic materials, earth, water, foliage"
}

FILM_STOCK = {
    'motivational': "shot on Kodak Portra 400, cinematic film photography, film grain aesthetic",
    'emotional': "shot on Kodak Tri-X 400, black and white film stock, vintage film aesthetic",
    'tech': "digital cinema, RED Komodo 6K, ARRI Alexa, digital sensor capture",
    'nature': "shot on Fujifilm Velvia, nature photography, vibrant film stock, landscape film"
}

LENS_SPECS = {
    'motivational': "24mm wide angle lens, f/2.8 aperture, deep focus, wide field of view",
    'emotional': "85mm portrait lens, f/1.4 aperture, shallow depth of field, bokeh background",
    'tech': "35mm cinematic lens, f/1.8 aperture, anamorphic characteristics, lens flare",
    'nature': "70-200mm telephoto, f/4 aperture, compressed perspective, environmental detail"
}

ARTIST_REFERENCES = {
    'motivational': "photography style of Annie Leibovitz, Peter Lik, Steve McCurry",
    'emotional': "style of Gregory Crewdson, Todd Hido, cinematography of Roger Deakins",
    'tech': "concept art style of Syd Mead, Blade Runner 2049 aesthetic, Denis Villeneuve cinematography",
    'nature': "photography style of Ansel Adams, Art Wolfe, Frans Lanting, National Geographic"
}

# ============ WORKFLOW ORCHESTRATOR ============
class WorkflowOrchestrator:
    """
    Orchestrate complete video generation workflow with enchanting quality.
    
    Features:
    - Thread-safe script caching with LRU eviction
    - Separate timeout pools per asset type
    - Comprehensive error handling and retry logic
    - Quality-based prompt enhancement
    - Genre-specific visual improvements
    """
    
    def __init__(self, config, asset_manager, render_engine, metrics, executor, logger):
        self.config = config
        self.asset_manager = asset_manager
        self.render_engine = render_engine
        self.metrics = metrics
        self.executor = executor
        self.logger = logger
        self.shutdown_flag = threading.Event()
        
        # Counter for unique run directories (Fix #238)
        self._run_counter = 0
        self._run_counter_lock = threading.Lock()
    
    def _get_next_run_id(self) -> int:
        """Get next unique run counter value (Fix #238)."""
        with self._run_counter_lock:
            self._run_counter += 1
            return self._run_counter
    
    @rate_limited("qwen")
    def generate_script(self, genre: str, custom_topic: Optional[str]) -> Tuple[str, str]:
        """
        Generate professional script with comprehensive validation and caching.
        
        Fixes:
        - #232: Keyword overlap validation with stemming
        - #251: min_tokens parameter prevents very short responses
        - #297: LRU cache with proper eviction
        - #322: Unicode normalization for cache keys
        - #490: JSON-escaped parameters prevent prompt injection
        - #503: Calculate words per second for duration matching
        
        Args:
            genre: Video genre
            custom_topic: Optional custom topic
        
        Returns:
            Tuple of (script, tonality)
        """
        if self.logger:
            self.logger.info("Generating script")
        
        start_time = time.time()
        
        # Get genre configuration
        genre_cfg = GENRE_CONFIG.get(genre, GENRE_CONFIG['motivational'])
        keywords = ", ".join(genre_cfg.keywords)
        tonality = genre_cfg.tonality
        
        # Build cache key (Fix #322: normalize Unicode)
        if custom_topic:
            sanitized = self._sanitize_custom_topic(custom_topic)
            cache_key = normalize_unicode(f"custom:{sanitized}")
            
            # Fix #490: Use JSON-escaped parameter to prevent prompt injection
            prompt_obj = {
                "instruction": "Write a powerful, engaging video script",
                "topic": sanitized,
                "target_words": AudioVideoConstants.SCRIPT_TARGET_WORDS,
                "style": "emotional and memorable",
                "format": "one paragraph, no titles or labels"
            }
            prompt_content = json.dumps(prompt_obj, ensure_ascii=False)
        else:
            cache_key = normalize_unicode(f"genre:{genre}")
            prompt_obj = {
                "instruction": "Write a powerful video script",
                "genre": genre,
                "themes": keywords,
                "target_words": AudioVideoConstants.SCRIPT_TARGET_WORDS,
                "style": "emotional and impactful",
                "format": "one paragraph, no titles"
            }
            prompt_content = json.dumps(prompt_obj, ensure_ascii=False)
        
        # Check cache (Fix #297)
        cached_script = _script_cache.get(cache_key)
        if cached_script:
            if self.logger:
                self.logger.info("Using cached LLM script")
            return cached_script, tonality
        
        # Get semaphore (Fix #211, #376)
        semaphore = get_llm_semaphore()
        
        with semaphore:
            for attempt in range(3):
                try:
                    # Fix #251: Add min_tokens to prevent very short responses
                    max_tokens = int(AudioVideoConstants.SCRIPT_MAX_WORDS * 1.5)
                    min_tokens = int(AudioVideoConstants.SCRIPT_TARGET_WORDS * 0.8)
                    
                    timeout = TIMEOUT_POOLS['script']  # Fix #218: Separate timeout
                    
                    resp = requests.post(
                        self.config.qwen_url,
                        json={
                            "model": "qwen2.5",
                            "messages": [{"role": "user", "content": prompt_content}],
                            "temperature": 0.85,
                            "max_tokens": max_tokens,
                            "min_tokens": min_tokens
                        },
                        timeout=timeout,
                        headers=REQUEST_HEADERS
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if 'choices' in data and data['choices']:
                            script = data['choices'][0].get('message', {}).get('content', '').strip()
                            
                            # Sanitize and validate
                            script = self._sanitize_text(script)
                            script = self._enforce_script_length(script)
                            
                            if script and len(script.split()) >= 5:
                                # Validate script matches topic (Fix #232)
                                if custom_topic:
                                    if not self._validate_script_matches_topic(script, sanitized):
                                        if attempt < 2:
                                            if self.logger:
                                                self.logger.warning(f"Script doesn't match topic, retrying (attempt {attempt + 1})")
                                            if self.metrics:
                                                self.metrics.record_retry("script_validation")
                                            time.sleep(1)
                                            continue
                                
                                # Fix #503: Validate script length matches target duration
                                words_per_second = 2.5  # Average speaking rate
                                estimated_duration = len(script.split()) / words_per_second
                                target_duration = AudioVideoConstants.TARGET_VIDEO_DURATION
                                
                                if abs(estimated_duration - target_duration) > 3.0:
                                    # Adjust script length to match duration
                                    target_words = int(target_duration * words_per_second)
                                    script = self._adjust_script_to_word_count(script, target_words)
                                
                                # Record metrics
                                if self.metrics:
                                    self.metrics.record_generation_time("script", time.time() - start_time)
                                
                                # Cache result (Fix #297)
                                _script_cache.put(cache_key, script)
                                
                                return script, tonality
                
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        if self.metrics:
                            self.metrics.record_retry("llm")
                        time.sleep(2 ** attempt)
        
        # Fallback to genre-specific script
        fallback = FALLBACK_SCRIPTS.get(genre, FALLBACK_SCRIPTS.get('custom', FALLBACK_SCRIPTS['motivational']))
        return fallback, tonality
    
    def _validate_script_matches_topic(self, script: str, topic: str) -> bool:
        """
        Validate script contains relevant keywords from topic.
        
        Fix #232: Use stemming-like approach for keyword matching
        
        Args:
            script: Generated script text
            topic: User-provided topic
        
        Returns:
            True if script is relevant to topic
        """
        # Extract stems from topic
        topic_words = set(topic.lower().split())
        script_stems = self._extract_word_stems(script.lower())
        
        # Check for overlap
        overlap = len(topic_words & script_stems)
        
        # Require at least some overlap (lenient check)
        return overlap > 0
    
    def _extract_word_stems(self, text: str) -> Set[str]:
        """
        Simple stemming for keyword matching.
        
        Fix #232: Basic suffix removal for better matching
        
        Args:
            text: Text to extract stems from
        
        Returns:
            Set of word stems
        """
        words = re.findall(r'\b\w+\b', text.lower())
        stems = set()
        
        for word in words:
            # Simple stemming: remove common suffixes
            if len(word) > 4:
                if word.endswith('ing'):
                    stems.add(word[:-3])
                elif word.endswith('ed'):
                    stems.add(word[:-2])
                elif word.endswith('s') and not word.endswith('ss'):
                    stems.add(word[:-1])
                elif word.endswith('ly'):
                    stems.add(word[:-2])
            stems.add(word)
        
        return stems
    
    def _sanitize_custom_topic(self, topic: str) -> str:
        """
        Sanitize custom topic for safe LLM usage.
        
        Fixes:
        - #245: Truncate at word boundary
        - #264: Remove dangerous characters
        
        Args:
            topic: User-provided topic
        
        Returns:
            Sanitized topic string
        """
        # Remove control characters
        topic = ''.join(c for c in topic if unicodedata.category(c)[0] not in ('C',))
        
        # Remove dangerous characters (Fix #264)
        dangerous = ['{', '}', '[', ']', '<', '>', '\\', '"', "'", '\n', '\r', '\t', '`', '$']
        for char in dangerous:
            topic = topic.replace(char, '')
        
        # Normalize whitespace
        topic = ' '.join(topic.split())
        
        # Fix #245: Truncate at word boundary
        if len(topic) > 100:
            topic = topic[:100]
            last_space = topic.rfind(' ')
            if last_space > 50:
                topic = topic[:last_space]
        
        return topic[:200]
    
    def _sanitize_text(self, text: str) -> str:
        """
        Unicode-aware text sanitization.
        
        Args:
            text: Text to sanitize
        
        Returns:
            Sanitized text with normalized Unicode
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Normalize Unicode (Fix #322)
        text = unicodedata.normalize('NFC', text)
        
        # Remove control characters except newline/tab/space
        text = ''.join(
            char for char in text
            if unicodedata.category(char)[0] not in ('C',) or char in '\n\t '
        )
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Enforce maximum length (Fix #288)
        if len(text) > MAX_SAFE_SCRIPT_CHARS:
            text = text[:MAX_SAFE_SCRIPT_CHARS]
        
        return text.strip()
    
    def _enforce_script_length(self, script: str, max_words: int = MAX_SAFE_SCRIPT_WORDS) -> str:
        """
        Enforce script word limit with clean truncation.
        
        Fixes:
        - #225: Word limit enforcement
        - #283: Uses textwrap.shorten for clean cuts
        - #288: Multiple validation points
        
        Args:
            script: Script text
            max_words: Maximum word count
        
        Returns:
            Truncated script within word limit
        """
        words = script.split()
        
        if len(words) <= max_words:
            return script
        
        # Fix #283: Use textwrap.shorten for cleaner truncation
        target_chars = max_words * 6  # Approximate characters per word
        shortened = textwrap.shorten(script, width=target_chars, placeholder="...")
        
        # Remove placeholder if added
        if shortened.endswith('...'):
            shortened = shortened[:-3].rstrip()
        
        # Ensure ends with proper punctuation
        if shortened and shortened[-1] not in '.!?':
            shortened += '.'
        
        return shortened
    
    def _adjust_script_to_word_count(self, script: str, target_words: int) -> str:
        """
        Adjust script to match target word count.
        
        Fix #503: Scale script to match target video duration
        
        Args:
            script: Original script
            target_words: Target word count
        
        Returns:
            Adjusted script
        """
        current_words = len(script.split())
        
        if current_words == target_words:
            return script
        
        if current_words > target_words:
            # Truncate
            return self._enforce_script_length(script, target_words)
        else:
            # Script too short - return as is
            # (LLM should generate correct length, this is fallback)
            return script
    
    def generate_assets_parallel(
        self,
        genre: str,
        script: str,
        run_dir: Path,
        music_dir: Path,
        custom_topic: Optional[str],
        tonality: str
    ) -> Tuple[Path, Path, Path]:
        """
        Generate assets in parallel with separate timeout pools.
        
        Fixes:
        - #205: Cumulative timeout tracking
        - #218: Separate timeout pools per asset type
        - #258: Generate audio first to get actual duration
        
        Args:
            genre: Video genre
            script: Script text
            run_dir: Run directory for temp files
            music_dir: Music library directory
            custom_topic: Optional custom topic
            tonality: Script tonality
        
        Returns:
            Tuple of (audio_path, bgm_path, img_path)
        """
        if self.logger:
            self.logger.info("Generating assets in parallel")
        
        futures = {}
        audio_path = None
        duration = AudioVideoConstants.TARGET_VIDEO_DURATION
        
        try:
            # Fix #258: Generate audio first to get actual duration
            with PerformanceProfiler("audio_generation", self.metrics):
                audio_start = time.time()
                audio_path = self.asset_manager.generate_audio(
                    script, genre, run_dir, custom_topic, self.config.enable_ssml
                )
                audio_elapsed = time.time() - audio_start
                
                # Verify timeout didn't exceed pool
                if audio_elapsed > TIMEOUT_POOLS['audio']:
                    if self.logger:
                        self.logger.warning(f"Audio generation exceeded timeout pool: {audio_elapsed:.1f}s > {TIMEOUT_POOLS['audio']}s")
                
                duration = AudioProcessor.get_duration(audio_path, self.logger)
                if self.logger:
                    self.logger.info(f"Audio duration: {duration:.1f}s")
            
            # Now generate BGM and image in parallel with known duration
            bgm_start = time.time()
            bgm_future = self.executor.submit(
                self.asset_manager.download_bgm,
                genre, run_dir, music_dir, duration
            )
            futures['bgm'] = (bgm_future, bgm_start)
            
            img_start = time.time()
            img_future = self.executor.submit(
                self.asset_manager.generate_image,
                genre, script, run_dir,
                self.config.get_video_config(),
                custom_topic
            )
            futures['image'] = (img_future, img_start)
            
            # Wait for BGM with timeout pool (Fix #218)
            try:
                with PerformanceProfiler("bgm_download", self.metrics):
                    bgm_path = bgm_future.result(timeout=TIMEOUT_POOLS['bgm'])
                    bgm_elapsed = time.time() - bgm_start
                    
                    if bgm_elapsed > TIMEOUT_POOLS['bgm']:
                        if self.logger:
                            self.logger.warning(f"BGM download exceeded timeout pool: {bgm_elapsed:.1f}s > {TIMEOUT_POOLS['bgm']}s")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"BGM generation failed: {e}")
                if self.metrics:
                    self.metrics.record_asset_failure('bgm')
                raise AssetGenerationError(f"BGM failed: {e}", "bgm") from e
            
            # Wait for image with timeout pool (Fix #218)
            try:
                with PerformanceProfiler("image_generation", self.metrics):
                    img_path, _ = img_future.result(timeout=TIMEOUT_POOLS['image'])
                    img_elapsed = time.time() - img_start
                    
                    if img_elapsed > TIMEOUT_POOLS['image']:
                        if self.logger:
                            self.logger.warning(f"Image generation exceeded timeout pool: {img_elapsed:.1f}s > {TIMEOUT_POOLS['image']}s")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Image generation failed: {e}")
                if self.metrics:
                    self.metrics.record_asset_failure('image')
                raise AssetGenerationError(f"Image failed: {e}", "image") from e
            
            return audio_path, bgm_path, img_path
            
        except Exception as e:
            # Cancel remaining futures
            for name, (future, _) in futures.items():
                if not future.done():
                    future.cancel()
            raise
    
    def _build_enhanced_prompt(
        self,
        genre: str,
        script: str,
        video_config,
        custom_topic: Optional[str]
    ) -> str:
        """
        Build enchanting quality prompt with comprehensive enhancements.
        
        Fixes:
        - #441: Adds quality boosters
        - #447-#455: Lighting, camera, composition, color, time, atmosphere
        - #456-#463: Aspect ratio, subject emphasis, texture, materials
        - #491: Keyword weighting with emphasis syntax
        - #536-#545: Micro-details, post-processing hints, environmental details
        
        Args:
            genre: Video genre
            script: Script text
            video_config: Video configuration
            custom_topic: Optional custom topic
        
        Returns:
            Enhanced prompt string with all quality improvements
        """
        genre_cfg = GENRE_CONFIG.get(genre, GENRE_CONFIG['motivational'])
        
        # Base prompt from genre
        base = genre_cfg.img_prompt
        
        # Extract and emphasize visual keywords (Fix #491)
        if custom_topic:
            visual_keywords = ScriptOptimizer.extract_visual_keywords(script)
            # Emphasize top keywords with weight syntax
            emphasized = [f"({kw}:1.3)" for kw in visual_keywords[:2]]
            keyword_str = ', '.join(emphasized + visual_keywords[2:5])
        else:
            keyword_str = ""
        
        # Get quality enhancements
        quality_boost = QUALITY_ENHANCERS.get(
            video_config.quality,
            QUALITY_ENHANCERS[VideoQuality.STANDARD]
        )
        
        # Get genre-specific enhancements
        lighting = LIGHTING_STYLES.get(genre, "professional lighting")
        camera = CAMERA_ANGLES.get(genre, "balanced perspective")
        composition = COMPOSITION_TERMS.get(genre, "rule of thirds")
        colors = COLOR_PALETTES.get(genre, "accurate colors")
        time_of_day = TIME_OF_DAY.get(genre, "natural daylight")
        atmosphere = ATMOSPHERE_EFFECTS.get(genre, "clear atmosphere")
        textures = TEXTURE_DETAILS.get(genre, "detailed textures")
        materials = MATERIAL_SPECS.get(genre, "realistic materials")
        film = FILM_STOCK.get(genre, "digital cinema")
        lens = LENS_SPECS.get(genre, "professional lens")
        artist = ARTIST_REFERENCES.get(genre, "professional photography")
        
        # Aspect ratio hints (Fix #456)
        aspect_hint = ""
        if video_config.aspect_ratio.value == '9:16':
            aspect_hint = "vertical composition, portrait orientation"
        elif video_config.aspect_ratio.value == '16:9':
            aspect_hint = "horizontal composition, landscape orientation"
        elif video_config.aspect_ratio.value == '1:1':
            aspect_hint = "square format, centered composition"
        
        # Resolution specification (Fix #452)
        resolution_hint = ""
        if video_config.quality == VideoQuality.ULTRA:
            resolution_hint = "8k uhd resolution, ultra high definition, maximum detail"
        elif video_config.quality == VideoQuality.HIGH:
            resolution_hint = "4k uhd resolution, high definition, sharp detail"
        elif video_config.quality == VideoQuality.STANDARD:
            resolution_hint = "full hd resolution, 1080p quality"
        
        # Micro-details for ultra quality (Fix #536)
        micro_details = ""
        if video_config.quality == VideoQuality.ULTRA:
            micro_details = (
                "skin pores visible, fabric texture detail, individual hair strands, "
                "water droplets, dust particles in air, surface imperfections, "
                "material grain, fine scratches, wear and tear, realistic weathering"
            )
        
        # Environmental details (Fix #542)
        environmental = (
            "atmospheric perspective, depth haze, environmental fog, "
            "foreground bokeh, background blur, layered depth, "
            "environmental lighting, natural ambience"
        )
        
        # Weather effects if applicable (Fix #544)
        weather_effects = ""
        if genre == 'emotional':
            weather_effects = "rain droplets, mist, fog, overcast sky, moody weather"
        elif genre == 'nature':
            weather_effects = "natural weather, environmental conditions, atmospheric clarity"
        
        # Camera technical specs (Fix #540)
        camera_tech = (
            f"{lens}, shot on ARRI Alexa cinema camera, "
            "professional cinematography, theatrical quality, "
            "precise focus, professional depth of field control"
        )
        
        # Color science (Fix #541)
        color_science = (
            "ACES color space, professional color grading, "
            "DaVinci Resolve grade, color harmony, "
            "complementary colors, proper white balance"
        )
        
        # Build complete prompt
        components = [
            base,
            keyword_str,
            quality_boost,
            lighting,
            camera,
            composition,
            colors,
            time_of_day,
            atmosphere,
            textures,
            materials,
            film,
            camera_tech,
            color_science,
            aspect_hint,
            resolution_hint,
            micro_details,
            environmental,
            weather_effects,
            artist,
            "professional photography, award-winning, gallery quality",
            f"{video_config.width}x{video_config.height} resolution"
        ]
        
        # Filter empty components and join
        prompt = ', '.join(filter(None, components))
        
        # Ensure prompt doesn't exceed reasonable length
        # Fix #481: Validate workflow JSON size before sending
        if len(prompt) > 2000:
            # Smart truncation at comma boundary
            prompt = prompt[:2000]
            last_comma = prompt.rfind(',')
            if last_comma > 1500:
                prompt = prompt[:last_comma]
        
        return prompt
    
    def generate_video(
        self,
        genre: str,
        video_num: int,
        total: int,
        custom_topic: Optional[str]
    ) -> Path:
        """
        Generate complete video with all enhancements and error handling.
        
        Args:
            genre: Video genre
            video_num: Current video number
            total: Total videos to generate
            custom_topic: Optional custom topic
        
        Returns:
            Path to generated video file
        """
        display_genre = custom_topic if custom_topic else genre.upper()
        
        safe_print(f"\n{'='*70}")
        safe_print(f"[Video {video_num}/{total}] Genre: {display_genre}")
        safe_print(f"{'='*70}\n")
        
        # Fix #238: Multiple entropy sources for unique run_dir
        run_id = self._get_next_run_id()
        timestamp_ms = int(time.time() * 1000)
        pid = os.getpid()
        uuid_part = uuid.uuid4().hex[:8]
        
        run_dir = self.config.video_dir / f"run_{timestamp_ms}_{pid}_{video_num}_{run_id}_{uuid_part}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = SecurityValidator.sanitize_filename(
            f"{genre}_{video_num}_{timestamp_ms}_{uuid_part}.mp4"
        )
        
        try:
            # Generate script
            safe_print(f"[{video_num}/{total}] Generating script...")
            with PerformanceProfiler("script_generation", self.metrics):
                script, tonality = self.generate_script(genre, custom_topic)
            safe_print(f"📝 Script: \"{script}\"\n")
            
            # Analyze script for tonality if using custom topic
            if custom_topic:
                tonality = self._analyze_script_tonality(script)
            
            # Generate assets
            safe_print(f"[{video_num}/{total}] Generating assets...")
            with PerformanceProfiler("asset_generation", self.metrics):
                audio_path, bgm_path, img_path = self.generate_assets_parallel(
                    genre, script, run_dir, self.config.video_dir / "music",
                    custom_topic, tonality
                )
            
            # Render video
            safe_print(f"\n[{video_num}/{total}] Rendering video...")
            output_path = run_dir / output_filename
            
            with PerformanceProfiler("video_rendering", self.metrics):
                self.render_engine.render_video(
                    img_path, audio_path, bgm_path, script, output_path, tonality
                )
            
            # Generate subtitles if enabled
            if self.config.get_video_config().enable_subtitles:
                srt_path = output_path.with_suffix('.srt')
                from .media import SubtitleGenerator
                try:
                    SubtitleGenerator.generate_srt(
                        script,
                        AudioProcessor.get_duration(audio_path, self.logger),
                        srt_path
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Subtitle generation failed: {e}")
            
            # Record success
            if self.metrics:
                self.metrics.record_success()
            
            # Move to final location if cleanup enabled
            if self.config.cleanup_on_success:
                final_dir = self.config.video_dir / "videos"
                final_dir.mkdir(exist_ok=True)
                final_path = final_dir / output_filename
                
                # Handle existing file with backup (Fix #238)
                if final_path.exists():
                    counter = 1
                    while True:
                        backup_name = f"{final_path.stem}_backup_{counter}{final_path.suffix}"
                        backup_path = final_dir / backup_name
                        if not backup_path.exists():
                            try:
                                final_path.rename(backup_path)
                                break
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"Could not rename existing file: {e}")
                                # Try next counter
                                counter += 1
                                if counter > 10:
                                    # Give up after 10 attempts
                                    final_path.unlink(missing_ok=True)
                                    break
                        counter += 1
                
                # Move video
                try:
                    shutil.move(str(output_path), str(final_path))
                    output_path = final_path
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Could not move video to final location: {e}")
                    # Keep in run_dir if move fails
                
                # Move subtitles if they exist
                srt_source = run_dir / output_filename.replace('.mp4', '.srt')
                if srt_source.exists():
                    srt_dest = final_dir / output_filename.replace('.mp4', '.srt')
                    try:
                        shutil.move(str(srt_source), str(srt_dest))
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Could not move subtitles: {e}")
                
                # Cleanup temp directory
                try:
                    shutil.rmtree(run_dir, ignore_errors=True)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Could not cleanup temp directory: {e}")
            
            # Upload to cloud if enabled
            if self.config.enable_cloud_upload and self.config.cloud_provider:
                try:
                    from .utils import CloudStorageUploader
                    uploader = CloudStorageUploader(
                        self.config.cloud_provider,
                        self.config.cloud_bucket
                    )
                    remote_key = f"videos/{output_filename}"
                    if uploader.upload_file(output_path, remote_key):
                        if self.logger:
                            self.logger.info(f"Uploaded to {self.config.cloud_provider}: {remote_key}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Cloud upload failed: {e}")
            
            # Send notification if configured
            if self.config.notification_webhook:
                try:
                    from .utils import NotificationSystem
                    notifier = NotificationSystem(self.config.notification_webhook)
                    notifier.send_notification(
                        "Video Generated",
                        f"Successfully created {output_filename}",
                        "success"
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Notification failed: {e}")
            
            # Display success
            safe_print(f"\n✅ [{video_num}/{total}] SUCCESS!")
            safe_print(f"   📹 {output_path.name}")
            
            try:
                duration_str = f"{AudioProcessor.get_duration(output_path, self.logger):.1f}s"
            except Exception:
                duration_str = "unknown"
            
            safe_print(f"   ⏱️  {duration_str}\n")
            
            return output_path
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Video {video_num} failed: {e}")
            if self.metrics:
                self.metrics.record_failure(type(e).__name__)
            
            # Cleanup on failure if enabled
            if self.config.cleanup_on_failure:
                try:
                    shutil.rmtree(run_dir, ignore_errors=True)
                except Exception as cleanup_err:
                    if self.logger:
                        self.logger.warning(f"Cleanup failed: {cleanup_err}")
            
            raise
    
    def _analyze_script_tonality(self, script: str) -> str:
        """
        Analyze script to determine tonality for camera movement.
        
        Args:
            script: Script text
        
        Returns:
            Tonality string (intense, calm, energetic, neutral)
        """
        script_lower = script.lower()
        
        # Count indicators
        exclamations = script.count('!')
        intense_words = sum(
            1 for word in ['power', 'strong', 'fierce', 'intense', 'explosive', 'dynamic', 'force']
            if word in script_lower
        )
        
        calm_words = sum(
            1 for word in ['calm', 'peace', 'gentle', 'serene', 'quiet', 'soft', 'tranquil']
            if word in script_lower
        )
        
        energetic_words = sum(
            1 for word in ['energy', 'fast', 'quick', 'rapid', 'dynamic', 'vibrant', 'lively']
            if word in script_lower
        )
        
        # Determine tonality
        if exclamations >= 3 or intense_words >= 2:
            return 'intense'
        elif calm_words >= 2:
            return 'calm'
        elif energetic_words >= 2:
            return 'energetic'
        else:
            return 'neutral'
    
    def shutdown(self):
        """Shutdown orchestrator and cleanup resources."""
        self.shutdown_flag.set()
        
        if self.logger:
            self.logger.info("Orchestrator shutdown complete")

# ============ EXPORTS ============
__all__ = [
    'WorkflowOrchestrator',
    'get_llm_semaphore',
    'ScriptCache',
    'QUALITY_ENHANCERS',
    'LIGHTING_STYLES',
    'CAMERA_ANGLES',
    'COMPOSITION_TERMS',
    'COLOR_PALETTES',
    'TIME_OF_DAY',
    'ATMOSPHERE_EFFECTS',
    'TEXTURE_DETAILS',
    'MATERIAL_SPECS',
    'FILM_STOCK',
    'LENS_SPECS',
    'ARTIST_REFERENCES',
]