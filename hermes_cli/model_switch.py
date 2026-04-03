"""Mid-chat model switching pipeline for CLI and gateway.

Rebuilt from scratch to provide:
- Short aliases (sonnet, opus, haiku, gpt5, deepseek, etc.)
- Fuzzy matching with "did you mean?" suggestions
- Clear, actionable error messages at every stage
- Proper provider auto-detection with credential validation
- Clean separation: parse → resolve alias → detect provider → validate → result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Model aliases — short names users can type instead of full slugs
# ═══════════════════════════════════════════════════════════════════════

# Maps short alias → (provider, full_model_name)
# These are the most common models people want to switch to quickly.
# When on OpenRouter, the full slug includes the provider prefix.
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # Anthropic Claude family
    "opus":            ("openrouter", "anthropic/claude-opus-4.6"),
    "sonnet":          ("openrouter", "anthropic/claude-sonnet-4.6"),
    "haiku":           ("openrouter", "anthropic/claude-haiku-4.5"),
    "claude":          ("openrouter", "anthropic/claude-opus-4.6"),
    "sonnet-4.5":      ("openrouter", "anthropic/claude-sonnet-4.5"),
    "sonnet-4.6":      ("openrouter", "anthropic/claude-sonnet-4.6"),
    "opus-4.6":        ("openrouter", "anthropic/claude-opus-4.6"),
    "haiku-4.5":       ("openrouter", "anthropic/claude-haiku-4.5"),

    # OpenAI GPT family
    "gpt5":            ("openrouter", "openai/gpt-5.4"),
    "gpt-5":           ("openrouter", "openai/gpt-5.4"),
    "gpt5-mini":       ("openrouter", "openai/gpt-5.4-mini"),
    "gpt5-pro":        ("openrouter", "openai/gpt-5.4-pro"),
    "gpt5-nano":       ("openrouter", "openai/gpt-5.4-nano"),
    "codex":           ("openrouter", "openai/gpt-5.3-codex"),

    # Google Gemini family
    "gemini":          ("openrouter", "google/gemini-3-pro-preview"),
    "gemini-pro":      ("openrouter", "google/gemini-3-pro-preview"),
    "gemini-flash":    ("openrouter", "google/gemini-3-flash-preview"),

    # DeepSeek
    "deepseek":        ("openrouter", "deepseek/deepseek-chat-v3-1106"),

    # Qwen
    "qwen":            ("openrouter", "qwen/qwen3.6-plus:free"),

    # Misc popular
    "grok":            ("openrouter", "x-ai/grok-4.20-beta"),
    "glm":             ("openrouter", "z-ai/glm-5"),
    "kimi":            ("openrouter", "moonshotai/kimi-k2.5"),
    "minimax":         ("openrouter", "minimax/minimax-m2.7"),
    "mimo":            ("openrouter", "xiaomi/mimo-v2-pro"),
    "nemotron":        ("openrouter", "nvidia/nemotron-3-super-120b-a12b"),
}


# ═══════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelSwitchResult:
    """Result of a model switch attempt."""

    success: bool
    new_model: str = ""
    target_provider: str = ""
    provider_changed: bool = False
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    persist: bool = False
    error_message: str = ""
    warning_message: str = ""
    is_custom_target: bool = False
    provider_label: str = ""
    resolved_via_alias: str = ""  # which alias was used, if any


@dataclass
class CustomAutoResult:
    """Result of switching to bare 'custom' provider with auto-detect."""

    success: bool
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    error_message: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Alias resolution
# ═══════════════════════════════════════════════════════════════════════

def resolve_alias(raw_input: str) -> Optional[tuple[str, str, str]]:
    """Resolve a short alias to (provider, model, alias_used).

    Returns None if the input isn't a known alias.
    """
    key = raw_input.strip().lower()
    if key in MODEL_ALIASES:
        provider, model = MODEL_ALIASES[key]
        return (provider, model, key)
    return None


def suggest_models(raw_input: str, limit: int = 3) -> list[str]:
    """Suggest similar model names when input doesn't match.

    Searches aliases, OpenRouter catalog, and provider catalogs.
    Returns up to `limit` suggestions.
    """
    from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS

    candidates: list[str] = []

    # Add all aliases
    candidates.extend(MODEL_ALIASES.keys())

    # Add OpenRouter model names (both full slug and bare name)
    for model_id, _ in OPENROUTER_MODELS:
        candidates.append(model_id)
        # Also add the bare name (after the slash)
        if "/" in model_id:
            candidates.append(model_id.split("/", 1)[1])

    # Add models from all provider catalogs
    for provider, models in _PROVIDER_MODELS.items():
        for m in models:
            candidates.append(m)
            if "/" in m:
                candidates.append(m.split("/", 1)[1])

    # Deduplicate while preserving order
    seen = set()
    unique: list[str] = []
    for c in candidates:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            unique.append(c)

    query = raw_input.strip().lower()
    matches = get_close_matches(query, [c.lower() for c in unique], n=limit, cutoff=0.5)

    # Map back to original casing
    lower_to_orig = {c.lower(): c for c in unique}
    return [lower_to_orig.get(m, m) for m in matches]


# ═══════════════════════════════════════════════════════════════════════
# Aggregator-aware model resolution
# ═══════════════════════════════════════════════════════════════════════

# Common vendor prefixes on OpenRouter (vendor/model slug format)
_OPENROUTER_VENDORS = {
    "openai", "anthropic", "google", "deepseek", "meta", "mistral",
    "qwen", "minimax", "x-ai", "z-ai", "moonshotai", "nvidia",
    "xiaomi", "stepfun", "arcee-ai", "cohere", "databricks",
}


def _resolve_on_aggregator(
    raw_model: str,
    current_provider: str,
) -> Optional[str]:
    """Try to resolve a bare model name within an aggregator (OpenRouter/Nous).

    Returns the full slug if found, None otherwise.  This prevents
    bare names from triggering unwanted provider switches.

    Resolution order:
      1. Exact match against catalog (full slug or bare name)
      2. Try vendor/model construction with known vendor prefixes
      3. Fuzzy match against catalog bare names
    """
    from hermes_cli.models import OPENROUTER_MODELS

    model_lower = raw_model.lower()

    # Build lookup tables
    slugs = [m for m, _ in OPENROUTER_MODELS]
    slug_lower = {m.lower(): m for m in slugs}
    bare_to_slug: dict[str, str] = {}
    for s in slugs:
        if "/" in s:
            bare = s.split("/", 1)[1].lower()
            bare_to_slug[bare] = s

    # 1. Exact match on full slug
    if model_lower in slug_lower:
        return slug_lower[model_lower]

    # 2. Exact match on bare name
    if model_lower in bare_to_slug:
        return bare_to_slug[model_lower]

    # 3. If input already has a slash (vendor/model format), accept as-is
    #    on the aggregator — OpenRouter will resolve it even if not in
    #    our static catalog
    if "/" in raw_model:
        vendor = raw_model.split("/", 1)[0].lower()
        if vendor in _OPENROUTER_VENDORS:
            return raw_model

    # 4. Try prepending known vendor prefixes
    for vendor in _OPENROUTER_VENDORS:
        candidate = f"{vendor}/{raw_model}"
        if candidate.lower() in slug_lower:
            return slug_lower[candidate.lower()]

    # 5. Fuzzy match on bare names (for close typos)
    all_bare = list(bare_to_slug.keys())
    close = get_close_matches(model_lower, all_bare, n=1, cutoff=0.75)
    if close:
        return bare_to_slug[close[0]]

    return None


# ═══════════════════════════════════════════════════════════════════════
# Core switch pipeline
# ═══════════════════════════════════════════════════════════════════════

def switch_model(
    raw_input: str,
    current_provider: str,
    current_model: str = "",
    current_base_url: str = "",
    current_api_key: str = "",
) -> ModelSwitchResult:
    """Core model-switching pipeline.

    Steps:
      1. Check alias table (sonnet, opus, gpt5, etc.)
      2. Handle vendor:model on aggregators (openai:gpt-5.4 → openai/gpt-5.4)
      3. Parse provider:model syntax
      4. If on aggregator, resolve within aggregator first
      5. Fall back to cross-provider detection
      6. Resolve credentials
      7. Catalog validation with fuzzy suggestions
      8. Build result

    The caller handles state mutation, config persistence, and output.
    """
    from hermes_cli.models import (
        parse_model_input,
        detect_provider_for_model,
        _PROVIDER_LABELS,
        _PROVIDER_MODELS,
        _KNOWN_PROVIDER_NAMES,
        OPENROUTER_MODELS,
        opencode_model_api_mode,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    stripped = raw_input.strip()
    if not stripped:
        return ModelSwitchResult(
            success=False,
            error_message="No model specified. Usage: /model <name> or /model provider:model",
        )

    on_aggregator = current_provider in {"openrouter", "nous"}

    # ── Step 1: Try alias resolution first ──
    alias_result = resolve_alias(stripped)
    resolved_alias = ""
    if alias_result:
        target_provider, new_model, resolved_alias = alias_result
    else:
        # ── Step 2: Handle vendor:model on aggregators ──
        # Users type "openai:gpt-5.4" expecting it to work.  On OpenRouter,
        # this means "openai/gpt-5.4" (the OpenRouter slug), not a switch
        # to a "openai" hermes provider.
        if on_aggregator and ":" in stripped:
            left, right = stripped.split(":", 1)
            left_lower = left.strip().lower()
            # If the left side is a vendor name (not a hermes provider),
            # convert colon to slash for the aggregator slug.
            if left_lower in _OPENROUTER_VENDORS and left_lower not in _KNOWN_PROVIDER_NAMES:
                slug = f"{left.strip()}/{right.strip()}"
                target_provider = current_provider
                new_model = slug
            else:
                target_provider, new_model = parse_model_input(stripped, current_provider)
        else:
            # ── Step 3: Standard parse ──
            target_provider, new_model = parse_model_input(stripped, current_provider)

    if not new_model:
        return ModelSwitchResult(
            success=False,
            error_message="No model name provided. Usage: /model <name> or /model provider:model",
        )

    # ── Step 4: Aggregator-aware resolution ──
    # When on OpenRouter/Nous and parse didn't switch providers,
    # try to resolve the model within the aggregator BEFORE falling
    # back to detect_provider_for_model (which might switch to a
    # direct provider the user didn't intend).
    _base = current_base_url or ""
    is_custom = current_provider == "custom" or (
        "localhost" in _base or "127.0.0.1" in _base
    )

    if not alias_result and target_provider == current_provider and on_aggregator:
        aggregator_slug = _resolve_on_aggregator(new_model, current_provider)
        if aggregator_slug:
            new_model = aggregator_slug
            # Stay on current aggregator — skip detect_provider_for_model
        else:
            # ── Step 5: Model not found on aggregator → try provider detection ──
            # This intentionally switches providers (e.g. user typed a model
            # that only exists on a direct provider).
            detected = detect_provider_for_model(new_model, current_provider)
            if detected:
                target_provider, new_model = detected
    elif not alias_result and target_provider == current_provider and not is_custom:
        # Non-aggregator, non-custom: standard detection
        detected = detect_provider_for_model(new_model, current_provider)
        if detected:
            target_provider, new_model = detected

    provider_changed = target_provider != current_provider

    # ── Step 6: Resolve credentials ──
    api_key = current_api_key
    base_url = current_base_url
    api_mode = ""

    try:
        runtime = resolve_runtime_provider(requested=target_provider)
        api_key = runtime.get("api_key", "")
        base_url = runtime.get("base_url", "")
        api_mode = runtime.get("api_mode", "")
    except Exception as e:
        provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
        error_str = str(e)

        if target_provider == "custom":
            return ModelSwitchResult(
                success=False,
                target_provider=target_provider,
                error_message=(
                    "No custom endpoint configured.\n"
                    "Set model.base_url in config.yaml, or set OPENAI_BASE_URL "
                    "in .env, or run: hermes setup"
                ),
            )

        return ModelSwitchResult(
            success=False,
            target_provider=target_provider,
            error_message=(
                f"No credentials for {provider_label}.\n"
                f"Run `hermes setup` to configure it, or use a provider:model "
                f"prefix to target a specific provider.\n"
                f"Detail: {error_str}"
            ),
        )

    # ── Step 7: Catalog validation ──
    known_models: list[str] = []
    if target_provider in {"openrouter", "nous"}:
        known_models = [m for m, _ in OPENROUTER_MODELS]
    elif target_provider in _PROVIDER_MODELS:
        known_models = list(_PROVIDER_MODELS[target_provider])

    model_lower = new_model.lower()
    found_in_catalog = any(m.lower() == model_lower for m in known_models)

    warning_message = ""
    if not found_in_catalog and known_models:
        close = get_close_matches(
            model_lower,
            [m.lower() for m in known_models],
            n=3,
            cutoff=0.5,
        )
        if close:
            lower_to_orig = {m.lower(): m for m in known_models}
            suggestions = [lower_to_orig.get(c, c) for c in close]
            sug_str = ", ".join(f"`{s}`" for s in suggestions)
            warning_message = f"Not in catalog — did you mean: {sug_str}?"
        else:
            warning_message = (
                f"`{new_model}` not in {target_provider} catalog — "
                f"sending as-is (provider may accept it)."
            )
    elif not found_in_catalog and not known_models:
        warning_message = (
            f"No catalog for {target_provider} — "
            f"accepting `{new_model}` as-is."
        )

    # ── Step 8: Build result ──
    provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
    is_custom_target = target_provider == "custom" or (
        base_url
        and "openrouter.ai" not in (base_url or "")
        and ("localhost" in (base_url or "") or "127.0.0.1" in (base_url or ""))
    )

    if target_provider in {"opencode-zen", "opencode-go"}:
        api_mode = opencode_model_api_mode(target_provider, new_model)

    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider=target_provider,
        provider_changed=provider_changed,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
        persist=True,
        warning_message=warning_message,
        is_custom_target=is_custom_target,
        provider_label=provider_label,
        resolved_via_alias=resolved_alias,
    )


def switch_to_custom_provider() -> CustomAutoResult:
    """Handle bare '/model custom' — resolve endpoint and auto-detect model.

    Returns a result object; the caller handles persistence and output.
    """
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        _auto_detect_local_model,
    )

    try:
        runtime = resolve_runtime_provider(requested="custom")
    except Exception as e:
        return CustomAutoResult(
            success=False,
            error_message=(
                f"No custom endpoint configured.\n"
                f"Set model.base_url in config.yaml or OPENAI_BASE_URL in .env.\n"
                f"Detail: {e}"
            ),
        )

    cust_base = runtime.get("base_url", "")
    cust_key = runtime.get("api_key", "")

    if not cust_base or "openrouter.ai" in cust_base:
        return CustomAutoResult(
            success=False,
            error_message=(
                "No custom endpoint configured.\n"
                "Set model.base_url in config.yaml, or set OPENAI_BASE_URL "
                "in .env, or run: hermes setup"
            ),
        )

    detected_model = _auto_detect_local_model(cust_base)
    if not detected_model:
        return CustomAutoResult(
            success=False,
            base_url=cust_base,
            api_key=cust_key,
            error_message=(
                f"Custom endpoint at {cust_base} is reachable but no model "
                f"was auto-detected.\n"
                f"Specify explicitly: /model custom:<model-name>"
            ),
        )

    return CustomAutoResult(
        success=True,
        model=detected_model,
        base_url=cust_base,
        api_key=cust_key,
    )
