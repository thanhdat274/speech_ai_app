# Proposal: Speech AI App Upgrade v2

## Context & Problem

Our Speech AI application currently performs well but has room for improvement in:
- **Performance**: Translation latency can be optimized with batch processing
- **Model Efficiency**: No int4 quantization support for low-VRAM GPUs
- **Translation Quality**: NLLB output needs post-processing fixes
- **Feature Completeness**: Missing speaker diarization, export functions, confidence scoring

## Proposed Solutions

### Phase 1: Critical Performance Improvements (P0)

#### 1. Batch Translation for NLLB
**Problem**: Currently translates sentences one-by-one, underutilizing GPU
**Solution**: Accumulate 3-5 sentences and translate in batch
**Impact**: 2-3x faster GPU utilization, lower latency per sentence
**Effort**: Low (2-3 hours)
**Files**: `core/streaming_translator.py`

#### 2. Int4 Quantization Support
**Problem**: GTX 1650/1050 Ti users struggle with int8 models
**Solution**: Add bitsandbytes int4 support for <2GB VRAM GPUs
**Impact**: Enable smooth operation on low-end GPUs
**Effort**: Medium (4-5 hours)
**Files**: `core/ai_engine.py`, `requirements.txt`

### Phase 2: Quality Enhancements (P1)

#### 3. Translation Post-Processor
**Problem**: NLLB outputs have artifacts (extra spaces, punctuation issues)
**Solution**: Add correction layer for common NLLB errors
**Impact**: Cleaner, more professional translations
**Effort**: Low (2 hours)
**Files**: `core/translation_post_processor.py` (new)

#### 4. Enhanced Whisper Prompts
**Problem**: Generic prompts don't adapt to domain content
**Solution**: Dynamic prompt with detected topics and hotwords
**Impact**: Better accuracy for technical/domain-specific content
**Effort**: Medium (3 hours)
**Files**: `core/ai_engine.py`

### Phase 3: New Features (P2)

#### 5. Export Functions
**Problem**: No way to save/download transcripts
**Solution**: Add export to SRT, JSON, VTT formats
**Impact**: Useful for content creators, researchers
**Effort**: Medium (4 hours)
**Files**: `core/transcript_exporter.py` (new)

#### 6. Confidence Scoring
**Problem**: Users can't tell translation quality at a glance
**Solution**: Display confidence score per sentence
**Impact**: Better user trust and error detection
**Effort**: Medium (4 hours)
**Files**: `core/confidence_scorer.py` (new), UI updates

## Scope Exclusions (Out of Scope for v2)

- **Speaker Diarization**: High effort, requires speechbrain dependency
- **Real-time Waveform**: UI-heavy, better as v3 feature
- **Prometheus Metrics**: Advanced monitoring, can be added later
- **Dark/Light Theme**: Low priority UI enhancement

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Translation latency (GPU) | 100-300ms | 50-150ms (batch) |
| Translation latency (CPU) | 200-500ms | 100-300ms |
| VRAM usage (small GPU) | 3-4GB | 2-3GB (int4) |
| Translation quality (manual) | Good | Excellent |
| Export formats | None | SRT, JSON, VTT |

## Rollout Plan

1. **Week 1**: Implement P0 tasks (Batch Translation + Int4)
2. **Week 2**: Implement P1 tasks (Post-Processor + Enhanced Prompts)
3. **Week 3**: Implement P2 tasks (Export + Confidence Scoring)
4. **Week 4**: Testing, bug fixes, documentation

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| bitsandbytes compatibility | Medium | High | Fallback to int8 if int4 fails |
| Batch translation coherence | Low | Medium | Context preservation in batch |
| Post-processor over-correction | Low | Low | Conservative rules, user toggle |
| UI complexity increase | Medium | Low | Keep enhancements optional |

## Dependencies

### New Python Packages (Optional)
```
bitsandbytes>=0.41.0  # For int4 quantization (GPU only)
```

### Existing Dependencies (No Changes)
- PySide6
- faster-whisper
- torch
- transformers (NLLB)
- numpy
- sounddevice

## Approval Required

- [ ] Proceed with P0 tasks immediately
- [ ] Include P1 tasks in same release
- [ ] Defer P2 tasks to v3 release
- [ ] Add Speaker Diarization to scope
