# LLM Prompting Layer Grounding Refactoring - Implementation Summary

## Changes Made

### File: `AI/query.py` (Lines 87-187)

#### FACT_PROMPT Refactoring

**Before:** Generic "extract the exact answer" with basic examples
**After:** Strict grounding enforcement with:

✅ **ROLE statement** - Clear identity as legal fact extractor
✅ **6 MANDATORY CONSTRAINTS** - Explicit rules against:
   - Inference and interpretation
   - External legal knowledge
   - Paraphrasing and expansion
   - Anything not in context

✅ **POSITIVE EXAMPLES** - Show correct grounding with direct quotes in [brackets]
   - Query about Act name → quote the context, then answer
   - Query about founder count → quote the context, then answer

✅ **NEGATIVE EXAMPLES** - Show common hallucinations to avoid:
   - External knowledge (e.g., comparing to Indian law)
   - Inference beyond explicit statements (e.g., about qualifications)
   - Required "Not found in provided context" response

✅ **GROUNDING PROCESS** - Step-by-step verification:
   1. Read question carefully
   2. Search for exact answer
   3. Quote context, then provide answer
   4. Or respond "Not found in provided context"
   5. Self-verify against constraints

#### EXPLANATION_PROMPT Refactoring

**Before:** Basic instruction to avoid external knowledge
**After:** Comprehensive grounding enforcement with:

✅ **ROLE statement** - "Extract ONLY from provided context. Zero external knowledge. Zero interpretation."
✅ **6 MANDATORY CONSTRAINTS** - Stricter language:
   - "MUST be explicitly stated"
   - "Do NOT infer relationships"
   - "Do NOT apply general knowledge"
   - "Do NOT paraphrase"
   - "Do NOT interpret vague language"

✅ **POSITIVE EXAMPLES** - Show proper grounding:
   - Each point includes exact quote in [brackets]
   - Minimal interpretation, maximum fidelity to source
   - Clear citation format

✅ **NEGATIVE EXAMPLES** - Show anti-patterns:
   - Using external knowledge
   - Making inferences from separate statements
   - Paraphrasing beyond what's explicit

✅ **GROUNDING INSTRUCTIONS** - Granular rules:
   - Show exact quote in [brackets]
   - Extract directly, no interpretation
   - Do NOT combine quotes to infer
   - Stop if cannot ground

✅ **MANDATORY REFUSAL** - "Not found in provided context" when insufficient

## Design Principles Applied

### Constraint Hierarchy
1. **Mandatory refusal** - Know when to say "Not found"
2. **Citation requirements** - Show source for every fact
3. **No-paraphrasing rule** - Use exact context text
4. **No-inference rule** - Don't connect dots
5. **No-external-knowledge rule** - Only context facts

### Anti-Hallucination Measures
- Examples show WRONG answers to highlight hallucinations
- Explicit rules against inference, interpretation, external knowledge
- Mandatory "Not found in provided context" response pattern
- Citation format enforces traceability
- Self-verification step in grounding process

### Domain Optimization
- Legal terminology (Nepal Company Act 2063)
- Nepali language support (examples in Nepali)
- Legal QA patterns (facts + procedures)
- Bilingual response support

## What Remains Unchanged

✅ `query_rag()` function logic - No changes needed
✅ Retrieval pipeline - Hybrid search, reranking, filtering all unchanged
✅ LLM model - Still using Mistral via Ollama
✅ Mode detection - Fact vs Explanation logic unchanged
✅ Context building - Still building from retrieved chunks
✅ Backend integration - FastAPI integration unaffected
✅ Date normalization - Post-processing unchanged

## Impact on LLM Behavior

### Stricter Extraction
- More refusals ("Not found in provided context")
- Lower hallucination rate
- Higher precision (only grounded answers)
- Potentially lower recall (may refuse partial answers)

### Better Traceability
- Every claim citable to context
- Citation format shows exact source text
- Verifiable grounding chain
- Audit trail for legal compliance

### Domain-Specific Accuracy
- Legal terminology respected
- Nepali language properly handled
- No misapplication of general knowledge
- Act-specific constraints enforced

## Testing Recommendations

### Test Cases to Verify Grounding

1. **Explicit Answer Case**
   ```
   Query: "What is the name of this Act?"
   Context: "यस ऐनको नाम 'कम्पनी ऐन, २०६३' रहेको छ।"
   Expected: Should quote context and provide fact
   ```

2. **Inference Case**
   ```
   Query: "What qualifications must shareholders have?"
   Context: Only says shareholder count requirements
   Expected: Should refuse with "Not found in provided context"
   ```

3. **External Knowledge Case**
   ```
   Query: "How does Nepal Company Act compare to Indian law?"
   Context: Only mentions Nepal Act
   Expected: Should refuse with "Not found in provided context"
   ```

4. **Partial Information Case**
   ```
   Query: "Explain the complete dissolution process"
   Context: Only partial process described
   Expected: Should quote what's available or refuse
   ```

5. **Explanation Mode Case**
   ```
   Query: "How are directors appointed?"
   Context: Multiple procedure steps mentioned
   Expected: Should quote each step with citations
   ```

## Verification Checklist

- [ ] No paraphrasing in responses (direct quotes used)
- [ ] All numerical claims traced to context
- [ ] Refusal responses match exact pattern
- [ ] No implicit inferences in explanations
- [ ] Citations are accurate and specific
- [ ] Backend integration still works
- [ ] Response format follows examples

## Future Enhancements

Potential improvements not implemented:
- Confidence scoring for grounded answers
- Automated hallucination detection
- Fact verification against multiple chunks
- Citation hyperlinks to source chunks
- Grounding quality metrics
