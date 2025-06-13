# CRITICAL MODEL LIMITATIONS & CLARIFICATIONS

## ⚠️ IMPORTANT: What This Model Actually Does

### The Reality Check

The 97.31% accuracy and 99.72% AUC are for **histopathological pattern classification**, NOT molecular subtype prediction. Here's what actually happened:

1. **Training Data**: EBHI-SEG contains 6 pathological categories:
   - Adenocarcinoma (795 images)
   - High-grade IN (186 images)
   - Low-grade IN (637 images)
   - Normal (76 images)
   - Polyp (474 images)
   - Serrated adenoma (58 images)

2. **What We Did**: We arbitrarily mapped these to molecular subtypes:
   - Adenocarcinoma → Canonical
   - High-grade IN → Canonical
   - Low-grade IN → Stromal
   - Polyp → Stromal
   - Serrated adenoma → Immune
   - Normal → Normal

3. **The Problem**: This mapping is **NOT validated**. We have:
   - ❌ NO molecular profiling data (RNA-seq, genomic)
   - ❌ NO ground truth molecular subtype labels
   - ❌ NO validation against actual CMS classifications
   - ❌ NO oligometastatic outcome data

## 🔍 Honest Assessment

### For Oligometastatic Prediction:
If you upload a WSI from someone with oligometastases, the model:
- **Can**: Identify histopathological patterns (tumor, stroma, etc.)
- **Cannot**: Reliably predict molecular subtype
- **Cannot**: Determine oligometastatic potential
- **Certainty**: Unknown without molecular validation

### The Gap:
```
What We Have:
Histopathology Images → Pattern Recognition → Pathological Categories

What We Need:
Histopathology Images + Molecular Data → Validated Model → True Molecular Subtypes
```

## 📊 Actual Performance Metrics

### Validated Performance:
- **Pathological classification**: 97.31% (adenocarcinoma vs polyp vs normal etc.)
- **Molecular subtype prediction**: UNKNOWN (not validated)

### What Would Be Needed:
1. WSI images WITH corresponding molecular profiling
2. CMS (Consensus Molecular Subtype) classifications
3. Clinical outcomes (oligometastatic vs polymetastatic)
4. Multi-institutional validation

## 🚫 What NOT to Claim

This model should NOT be used to:
- Make clinical decisions about molecular subtypes
- Predict oligometastatic potential
- Guide treatment selection based on molecular classification
- Replace actual molecular profiling

## ✅ What This Model CAN Do

- Classify histopathological patterns with high accuracy
- Serve as a research tool for morphology analysis
- Potentially identify patterns that MIGHT correlate with molecular subtypes
- Provide a foundation for future work WITH molecular validation

## 💡 The Path Forward

To achieve true molecular subtype prediction:

1. **Obtain paired data**: WSI + molecular profiling from same samples
2. **Validate mappings**: Confirm if morphological patterns actually correlate with molecular subtypes
3. **Clinical validation**: Test on oligometastatic cohorts with known outcomes
4. **EPOC integration**: Use EPOC data if it includes molecular ground truth

## 🎯 Bottom Line

**Current Model**: Morphology classifier incorrectly labeled as molecular classifier
**True Capability**: Unknown molecular subtype prediction accuracy
**Clinical Utility**: Research only - NOT for clinical use

---

**Note to Users**: The high accuracy numbers (97.31%, 99.72% AUC) apply ONLY to distinguishing between pathological categories (adenocarcinoma, polyp, etc.), NOT to molecular subtype prediction. True molecular subtype prediction accuracy is unknown without proper validation data. 