# Requirements for True Molecular Subtype Prediction

## 🎯 The Current Gap

### What We Have:
- **Histopathological images** → Pattern recognition model
- **Pathological classifications** (adenocarcinoma, polyps, etc.)
- **Arbitrary mappings** to molecular subtypes (unvalidated)
- **97.31% accuracy** for pathological pattern classification

### What We Need:
- **Paired data**: WSI + molecular profiling from same samples
- **Ground truth**: CMS (Consensus Molecular Subtype) classifications
- **Clinical outcomes**: Oligometastatic vs polymetastatic data
- **Treatment response**: Therapy outcomes by subtype

## 🔬 Required Validation Data

### 1. Molecular Profiling Data
For each histopathology sample, we need:
- **RNA sequencing data** for gene expression profiles
- **CMS classification** (CMS1-4 subtypes)
- **Key mutations**: BRAF, KRAS, MSI status
- **Methylation status**: CIMP status

### 2. Clinical Correlation
- **Metastatic patterns**: Number and location of metastases
- **Oligometastatic status**: ≤5 metastases (Pitroda definition)
- **Survival data**: Overall survival, progression-free survival
- **Treatment data**: Chemotherapy, targeted therapy, immunotherapy responses

### 3. Sample Size Requirements
- **Minimum**: 500-1000 cases with complete data
- **Ideal**: 2000+ cases across multiple institutions
- **Balance**: Equal representation of molecular subtypes

## 📊 Validation Study Design

### Phase 1: Correlation Analysis
1. Collect WSI + molecular data pairs
2. Extract morphological features from WSI
3. Correlate features with molecular subtypes
4. Validate assumed mappings (e.g., serrated → immune)

### Phase 2: Model Development
1. Train model with true molecular labels
2. Cross-validation across institutions
3. Test on held-out cohorts
4. Compare to pathologist predictions

### Phase 3: Clinical Validation
1. Prospective testing on new cases
2. Correlation with clinical outcomes
3. Assessment of oligometastatic prediction
4. Integration with treatment decisions

## 🚫 Why Current Mappings May Be Wrong

### Adenocarcinoma → Canonical?
- Adenocarcinomas can be ANY molecular subtype
- Morphology alone cannot determine molecular profile
- Need actual gene expression data

### Serrated Adenoma → Immune?
- Based on general association with MSI
- Not all serrated lesions are MSI-high
- Many exceptions exist

### Polyps → Stromal?
- Oversimplification of diverse pathology
- Polyps can have various molecular profiles
- Stromal content varies widely

## ✅ Proper Validation Metrics

### For Molecular Subtype Prediction:
- **Accuracy**: Against true CMS classification
- **Concordance**: With RNA-seq based typing
- **Clinical correlation**: With known subtype behaviors

### For Oligometastatic Prediction:
- **Sensitivity**: Identifying true oligometastatic cases
- **Specificity**: Excluding polymetastatic cases
- **PPV/NPV**: Clinical decision support metrics

## 🎯 What EPOC Data Should Include

For the model to achieve true molecular subtype prediction:

1. **Mandatory**:
   - WSI images
   - CMS classification or RNA-seq data
   - Basic clinical data

2. **Highly Valuable**:
   - Metastatic pattern data
   - Treatment response
   - Survival outcomes
   - Multi-timepoint data

3. **Nice to Have**:
   - Matched primary-metastasis pairs
   - ctDNA data
   - Radiological imaging
   - Detailed treatment timelines

## 💡 Alternative Approaches

If molecular data is unavailable:

1. **Immunohistochemistry surrogate**:
   - Use IHC markers (CDX2, CK20, etc.)
   - Less accurate than RNA-seq
   - More feasible for routine use

2. **Hybrid approach**:
   - Morphology + limited IHC
   - Computational pathology features
   - Clinical data integration

3. **Transfer learning**:
   - Pre-train on public datasets with molecular data
   - Fine-tune on local cohort
   - Validate on subset with molecular profiling

## 📋 Conclusion

Without proper molecular validation data, we cannot claim:
- Molecular subtype prediction accuracy
- Oligometastatic identification capability
- Clinical utility for treatment selection

The current model is a **morphology classifier** that needs molecular ground truth to become a true **molecular subtype predictor**. 