# Molecular Subtype Naming Reference

## Standardized Names (Internal Use)
- canonical (lowercase)
- immune (lowercase)
- stromal (lowercase)

## Display Names (UI)
- Canonical (proper case)
- Immune (proper case)  
- Stromal (proper case)

## Deprecated Names (Do Not Use)
- SNF1, SNF2, SNF3
- CNS1, CNS2, CNS3, CNS4
- CMS1, CMS2, CMS3, CMS4

## Code Examples

### Correct Internal Usage:
```python
subtypes = ['canonical', 'immune', 'stromal']
if predicted_subtype.lower() == 'canonical':
    # Handle canonical subtype
```

### Correct UI Display:
```python
display_name = {
    'canonical': 'Canonical',
    'immune': 'Immune', 
    'stromal': 'Stromal'
}[subtype.lower()]
```

### Case-Insensitive Lookups:
```python
subtype_key = predicted_subtype.lower()
info = subtype_info.get(subtype_key, default_info)
```
