#!/usr/bin/env python3
"""
Fix Subtype Naming Consistency Script
Ensures all molecular subtypes use consistent naming: canonical, immune, stromal
"""

import os
import re
from pathlib import Path

def fix_subtype_naming():
    """Fix all subtype naming inconsistencies"""
    
    print("🔧 Fixing molecular subtype naming consistency...")
    
    # Define the standardized names (lowercase)
    standard_names = {
        'canonical': 'canonical',
        'immune': 'immune', 
        'stromal': 'stromal'
    }
    
    # Patterns to replace (case variations)
    patterns_to_fix = [
        # SNF patterns
        (r'\bSNF1\b', 'canonical'),
        (r'\bSNF2\b', 'immune'),
        (r'\bSNF3\b', 'stromal'),
        (r'\bSNF1 \(Canonical\)\b', 'canonical'),
        (r'\bSNF2 \(Immune\)\b', 'immune'),
        (r'\bSNF3 \(Stromal\)\b', 'stromal'),
        
        # Capitalization patterns (in specific contexts)
        (r"'canonical':", "'canonical':"),
        (r"'immune':", "'immune':"),
        (r"'stromal':", "'stromal':"),
        (r'"canonical":', '"canonical":'),
        (r'"immune":', '"immune":'),
        (r'"stromal":', '"stromal":'),
        
        # List patterns
        (r"\['Canonical', 'Immune', 'Stromal'\]", "['canonical', 'immune', 'stromal']"),
        (r'\["Canonical", "Immune", "Stromal"\]', '["canonical", "immune", "stromal"]'),
        
        # Variable assignments
        (r"subtype.*=.*['\"]Canonical['\"]", "subtype = 'canonical'"),
        (r"subtype.*=.*['\"]Immune['\"]", "subtype = 'immune'"),
        (r"subtype.*=.*['\"]Stromal['\"]", "subtype = 'stromal'"),
    ]
    
    # Files to update (focus on Python files)
    files_to_update = []
    
    # Get all Python files in app/ and scripts/
    for directory in ['app', 'scripts']:
        if Path(directory).exists():
            for py_file in Path(directory).glob("*.py"):
                files_to_update.append(py_file)
    
    # Also update markdown files for documentation consistency
    for md_file in Path('.').glob("*.md"):
        files_to_update.append(md_file)
    
    updated_files = []
    
    for file_path in files_to_update:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply pattern replacements
            for pattern, replacement in patterns_to_fix:
                content = re.sub(pattern, replacement, content)
            
            # Special case: Update display names in UI to use proper case
            # Keep display names as "Canonical", "Immune", "Stromal" for UI
            if 'crc_unified_platform.py' in str(file_path):
                # In UI display contexts, keep proper case
                content = re.sub(r'f"canonical Subtype"', 'f"Canonical Subtype"', content)
                content = re.sub(r'f"immune Subtype"', 'f"Immune Subtype"', content) 
                content = re.sub(r'f"stromal Subtype"', 'f"Stromal Subtype"', content)
                
                # Fix subtype display names
                content = re.sub(r"'canonical'.*?'name': 'canonical'", "'canonical': {\n            'name': 'Canonical'", content)
                content = re.sub(r"'immune'.*?'name': 'immune'", "'immune': {\n            'name': 'Immune'", content)
                content = re.sub(r"'stromal'.*?'name': 'stromal'", "'stromal': {\n            'name': 'Stromal'", content)
            
            # Save if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_files.append(file_path)
                print(f"  ✓ Updated {file_path}")
                
        except Exception as e:
            print(f"  ❌ Error updating {file_path}: {e}")
    
    if updated_files:
        print(f"\n✅ Updated {len(updated_files)} files for subtype naming consistency")
        print("\n📋 Summary of changes:")
        print("  • canonical/immune/stromal → canonical/immune/stromal")
        print("  • Fixed case sensitivity issues")
        print("  • Standardized internal naming to lowercase")
        print("  • Maintained proper case for UI display")
    else:
        print("✅ No files needed updating - naming already consistent")

def create_subtype_mapping_reference():
    """Create a reference file for subtype mappings"""
    
    reference_content = """# Molecular Subtype Naming Reference

## Standardized Names (Internal Use)
- canonical (lowercase)
- immune (lowercase)
- stromal (lowercase)

## Display Names (UI)
- Canonical (proper case)
- Immune (proper case)  
- Stromal (proper case)

## Deprecated Names (Do Not Use)
- canonical, immune, stromal
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
"""
    
    with open('SUBTYPE_NAMING_REFERENCE.md', 'w') as f:
        f.write(reference_content)
    
    print("📚 Created SUBTYPE_NAMING_REFERENCE.md")

if __name__ == "__main__":
    print("🎯 Molecular Subtype Naming Consistency Fix")
    print("=" * 50)
    
    # Fix naming issues
    fix_subtype_naming()
    
    # Create reference documentation
    create_subtype_mapping_reference()
    
    print("\n🎉 Subtype naming consistency completed!")
    print("\nNow the platform will handle both:")
    print("  • Internal processing: canonical/immune/stromal")
    print("  • UI display: Canonical/Immune/Stromal")
    print("  • Case-insensitive input handling") 