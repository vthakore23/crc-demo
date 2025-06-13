#!/usr/bin/env python3
"""
Create EPOC Data Manifest
Scans WSI directory and creates manifest with clinical metadata
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_wsi_files(wsi_dir: str) -> List[Dict]:
    """Find all WSI files in directory"""
    wsi_extensions = ['.svs', '.ndpi', '.mrxs', '.tiff', '.tif']
    wsi_files = []
    
    wsi_path = Path(wsi_dir)
    if not wsi_path.exists():
        raise ValueError(f"WSI directory does not exist: {wsi_dir}")
    
    for ext in wsi_extensions:
        files = list(wsi_path.rglob(f"*{ext}"))
        for file_path in files:
            # Extract patient ID from filename (adjust pattern as needed)
            patient_id = file_path.stem.split('_')[0]  # Assumes format: PATIENTID_...
            
            wsi_info = {
                'patient_id': patient_id,
                'file_path': str(file_path.absolute()),
                'filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'extension': ext,
                'directory': str(file_path.parent)
            }
            wsi_files.append(wsi_info)
    
    logger.info(f"Found {len(wsi_files)} WSI files")
    return wsi_files


def load_clinical_metadata(metadata_file: Optional[str]) -> Dict:
    """Load clinical metadata from CSV file"""
    if not metadata_file or not Path(metadata_file).exists():
        logger.warning("No clinical metadata file provided or file not found")
        return {}
    
    try:
        df = pd.read_csv(metadata_file)
        
        # Expected columns for EPOC dataset
        expected_columns = [
            'patient_id', 'molecular_subtype', 'institution', 
            'age', 'gender', 'stage', 'msi_status', 'braf_mutation',
            'survival_months', 'vital_status'
        ]
        
        # Check for required columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing clinical metadata columns: {missing_cols}")
        
        # Convert to dictionary indexed by patient_id
        metadata_dict = {}
        for _, row in df.iterrows():
            patient_id = str(row['patient_id'])
            metadata_dict[patient_id] = {
                'molecular_subtype': row.get('molecular_subtype', 'Unknown'),
                'institution': row.get('institution', 'Unknown'),
                'age': int(row.get('age', 0)) if pd.notna(row.get('age')) else None,
                'gender': row.get('gender', 'Unknown'),
                'stage': row.get('stage', 'Unknown'),
                'msi_status': row.get('msi_status', 'Unknown'),
                'braf_mutation': row.get('braf_mutation', 'Unknown'),
                'survival_months': float(row.get('survival_months', 0)) if pd.notna(row.get('survival_months')) else None,
                'vital_status': row.get('vital_status', 'Unknown')
            }
        
        logger.info(f"Loaded clinical metadata for {len(metadata_dict)} patients")
        return metadata_dict
        
    except Exception as e:
        logger.error(f"Error loading clinical metadata: {e}")
        return {}


def validate_molecular_subtypes(manifest_data: Dict) -> Dict:
    """Validate and standardize molecular subtype labels"""
    subtype_mapping = {
        # Canonical subtype variations
        'canonical': 'Canonical',
        'Canonical': 'Canonical',
        'CANONICAL': 'Canonical',
        'canonical_e2f_myc': 'Canonical',
        'e2f_myc': 'Canonical',
        'proliferative': 'Canonical',
        
        # Immune subtype variations
        'immune': 'Immune',
        'Immune': 'Immune',
        'IMMUNE': 'Immune',
        'immune_msi': 'Immune',
        'msi_immune': 'Immune',
        'inflammatory': 'Immune',
        
        # Stromal subtype variations
        'stromal': 'Stromal',
        'Stromal': 'Stromal',
        'STROMAL': 'Stromal',
        'stromal_emt': 'Stromal',
        'emt_stromal': 'Stromal',
        'mesenchymal': 'Stromal',
        
        # Unknown/other
        'unknown': 'Unknown',
        'Unknown': 'Unknown',
        'other': 'Unknown',
        'mixed': 'Unknown'
    }
    
    corrected_count = 0
    for slide in manifest_data['slides']:
        original_subtype = slide['metadata'].get('molecular_subtype', 'Unknown')
        if original_subtype in subtype_mapping:
            standardized = subtype_mapping[original_subtype]
            if standardized != original_subtype:
                slide['metadata']['molecular_subtype'] = standardized
                corrected_count += 1
        else:
            logger.warning(f"Unknown molecular subtype: {original_subtype}")
            slide['metadata']['molecular_subtype'] = 'Unknown'
    
    if corrected_count > 0:
        logger.info(f"Standardized {corrected_count} molecular subtype labels")
    
    return manifest_data


def create_manifest(wsi_dir: str, 
                   output_manifest: str,
                   metadata_file: Optional[str] = None,
                   include_clinical_metadata: bool = True) -> Dict:
    """Create comprehensive EPOC data manifest"""
    
    logger.info("Creating EPOC data manifest...")
    
    # Find all WSI files
    wsi_files = find_wsi_files(wsi_dir)
    
    # Load clinical metadata
    clinical_metadata = {}
    if include_clinical_metadata and metadata_file:
        clinical_metadata = load_clinical_metadata(metadata_file)
    
    # Create manifest structure
    manifest = {
        'dataset': 'EPOC',
        'version': '1.0',
        'created_date': datetime.now().isoformat(),
        'total_slides': len(wsi_files),
        'wsi_directory': str(Path(wsi_dir).absolute()),
        'slides': []
    }
    
    # Process each WSI file
    patients_with_metadata = 0
    patients_without_metadata = 0
    
    for wsi_info in wsi_files:
        patient_id = wsi_info['patient_id']
        
        # Get clinical metadata for this patient
        patient_metadata = clinical_metadata.get(patient_id, {})
        if patient_metadata:
            patients_with_metadata += 1
        else:
            patients_without_metadata += 1
            # Set default values
            patient_metadata = {
                'molecular_subtype': 'Unknown',
                'institution': 'Unknown',
                'age': None,
                'gender': 'Unknown',
                'stage': 'Unknown',
                'msi_status': 'Unknown',
                'braf_mutation': 'Unknown',
                'survival_months': None,
                'vital_status': 'Unknown'
            }
        
        slide_entry = {
            'patient_id': patient_id,
            'path': wsi_info['file_path'],
            'filename': wsi_info['filename'],
            'file_size_bytes': wsi_info['file_size'],
            'format': wsi_info['extension'],
            'metadata': patient_metadata
        }
        
        manifest['slides'].append(slide_entry)
    
    # Validate and standardize molecular subtypes
    manifest = validate_molecular_subtypes(manifest)
    
    # Add summary statistics
    subtype_counts = {}
    institution_counts = {}
    
    for slide in manifest['slides']:
        subtype = slide['metadata']['molecular_subtype']
        institution = slide['metadata']['institution']
        
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        institution_counts[institution] = institution_counts.get(institution, 0) + 1
    
    manifest['summary'] = {
        'patients_with_metadata': patients_with_metadata,
        'patients_without_metadata': patients_without_metadata,
        'molecular_subtype_distribution': subtype_counts,
        'institution_distribution': institution_counts,
        'total_file_size_gb': sum(slide['file_size_bytes'] for slide in manifest['slides']) / (1024**3)
    }
    
    # Save manifest
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to: {output_manifest}")
    logger.info(f"Total slides: {manifest['total_slides']}")
    logger.info(f"Patients with metadata: {patients_with_metadata}")
    logger.info(f"Patients without metadata: {patients_without_metadata}")
    logger.info(f"Molecular subtype distribution: {subtype_counts}")
    logger.info(f"Institution distribution: {institution_counts}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Create EPOC data manifest')
    parser.add_argument('--wsi_dir', required=True, help='Directory containing WSI files')
    parser.add_argument('--output_manifest', required=True, help='Output manifest JSON file')
    parser.add_argument('--metadata_file', help='CSV file with clinical metadata')
    parser.add_argument('--include_clinical_metadata', action='store_true', 
                       help='Include clinical metadata in manifest')
    
    args = parser.parse_args()
    
    try:
        manifest = create_manifest(
            wsi_dir=args.wsi_dir,
            output_manifest=args.output_manifest,
            metadata_file=args.metadata_file,
            include_clinical_metadata=args.include_clinical_metadata
        )
        
        print(f"\n✅ Successfully created EPOC manifest:")
        print(f"   📁 WSI Directory: {args.wsi_dir}")
        print(f"   📄 Manifest File: {args.output_manifest}")
        print(f"   🔬 Total Slides: {manifest['total_slides']}")
        print(f"   👥 Patients with Metadata: {manifest['summary']['patients_with_metadata']}")
        print(f"   📊 Molecular Subtypes: {manifest['summary']['molecular_subtype_distribution']}")
        
    except Exception as e:
        logger.error(f"Failed to create manifest: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 