#!/usr/bin/env python3
"""
PDF Report Generator for CRC Analysis Platform
Generates comprehensive analysis reports with statistics and methodology
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor
import datetime
import io
import base64
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

class CRCReportGenerator:
    """Generate professional PDF reports for CRC analysis results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=HexColor('#1a1a2e'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#0f3460'),
            spaceBefore=20,
            spaceAfter=10,
            alignment=TA_LEFT
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=HexColor('#16213e'),
            spaceBefore=15,
            spaceAfter=10,
            borderWidth=1,
            borderColor=HexColor('#00d4ff'),
            borderPadding=5
        ))
        
        # Result emphasis style
        self.styles.add(ParagraphStyle(
            name='ResultEmphasis',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=HexColor('#e74c3c'),
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=5
        ))
        
        # Mathematical formula style
        self.styles.add(ParagraphStyle(
            name='Formula',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Courier',
            textColor=HexColor('#2c3e50'),
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=5,
            leftIndent=20,
            rightIndent=20,
            backColor=HexColor('#ecf0f1')
        ))
    
    def generate_tissue_classification_report(self, results, image_data, timestamp=None):
        """Generate PDF report for tissue classification results"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
        story = []
        
        # Header
        story.append(Paragraph("CRC Tissue Classification Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Report metadata
        if not timestamp:
            timestamp = datetime.datetime.now()
        
        metadata_data = [
            ['Report Generated:', timestamp.strftime("%Y-%m-%d %H:%M:%S")],
            ['Analysis Type:', 'Tissue Classification (8 Classes)'],
            ['Model:', 'ResNet50 (91.4% Test Accuracy)'],
            ['Framework:', 'PyTorch 2.0']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#e8f5e9')),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1b5e20')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Primary Results Section
        story.append(Paragraph("Primary Classification Results", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Main result box
        result_data = [
            ['Primary Tissue Type:', results['primary_class']],
            ['Confidence Score:', f"{results['confidence']:.2f}%"],
            ['Diversity Score:', f"{results.get('diversity_score', 0):.2f}%"]
        ]
        
        result_table = Table(result_data, colWidths=[2.5*inch, 3.5*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#e3f2fd')),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#0d47a1')),
            ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#1976d2')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 1, HexColor('#1976d2')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#90caf9'))
        ]))
        story.append(result_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed Predictions
        story.append(Paragraph("Detailed Tissue Predictions", self.styles['SectionHeader']))
        
        # All tissue probabilities
        tissue_headers = ['Tissue Type', 'Probability (%)', 'Confidence']
        tissue_data = [tissue_headers]
        
        classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
                  'Debris', 'Mucosa', 'Adipose', 'Empty']
        
        for i, (cls, prob) in enumerate(zip(classes, results['probabilities'])):
            confidence_level = 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
            tissue_data.append([cls, f"{prob*100:.2f}", confidence_level])
        
        tissue_table = Table(tissue_data, colWidths=[2*inch, 2*inch, 2*inch])
        tissue_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f5f5f5')])
        ]))
        story.append(tissue_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Statistical Analysis Section
        story.append(Paragraph("Statistical Analysis", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Entropy calculation
        story.append(Paragraph("Tissue Diversity (Shannon Entropy):", self.styles['SectionHeader']))
        story.append(Paragraph("H = -Σ p(x) × log(p(x))", self.styles['Formula']))
        
        entropy = -np.sum(results['probabilities'] * np.log(results['probabilities'] + 1e-8))
        normalized_entropy = entropy / np.log(len(classes))
        
        story.append(Paragraph(f"Raw Entropy: {entropy:.4f}", self.styles['Normal']))
        story.append(Paragraph(f"Normalized Entropy: {normalized_entropy:.4f}", self.styles['Normal']))
        story.append(Paragraph(f"Diversity Score: {normalized_entropy * 100:.2f}%", self.styles['ResultEmphasis']))
        story.append(Spacer(1, 0.2*inch))
        
        # Confidence Metrics
        story.append(Paragraph("Confidence Metrics:", self.styles['SectionHeader']))
        
        # Softmax calculation explanation
        story.append(Paragraph("Softmax Probability Calculation:", self.styles['Normal']))
        story.append(Paragraph("P(class_i) = exp(z_i) / Σ exp(z_j)", self.styles['Formula']))
        
        # Top-k accuracy
        top3_acc = sum([p for p in sorted(results['probabilities'], reverse=True)[:3]]) * 100
        story.append(Paragraph(f"Top-3 Cumulative Probability: {top3_acc:.2f}%", self.styles['Normal']))
        
        # Prediction uncertainty
        max_prob = max(results['probabilities'])
        second_max_prob = sorted(results['probabilities'], reverse=True)[1]
        margin = max_prob - second_max_prob
        story.append(Paragraph(f"Prediction Margin: {margin*100:.2f}%", self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Model Architecture
        story.append(PageBreak())
        story.append(Paragraph("Model Architecture & Methodology", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 0.1*inch))
        
        arch_data = [
            ['Component', 'Details'],
            ['Base Model', 'ResNet50 (Pretrained on ImageNet)'],
            ['Input Size', '224 × 224 × 3 (RGB)'],
            ['Feature Extractor', '2048-dimensional features'],
            ['Classifier Head', 'FC(2048→512) → ReLU → Dropout(0.5) → FC(512→8)'],
            ['Training Data', '100,000 histopathology patches'],
            ['Validation', '5-fold cross-validation'],
            ['Optimizer', 'AdamW (lr=0.001, weight_decay=0.01)'],
            ['Loss Function', 'CrossEntropyLoss with label smoothing']
        ]
        
        arch_table = Table(arch_data, colWidths=[2.5*inch, 3.5*inch])
        arch_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#263238')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#eceff1'), colors.white])
        ]))
        story.append(arch_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Clinical Interpretation
        story.append(Paragraph("Clinical Interpretation Guidelines", self.styles['CustomSubtitle']))
        
        interpretation = self._get_tissue_interpretation(results)
        for line in interpretation:
            story.append(Paragraph(f"• {line}", self.styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Quality Metrics
        story.append(Paragraph("Analysis Quality Metrics", self.styles['SectionHeader']))
        
        quality_data = [
            ['Metric', 'Value', 'Status'],
            ['Model Confidence', f"{results['confidence']:.2f}%", 
             'Excellent' if results['confidence'] > 80 else 'Good' if results['confidence'] > 60 else 'Fair'],
            ['Prediction Margin', f"{margin*100:.2f}%",
             'Strong' if margin > 0.3 else 'Moderate' if margin > 0.1 else 'Weak'],
            ['Tissue Heterogeneity', f"{results.get('diversity_score', 0):.2f}%",
             'High' if results.get('diversity_score', 0) > 70 else 'Moderate' if results.get('diversity_score', 0) > 40 else 'Low']
        ]
        
        quality_table = Table(quality_data, colWidths=[2*inch, 2*inch, 2*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#00796b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#e0f2f1'), colors.white])
        ]))
        story.append(quality_table)
        
        # Disclaimer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Disclaimer", self.styles['Heading4']))
        story.append(Paragraph(
            "This report is generated by an AI system and should be used as a supplementary tool only. "
            "Clinical decisions should be made by qualified healthcare professionals based on comprehensive "
            "patient evaluation and additional diagnostic tests.",
            self.styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_molecular_subtype_report(self, results, image_data, timestamp=None):
        """Generate PDF report for molecular subtype analysis"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
        story = []
        
        # Header
        story.append(Paragraph("CRC Molecular Subtype Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Report metadata
        if not timestamp:
            timestamp = datetime.datetime.now()
        
        metadata_data = [
            ['Report Generated:', timestamp.strftime("%Y-%m-%d %H:%M:%S")],
            ['Analysis Type:', 'Molecular Subtype Classification'],
            ['Reference:', 'Pitroda et al. Nature Communications 2018'],
            ['Subtypes:', 'SNF1 (Canonical), SNF2 (Immune), SNF3 (Stromal)']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#fff3e0')),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#e65100')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Primary Results
        story.append(Paragraph("Molecular Subtype Classification", self.styles['CustomSubtitle']))
        
        # Define subtype colors
        subtype_colors = {
            'SNF1 (Canonical)': HexColor('#e74c3c'),
            'SNF2 (Immune)': HexColor('#27ae60'),
            'SNF3 (Stromal)': HexColor('#e67e22')
        }
        
        # Main result with color coding
        result_data = [
            ['Predicted Subtype:', results['subtype']],
            ['Confidence:', f"{results['confidence']:.2f}%"],
            ['Confidence Level:', results['confidence_metrics']['classification_certainty']],
            ['Risk Category:', results['risk_category']],
            ['10-Year Survival:', results['subtype_info']['survival']]
        ]
        
        result_table = Table(result_data, colWidths=[2.5*inch, 3.5*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f5f5f5')),
            ('TEXTCOLOR', (1, 0), (1, 0), subtype_colors.get(results['subtype'], colors.black)),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 2, subtype_colors.get(results['subtype'], colors.black)),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(result_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Subtype Probabilities
        story.append(Paragraph("Subtype Probability Distribution", self.styles['SectionHeader']))
        
        prob_data = [
            ['Subtype', 'Probability (%)', 'Survival Rate', 'Characteristics']
        ]
        
        for i, (prob, subtype_info) in enumerate(zip(results['probabilities'], results['all_subtypes'])):
            prob_data.append([
                subtype_info['name'],
                f"{prob*100:.2f}",
                subtype_info['survival'],
                subtype_info['features'][:40] + '...'
            ])
        
        prob_table = Table(prob_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 2.1*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#ecf0f1')])
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Tissue Composition Analysis
        story.append(Paragraph("Tissue Composition Analysis", self.styles['CustomSubtitle']))
        
        tissue_data = [
            ['Tissue Type', 'Percentage', 'Clinical Relevance']
        ]
        
        relevance_map = {
            'tumor': 'Primary malignant tissue',
            'stroma': 'Supportive connective tissue',
            'lymphocytes': 'Immune cell infiltration',
            'complex': 'Mixed tissue patterns',
            'debris': 'Necrotic/cellular debris',
            'mucosa': 'Normal epithelial tissue',
            'adipose': 'Fat tissue',
            'empty': 'Background/no tissue'
        }
        
        for tissue, value in results['tissue_composition'].items():
            if value > 0.01:  # Only show significant tissues
                tissue_data.append([
                    tissue.capitalize(),
                    f"{value*100:.1f}%",
                    relevance_map.get(tissue, 'N/A')
                ])
        
        tissue_comp_table = Table(tissue_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
        tissue_comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ecf0f1'), colors.white])
        ]))
        story.append(tissue_comp_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Molecular Signatures
        story.append(Paragraph("Molecular Signature Analysis", self.styles['SectionHeader']))
        
        sig_data = [
            ['Signature', 'Score', 'Interpretation'],
            ['Immune Infiltration', f"{results['molecular_signatures']['immune_infiltration']*100:.1f}%",
             'High' if results['molecular_signatures']['immune_infiltration'] > 0.3 else 'Moderate' if results['molecular_signatures']['immune_infiltration'] > 0.15 else 'Low'],
            ['Fibrosis Level', f"{results['molecular_signatures']['fibrosis_level']*100:.1f}%",
             'Marked' if results['molecular_signatures']['fibrosis_level'] > 0.3 else 'Moderate' if results['molecular_signatures']['fibrosis_level'] > 0.15 else 'Minimal'],
            ['Tumor Architecture', f"{results['molecular_signatures']['tumor_architecture']*100:.1f}%",
             'Solid nests' if results['molecular_signatures']['tumor_architecture'] > 0.3 else 'Mixed' if results['molecular_signatures']['tumor_architecture'] > 0.15 else 'Dispersed'],
            ['Tissue Organization', f"{results['molecular_signatures']['tissue_organization']*100:.1f}%",
             'Well-organized' if results['molecular_signatures']['tissue_organization'] > 0.5 else 'Moderate' if results['molecular_signatures']['tissue_organization'] > 0.25 else 'Disorganized'],
            ['Architectural Complexity', f"{results['molecular_signatures']['architectural_complexity']*100:.1f}%",
             'Complex' if results['molecular_signatures']['architectural_complexity'] > 0.5 else 'Moderate' if results['molecular_signatures']['architectural_complexity'] > 0.25 else 'Simple']
        ]
        
        sig_table = Table(sig_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#8e44ad')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f4ecf7'), colors.white])
        ]))
        story.append(sig_table)
        story.append(PageBreak())
        
        # Mathematical Methods
        story.append(Paragraph("Mathematical Methods", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Subtype scoring formulas
        story.append(Paragraph("Molecular Subtype Scoring Algorithm:", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.05*inch))
        
        story.append(Paragraph("SNF1 (Canonical) Score:", self.styles['Normal']))
        story.append(Paragraph(
            "Score = (Tumor×1.5 + (1-Immune)×1.2 + (1-Stromal)×0.8 + Organization×0.5 + (1-Lymph_density)×0.8) / 5.3",
            self.styles['Formula']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("SNF2 (Immune) Score:", self.styles['Normal']))
        story.append(Paragraph(
            "Score = (Immune×2.0 + Lymph_density×1.5 + (1-Stromal)×1.0 + (1-Fibrotic)×0.8 + Mucosa×0.5) / 5.8",
            self.styles['Formula']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("SNF3 (Stromal) Score:", self.styles['Normal']))
        story.append(Paragraph(
            "Score = (Stromal×1.8 + Fibrotic×1.5 + Fibrosis×1.2 + Complexity×0.8 + (1-Immune)×0.7) / 6.0",
            self.styles['Formula']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Confidence calculation
        story.append(Paragraph("Confidence Metrics:", self.styles['SectionHeader']))
        story.append(Paragraph("Signature Strength = max(scores) - mean(scores)", self.styles['Formula']))
        story.append(Paragraph("Adjusted Confidence = raw_confidence × (0.7 + 0.3 × signature_strength)", self.styles['Formula']))
        story.append(Spacer(1, 0.3*inch))
        
        # Clinical Recommendations
        story.append(Paragraph("Clinical Recommendations", self.styles['CustomSubtitle']))
        
        # Therapeutic strategy
        therapy_data = [
            ['Category', 'Recommendation'],
            ['Primary Therapy', results['subtype_info']['therapeutic']],
            ['Key Mutations', results['subtype_info']['mutations']],
            ['Prognostic Features', results['subtype_info']['features']],
            ['Risk Stratification', results['risk_category']]
        ]
        
        therapy_table = Table(therapy_data, colWidths=[2*inch, 4*inch])
        therapy_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#e8f5e9')),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1b5e20')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(therapy_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Follow-up recommendations
        story.append(Paragraph("Follow-up Recommendations:", self.styles['SectionHeader']))
        
        follow_up = self._get_subtype_followup(results['subtype_idx'])
        for rec in follow_up:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Quality Assessment
        story.append(Paragraph("Analysis Quality Assessment", self.styles['SectionHeader']))
        
        quality_metrics = [
            ['Metric', 'Value', 'Assessment'],
            ['Classification Certainty', results['confidence_metrics']['classification_certainty'],
             'Reliable' if results['confidence_metrics']['classification_certainty'] == 'High' else 'Consider validation'],
            ['Signature Strength', f"{results['confidence_metrics']['signature_strength']:.3f}",
             'Strong' if results['confidence_metrics']['signature_strength'] > 0.3 else 'Moderate' if results['confidence_metrics']['signature_strength'] > 0.15 else 'Weak'],
            ['Model Confidence', f"{results['confidence']:.1f}%",
             'High' if results['confidence'] > 70 else 'Moderate' if results['confidence'] > 50 else 'Low']
        ]
        
        quality_table = Table(quality_metrics, colWidths=[2*inch, 2*inch, 2*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#00acc1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#e0f7fa'), colors.white])
        ]))
        story.append(quality_table)
        
        # Confidence Reasons - NEW SECTION
        if 'confidence_reasons' in results['confidence_metrics'] and results['confidence_metrics']['confidence_reasons']:
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Classification Confidence Factors", self.styles['SectionHeader']))
            story.append(Paragraph("The following features contributed to the high confidence prediction:", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            for reason in results['confidence_metrics']['confidence_reasons']:
                story.append(Paragraph(f"• {reason}", self.styles['Normal']))
            
            story.append(Spacer(1, 0.2*inch))
        
        # References
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("References", self.styles['Heading4']))
        story.append(Paragraph(
            "1. Pitroda SP, et al. Integrated molecular subtyping defines a curable oligometastatic state "
            "in colorectal liver metastasis. Nature Communications. 2018;9(1):1793.",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            "2. CRC Unified Analysis Platform. AI-powered tissue classification and molecular subtyping. 2024.",
            self.styles['Normal']
        ))
        
        # Disclaimer
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Disclaimer", self.styles['Heading4']))
        story.append(Paragraph(
            "This molecular subtyping analysis is based on histopathological patterns and should be validated "
            "with molecular profiling (RNA-seq/miRNA) when available. Clinical decisions should incorporate "
            "comprehensive patient data and multidisciplinary consultation.",
            self.styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _get_tissue_interpretation(self, results):
        """Generate clinical interpretation for tissue classification"""
        interpretations = []
        
        primary = results['primary_class']
        confidence = results['confidence']
        
        if primary == 'Tumor':
            interpretations.append("Malignant tissue identified with characteristic cellular patterns")
            interpretations.append("Consider correlation with clinical staging and treatment planning")
        elif primary == 'Stroma':
            interpretations.append("Predominant stromal tissue suggesting desmoplastic reaction")
            interpretations.append("May indicate tumor-associated fibrosis or reactive changes")
        elif primary == 'Lymphocytes':
            interpretations.append("Significant lymphocytic infiltration detected")
            interpretations.append("Consider immunohistochemistry for lymphocyte subtyping")
        elif primary == 'Complex':
            interpretations.append("Mixed tissue patterns requiring careful evaluation")
            interpretations.append("May benefit from additional sectioning or staining")
        
        if confidence < 60:
            interpretations.append("Lower confidence suggests ambiguous tissue patterns")
            interpretations.append("Recommend pathologist review for definitive diagnosis")
        
        return interpretations
    
    def _get_subtype_followup(self, subtype_idx):
        """Generate follow-up recommendations based on subtype"""
        recommendations = {
            0: [  # SNF1
                "Consider genomic profiling for NOTCH1 and PIK3C2B mutations",
                "Evaluate for DNA damage response inhibitor eligibility",
                "Monitor for VEGFA amplification status",
                "Regular imaging surveillance every 3-4 months"
            ],
            1: [  # SNF2
                "Priority evaluation for surgical resection",
                "Assess MSI/MMR status for immunotherapy options",
                "Consider oligometastatic treatment protocols",
                "Extended surveillance intervals if complete resection achieved"
            ],
            2: [  # SNF3
                "Immediate consideration for anti-VEGF therapy",
                "Evaluate for clinical trials with anti-fibrotic agents",
                "Intensive systemic therapy planning required",
                "Close monitoring with imaging every 2-3 months"
            ]
        }
        
        return recommendations.get(subtype_idx, [])
    
    def create_visualization_figure(self, results, analysis_type):
        """Create a matplotlib figure for embedding in the report"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        if analysis_type == 'tissue':
            # Tissue distribution bar chart
            ax1 = axes[0, 0]
            classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
                      'Debris', 'Mucosa', 'Adipose', 'Empty']
            probs = results['probabilities'] * 100
            colors_list = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            bars = ax1.bar(range(len(classes)), probs, color=colors_list)
            ax1.set_xticks(range(len(classes)))
            ax1.set_xticklabels(classes, rotation=45, ha='right')
            ax1.set_ylabel('Probability (%)')
            ax1.set_title('Tissue Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Confidence gauge
            ax2 = axes[0, 1]
            confidence = results['confidence']
            wedges, texts = ax2.pie([confidence, 100-confidence], 
                                    colors=['#2ecc71', '#ecf0f1'],
                                    startangle=90,
                                    counterclock=False)
            ax2.text(0, 0, f'{confidence:.1f}%', ha='center', va='center', 
                    fontsize=20, fontweight='bold')
            ax2.set_title('Classification Confidence')
            
            # Top 3 predictions
            ax3 = axes[1, 0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_classes = [classes[i] for i in top3_idx]
            top3_probs = [probs[i] for i in top3_idx]
            ax3.barh(range(3), top3_probs, color=['#e74c3c', '#3498db', '#f39c12'])
            ax3.set_yticks(range(3))
            ax3.set_yticklabels([f'{i+1}. {cls}' for i, cls in enumerate(top3_classes)])
            ax3.set_xlabel('Probability (%)')
            ax3.set_title('Top 3 Predictions')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Diversity score visualization
            ax4 = axes[1, 1]
            diversity = results.get('diversity_score', 0)
            ax4.text(0.5, 0.5, f'Tissue\nDiversity\n\n{diversity:.1f}%', 
                    ha='center', va='center', fontsize=16,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Heterogeneity Score')
            
        else:  # molecular subtype
            # Subtype probabilities
            ax1 = axes[0, 0]
            subtypes = ['SNF1', 'SNF2', 'SNF3']
            probs = results['probabilities'] * 100
            colors_sub = ['#e74c3c', '#27ae60', '#e67e22']
            bars = ax1.bar(subtypes, probs, color=colors_sub)
            ax1.set_ylabel('Probability (%)')
            ax1.set_title('Molecular Subtype Distribution')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Survival rates
            ax2 = axes[0, 1]
            survivals = [37, 64, 20]
            ax2.bar(subtypes, survivals, color=colors_sub, alpha=0.7)
            ax2.set_ylabel('10-Year Survival (%)')
            ax2.set_title('Survival Rates by Subtype')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Molecular signatures radar
            ax3 = axes[1, 0]
            signatures = results['molecular_signatures']
            sig_names = list(signatures.keys())
            sig_values = list(signatures.values())
            angles = np.linspace(0, 2*np.pi, len(sig_names), endpoint=False).tolist()
            sig_values += sig_values[:1]
            angles += angles[:1]
            ax3 = plt.subplot(2, 2, 3, projection='polar')
            ax3.plot(angles, sig_values, 'o-', linewidth=2, color='#3498db')
            ax3.fill(angles, sig_values, alpha=0.25, color='#3498db')
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels([s.replace('_', '\n') for s in sig_names], size=8)
            ax3.set_ylim(0, 1)
            ax3.set_title('Molecular Signatures', pad=20)
            ax3.grid(True)
            
            # Risk assessment
            ax4 = axes[1, 1]
            risk_colors = {'Low Risk': '#27ae60', 'Intermediate Risk': '#f39c12', 'High Risk': '#e74c3c'}
            risk_level = 'High Risk' if 'High Risk' in results['risk_category'] else \
                        'Low Risk' if 'Low Risk' in results['risk_category'] else 'Intermediate Risk'
            wedge = mpatches.Wedge((0.5, 0.5), 0.4, 0, 360, 
                                  fc=risk_colors.get(risk_level, '#95a5a6'),
                                  ec='black', linewidth=2)
            ax4.add_patch(wedge)
            ax4.text(0.5, 0.5, risk_level, ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Risk Category')
        
        plt.tight_layout()
        
        # Save to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer 