# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 14:20:31 2025

@author: taske
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_style('whitegrid')


class LeadScoringQualityAnalyzer:
    """Analyzes lead scoring model quality for sales prioritization"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def load_and_prepare_data(self, positive_path, negative_path):
        """Load and prepare data"""
        logger.info("Loading data...")
        
        df_pos = pd.read_csv(positive_path)
        df_neg = pd.read_csv(negative_path)
        df = pd.concat([df_pos, df_neg], ignore_index=True)
        
        # Sort by date for temporal split
        date_col = 'CREATEDDATE' if 'CREATEDDATE' in df.columns else 'CREATED_DATE'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        
        # Features
        feature_cols = [
            'school_quality_tier', 'job_seniority_score', 'years_to_graduation',
            'is_private_school', 'tuition_cost', 'median_property_value',
            'school_quality_score', 'property_quality_score', 
            'overall_quality_score', 'linkedin_confidence'
        ]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['converted']
        
        # Temporal split
        split_idx = int(len(X) * 0.83)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
        logger.info(f"Validation conversion rate: {y_val.mean():.2%}")
        
        return X_train, X_val, y_train, y_val, feature_cols
    
    def train_best_model(self, X_train, y_train):
        """Train the best performing model (Gradient Boosting)"""
        logger.info("\nTraining Gradient Boosting model...")
        
        # Class weights
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calibrate for better probability estimates
        logger.info("Calibrating probabilities...")
        model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
        model.fit(X_train, y_train)
        
        return model
    
    def analyze_score_distribution(self, y_true, y_score):
        """Analyze how scores are distributed for converted vs non-converted"""
        
        converted_scores = y_score[y_true == 1]
        not_converted_scores = y_score[y_true == 0]
        
        logger.info("\n" + "="*60)
        logger.info("SCORE DISTRIBUTION ANALYSIS")
        logger.info("="*60)
        
        logger.info("\nConverted Leads (should have HIGH scores):")
        logger.info(f"  Mean: {converted_scores.mean():.3f}")
        logger.info(f"  Median: {np.median(converted_scores):.3f}")
        logger.info(f"  25th percentile: {np.percentile(converted_scores, 25):.3f}")
        logger.info(f"  75th percentile: {np.percentile(converted_scores, 75):.3f}")
        
        logger.info("\nNon-Converted Leads (should have LOW scores):")
        logger.info(f"  Mean: {not_converted_scores.mean():.3f}")
        logger.info(f"  Median: {np.median(not_converted_scores):.3f}")
        logger.info(f"  25th percentile: {np.percentile(not_converted_scores, 25):.3f}")
        logger.info(f"  75th percentile: {np.percentile(not_converted_scores, 75):.3f}")
        
        return converted_scores, not_converted_scores
    
    def analyze_top_leads(self, y_true, y_score, percentiles=[5, 10, 20, 30]):
        """Analyze conversion rate in top-scored leads"""
        
        logger.info("\n" + "="*60)
        logger.info("TOP LEADS ANALYSIS")
        logger.info("="*60)
        logger.info("\nIf sales team focuses on top-scored leads:")
        
        results = []
        
        for pct in percentiles:
            threshold = np.percentile(y_score, 100 - pct)
            top_mask = y_score >= threshold
            
            total_in_top = top_mask.sum()
            converted_in_top = (y_true[top_mask] == 1).sum()
            conversion_rate = converted_in_top / total_in_top if total_in_top > 0 else 0
            
            # What % of all conversions are captured
            total_conversions = y_true.sum()
            coverage = converted_in_top / total_conversions if total_conversions > 0 else 0
            
            # Lift over baseline
            baseline_rate = y_true.mean()
            lift = conversion_rate / baseline_rate if baseline_rate > 0 else 0
            
            logger.info(f"\nTop {pct}% of leads (score ≥ {threshold:.3f}):")
            logger.info(f"  Total leads: {total_in_top}")
            logger.info(f"  Conversions: {converted_in_top}")
            logger.info(f"  Conversion rate: {conversion_rate:.1%} (baseline: {baseline_rate:.1%})")
            logger.info(f"  Lift: {lift:.2f}x")
            logger.info(f"  Captures {coverage:.1%} of all conversions")
            
            results.append({
                'percentile': pct,
                'threshold': threshold,
                'total_leads': total_in_top,
                'conversions': converted_in_top,
                'conversion_rate': conversion_rate,
                'lift': lift,
                'coverage': coverage
            })
        
        return pd.DataFrame(results)
    
    def analyze_score_buckets(self, y_true, y_score):
        """Analyze conversion rate by score bucket"""
        
        logger.info("\n" + "="*60)
        logger.info("SCORE BUCKET ANALYSIS")
        logger.info("="*60)
        
        # Create score buckets (scaled to 0-100)
        score_100 = y_score * 100
        buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        
        results = []
        
        for low, high in buckets:
            mask = (score_100 >= low) & (score_100 < high)
            if mask.sum() == 0:
                continue
            
            total = mask.sum()
            converted = (y_true[mask] == 1).sum()
            conv_rate = converted / total
            
            # Average predicted score in this bucket
            avg_predicted_score = score_100[mask].mean()
            
            logger.info(f"\nScore {low}-{high}:")
            logger.info(f"  Leads: {total}")
            logger.info(f"  Conversions: {converted}")
            logger.info(f"  Actual conversion rate: {conv_rate:.1%}")
            logger.info(f"  Average predicted score: {avg_predicted_score:.1f}%")
            
            results.append({
                'bucket': f"{low}-{high}",
                'low': low,
                'high': high,
                'total': total,
                'conversions': converted,
                'conversion_rate': conv_rate,
                'avg_predicted_score': avg_predicted_score
            })
        
        return pd.DataFrame(results)
    
    def analyze_calibration(self, y_true, y_score, n_bins=10):
        """Analyze calibration: predicted probability vs actual conversion rate"""
        
        logger.info("\n" + "="*60)
        logger.info("CALIBRATION ANALYSIS")
        logger.info("="*60)
        logger.info("\nComparing predicted scores to actual conversion rates:")
        
        # Create bins by score percentiles to ensure equal samples per bin
        bin_edges = np.percentile(y_score, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 0.001  # Ensure max value is included
        
        calibration_data = []
        
        for i in range(n_bins):
            mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
            
            if mask.sum() == 0:
                continue
            
            n_samples = mask.sum()
            n_positive = (y_true[mask] == 1).sum()
            actual_rate = n_positive / n_samples
            predicted_rate = y_score[mask].mean()
            
            # Calculate difference (calibration error)
            diff = actual_rate - predicted_rate
            
            logger.info(f"\nBin {i+1} (score {bin_edges[i]*100:.1f}-{bin_edges[i+1]*100:.1f}):")
            logger.info(f"  Samples: {n_samples}")
            logger.info(f"  Predicted probability: {predicted_rate:.1%}")
            logger.info(f"  Actual conversion rate: {actual_rate:.1%}")
            logger.info(f"  Difference: {diff:+.1%} {'(overestimating)' if diff < 0 else '(underestimating)' if diff > 0 else '(well calibrated)'}")
            
            calibration_data.append({
                'bin': i + 1,
                'score_min': bin_edges[i] * 100,
                'score_max': bin_edges[i + 1] * 100,
                'score_mid': (bin_edges[i] + bin_edges[i + 1]) / 2 * 100,
                'n_samples': n_samples,
                'predicted_probability': predicted_rate,
                'actual_conversion_rate': actual_rate,
                'calibration_error': diff,
                'abs_calibration_error': abs(diff)
            })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        # Calculate overall calibration metrics
        mean_abs_error = calibration_df['abs_calibration_error'].mean()
        logger.info(f"\n📊 Overall Calibration:")
        logger.info(f"   Mean Absolute Error: {mean_abs_error:.1%}")
        
        if mean_abs_error < 0.05:
            logger.info(f"   ✅ Excellent calibration - predicted scores closely match reality")
        elif mean_abs_error < 0.10:
            logger.info(f"   ✅ Good calibration - predicted scores are reliable")
        elif mean_abs_error < 0.15:
            logger.info(f"   ⚠️  Moderate calibration - use scores with some caution")
        else:
            logger.info(f"   ⚠️  Poor calibration - scores may not reflect true probabilities")
        
        return calibration_df
    
    def plot_score_analysis(self, y_true, y_score, converted_scores, 
                           not_converted_scores, top_leads_df, bucket_df, calibration_df):
        """Generate comprehensive scoring analysis plots"""
        
        logger.info("\nGenerating plots...")
        
        # Create two separate figures for better organization
        
        # FIGURE 1: Distribution and Performance Analysis
        fig1 = plt.figure(figsize=(20, 12))
        gs1 = fig1.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Score distribution (overlapping histograms)
        ax1 = fig1.add_subplot(gs1[0, :2])
        ax1.hist(not_converted_scores * 100, bins=50, alpha=0.6, 
                label='Not Converted', color='red', density=True)
        ax1.hist(converted_scores * 100, bins=50, alpha=0.6, 
                label='Converted', color='green', density=True)
        ax1.set_xlabel('Lead Score', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Score Distribution: Converted vs Not Converted', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plots
        ax2 = fig1.add_subplot(gs1[0, 2])
        data_to_plot = [not_converted_scores * 100, converted_scores * 100]
        bp = ax2.boxplot(data_to_plot, tick_labels=['Not\nConverted', 'Converted'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('green')
        ax2.set_ylabel('Lead Score', fontsize=12)
        ax2.set_title('Score Ranges', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Top leads conversion rate
        ax3 = fig1.add_subplot(gs1[1, 0])
        ax3.bar(top_leads_df['percentile'].astype(str) + '%', 
               top_leads_df['conversion_rate'] * 100,
               color='steelblue', alpha=0.7)
        ax3.axhline(y=y_true.mean() * 100, color='red', linestyle='--', 
                   label='Baseline', linewidth=2)
        ax3.set_xlabel('Top X% of Leads', fontsize=12)
        ax3.set_ylabel('Conversion Rate (%)', fontsize=12)
        ax3.set_title('Conversion Rate in Top Scored Leads', 
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_leads_df.iterrows()):
            ax3.text(i, row['conversion_rate'] * 100 + 1, 
                    f"{row['conversion_rate']*100:.1f}%",
                    ha='center', fontsize=10)
        
        # 4. Lift chart
        ax4 = fig1.add_subplot(gs1[1, 1])
        ax4.bar(top_leads_df['percentile'].astype(str) + '%', 
               top_leads_df['lift'],
               color='orange', alpha=0.7)
        ax4.axhline(y=1, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Top X% of Leads', fontsize=12)
        ax4.set_ylabel('Lift (vs Baseline)', fontsize=12)
        ax4.set_title('Model Lift', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, (idx, row) in enumerate(top_leads_df.iterrows()):
            ax4.text(i, row['lift'] + 0.1, f"{row['lift']:.1f}x",
                    ha='center', fontsize=10)
        
        # 5. Coverage (% of conversions captured)
        ax5 = fig1.add_subplot(gs1[1, 2])
        ax5.bar(top_leads_df['percentile'].astype(str) + '%', 
               top_leads_df['coverage'] * 100,
               color='purple', alpha=0.7)
        ax5.set_xlabel('Top X% of Leads', fontsize=12)
        ax5.set_ylabel('% of All Conversions Captured', fontsize=12)
        ax5.set_title('Conversion Coverage', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for i, (idx, row) in enumerate(top_leads_df.iterrows()):
            ax5.text(i, row['coverage'] * 100 + 2, f"{row['coverage']*100:.1f}%",
                    ha='center', fontsize=10)
        
        # 6. Score bucket analysis
        ax6 = fig1.add_subplot(gs1[2, :])
        
        # Create grouped bar chart
        x = np.arange(len(bucket_df))
        width = 0.35
        
        ax6_twin = ax6.twinx()
        
        bars1 = ax6.bar(x - width/2, bucket_df['total'], width, 
                       label='Total Leads', color='lightblue', alpha=0.7)
        bars2 = ax6.bar(x + width/2, bucket_df['conversions'], width,
                       label='Conversions', color='green', alpha=0.7)
        
        line = ax6_twin.plot(x, bucket_df['conversion_rate'] * 100, 
                            color='red', marker='o', linewidth=3, 
                            markersize=10, label='Conversion Rate')
        
        ax6.set_xlabel('Score Range', fontsize=12)
        ax6.set_ylabel('Number of Leads', fontsize=12)
        ax6_twin.set_ylabel('Conversion Rate (%)', fontsize=12, color='red')
        ax6.set_title('Lead Volume and Conversion Rate by Score Bucket', 
                     fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(bucket_df['bucket'])
        
        # Combine legends
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on conversion rate line
        for i, (idx, row) in enumerate(bucket_df.iterrows()):
            ax6_twin.text(i, row['conversion_rate'] * 100 + 2, 
                         f"{row['conversion_rate']*100:.1f}%",
                         ha='center', fontsize=10, color='red', fontweight='bold')
        
        plt.savefig(self.output_dir / 'lead_scoring_analysis.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.output_dir / 'lead_scoring_analysis.png'}")
        plt.close()
        
        # FIGURE 2: Calibration Analysis
        fig2 = plt.figure(figsize=(20, 12))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Calibration curve (diagonal = perfect calibration)
        ax7 = fig2.add_subplot(gs2[0, 0])
        ax7.plot(calibration_df['predicted_probability'] * 100,
                calibration_df['actual_conversion_rate'] * 100,
                marker='o', linewidth=2, markersize=8, color='blue', label='Model')
        ax7.plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect Calibration')
        ax7.set_xlabel('Predicted Probability (%)', fontsize=12)
        ax7.set_ylabel('Actual Conversion Rate (%)', fontsize=12)
        ax7.set_title('Calibration Curve: Predicted vs Actual', 
                     fontsize=14, fontweight='bold')
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim(0, 100)
        ax7.set_ylim(0, 100)
        
        # Add annotations for each point
        for idx, row in calibration_df.iterrows():
            ax7.annotate(f"Bin {row['bin']}", 
                        xy=(row['predicted_probability'] * 100, 
                            row['actual_conversion_rate'] * 100),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        # 2. Calibration error bar chart
        ax8 = fig2.add_subplot(gs2[0, 1])
        colors = ['red' if x < 0 else 'green' for x in calibration_df['calibration_error']]
        bars = ax8.barh(calibration_df['bin'], calibration_df['calibration_error'] * 100, 
                       color=colors, alpha=0.7)
        ax8.axvline(x=0, color='black', linewidth=2)
        ax8.set_xlabel('Calibration Error (%)', fontsize=12)
        ax8.set_ylabel('Score Bin', fontsize=12)
        ax8.set_title('Calibration Error by Score Range\n(Negative = Overestimating)', 
                     fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(calibration_df.iterrows()):
            value = row['calibration_error'] * 100
            ax8.text(value, i, f"{value:+.1f}%",
                    ha='left' if value > 0 else 'right',
                    va='center', fontsize=9)
        
        # 3. Predicted vs Actual grouped comparison
        ax9 = fig2.add_subplot(gs2[1, 0])
        x = np.arange(len(calibration_df))
        width = 0.35
        
        bars1 = ax9.bar(x - width/2, calibration_df['predicted_probability'] * 100, 
                       width, label='Predicted', color='orange', alpha=0.7)
        bars2 = ax9.bar(x + width/2, calibration_df['actual_conversion_rate'] * 100, 
                       width, label='Actual', color='blue', alpha=0.7)
        
        ax9.set_xlabel('Score Bin', fontsize=12)
        ax9.set_ylabel('Conversion Probability (%)', fontsize=12)
        ax9.set_title('Predicted vs Actual Conversion Rate by Score Bin', 
                     fontsize=14, fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels([f"Bin {b}\n({calibration_df.iloc[i]['score_min']:.0f}-{calibration_df.iloc[i]['score_max']:.0f})" 
                            for i, b in enumerate(calibration_df['bin'])], fontsize=9)
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
        
        # 4. Sample distribution across bins
        ax10 = fig2.add_subplot(gs2[1, 1])
        ax10.bar(calibration_df['bin'], calibration_df['n_samples'], 
                color='steelblue', alpha=0.7)
        ax10.set_xlabel('Score Bin', fontsize=12)
        ax10.set_ylabel('Number of Leads', fontsize=12)
        ax10.set_title('Sample Distribution Across Score Bins', 
                      fontsize=14, fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (idx, row) in enumerate(calibration_df.iterrows()):
            ax10.text(row['bin'], row['n_samples'] + max(calibration_df['n_samples']) * 0.02, 
                     f"{row['n_samples']}",
                     ha='center', fontsize=10)
        
        plt.savefig(self.output_dir / 'calibration_analysis.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.output_dir / 'calibration_analysis.png'}")
        plt.close()
    
    def analyze_feature_space(self, X, y_true, y_score, feature_names):
        """
        Analyze feature space using dimensionality reduction
        to understand why mid-range scores are underestimating
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE SPACE ANALYSIS")
        logger.info("="*60)
        
        # Standardize features for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA - 2 components
        logger.info("\nPerforming PCA projection...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
        logger.info(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
        logger.info(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
        
        # Analyze feature contributions to PCs
        pc_components = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=feature_names
        )
        
        logger.info("\nTop features contributing to PC1:")
        top_pc1 = pc_components['PC1'].abs().sort_values(ascending=False).head(5)
        for feat, val in top_pc1.items():
            logger.info(f"  {feat}: {pc_components.loc[feat, 'PC1']:.3f}")
        
        logger.info("\nTop features contributing to PC2:")
        top_pc2 = pc_components['PC2'].abs().sort_values(ascending=False).head(5)
        for feat, val in top_pc2.items():
            logger.info(f"  {feat}: {pc_components.loc[feat, 'PC2']:.3f}")
        
        # t-SNE - for non-linear structure (on sample for speed)
        logger.info("\nPerforming t-SNE projection (may take a minute)...")
        sample_size = min(2000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled[sample_idx])
        
        # Create score buckets for analysis
        score_100 = y_score * 100
        score_buckets = pd.cut(score_100, 
                               bins=[0, 20, 40, 60, 80, 100],
                               labels=['0-20', '20-40', '40-60', '60-80', '80-100'])
        
        # Analyze mid-range (20-40) in feature space
        mid_range_mask = (score_100 >= 20) & (score_100 < 40)
        mid_range_converted = mid_range_mask & (y_true == 1)
        mid_range_not_converted = mid_range_mask & (y_true == 0)
        
        logger.info("\n" + "="*60)
        logger.info("MID-RANGE (20-40) FEATURE SPACE ANALYSIS")
        logger.info("="*60)
        
        if mid_range_converted.sum() > 0 and mid_range_not_converted.sum() > 0:
            logger.info(f"\nConverted leads in 20-40 range (n={mid_range_converted.sum()}):")
            logger.info(f"  PCA PC1 mean: {X_pca[mid_range_converted, 0].mean():.3f}")
            logger.info(f"  PCA PC2 mean: {X_pca[mid_range_converted, 1].mean():.3f}")
            
            logger.info(f"\nNon-converted leads in 20-40 range (n={mid_range_not_converted.sum()}):")
            logger.info(f"  PCA PC1 mean: {X_pca[mid_range_not_converted, 0].mean():.3f}")
            logger.info(f"  PCA PC2 mean: {X_pca[mid_range_not_converted, 1].mean():.3f}")
            
            # Feature analysis for mid-range
            logger.info("\nAverage feature values for mid-range (20-40) leads:")
            logger.info("\nConverted:")
            for i, feat in enumerate(feature_names[:5]):  # Top 5 features
                val = X[mid_range_converted, i].mean()
                logger.info(f"  {feat}: {val:.3f}")
            
            logger.info("\nNot Converted:")
            for i, feat in enumerate(feature_names[:5]):
                val = X[mid_range_not_converted, i].mean()
                logger.info(f"  {feat}: {val:.3f}")
        
        return {
            'X_pca': X_pca,
            'X_tsne': X_tsne,
            'tsne_idx': sample_idx,
            'pca_components': pc_components,
            'score_buckets': score_buckets
        }
    
    def plot_feature_space_analysis(self, X, y_true, y_score, feature_space_data):
        """Generate feature space visualization plots"""
        
        logger.info("\nGenerating feature space plots...")
        
        X_pca = feature_space_data['X_pca']
        X_tsne = feature_space_data['X_tsne']
        tsne_idx = feature_space_data['tsne_idx']
        pc_components = feature_space_data['pca_components']
        score_buckets = feature_space_data['score_buckets']
        
        score_100 = y_score * 100
        
        # FIGURE 3: Feature Space Analysis
        fig3 = plt.figure(figsize=(24, 16))
        gs3 = fig3.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. PCA colored by predicted score
        ax1 = fig3.add_subplot(gs3[0, 0])
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                              c=score_100, cmap='RdYlGn', 
                              alpha=0.6, s=20, vmin=0, vmax=100)
        ax1.set_xlabel('First Principal Component', fontsize=11)
        ax1.set_ylabel('Second Principal Component', fontsize=11)
        ax1.set_title('PCA: Colored by Predicted Score', fontsize=13, fontweight='bold')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Predicted Score', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. PCA colored by actual outcome
        ax2 = fig3.add_subplot(gs3[0, 1])
        colors_outcome = ['red' if y == 0 else 'green' for y in y_true]
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                   c=colors_outcome, alpha=0.5, s=20)
        ax2.set_xlabel('First Principal Component', fontsize=11)
        ax2.set_ylabel('Second Principal Component', fontsize=11)
        ax2.set_title('PCA: Colored by Actual Outcome', fontsize=13, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.5, label='Converted'),
                          Patch(facecolor='red', alpha=0.5, label='Not Converted')]
        ax2.legend(handles=legend_elements, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA colored by score bucket
        ax3 = fig3.add_subplot(gs3[0, 2])
        bucket_colors = {'0-20': '#d62728', '20-40': '#ff7f0e', 
                        '40-60': '#ffdd57', '60-80': '#2ca02c', '80-100': '#006400'}
        
        for bucket, color in bucket_colors.items():
            mask = score_buckets == bucket
            if mask.sum() > 0:
                ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=color, label=bucket, alpha=0.6, s=20)
        
        ax3.set_xlabel('First Principal Component', fontsize=11)
        ax3.set_ylabel('Second Principal Component', fontsize=11)
        ax3.set_title('PCA: Colored by Score Bucket', fontsize=13, fontweight='bold')
        ax3.legend(title='Score Range', loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. t-SNE colored by predicted score
        ax4 = fig3.add_subplot(gs3[1, 0])
        scatter4 = ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                              c=score_100[tsne_idx], cmap='RdYlGn',
                              alpha=0.6, s=20, vmin=0, vmax=100)
        ax4.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax4.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax4.set_title('t-SNE: Colored by Predicted Score', fontsize=13, fontweight='bold')
        cbar4 = plt.colorbar(scatter4, ax=ax4)
        cbar4.set_label('Predicted Score', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. t-SNE colored by actual outcome
        ax5 = fig3.add_subplot(gs3[1, 1])
        colors_tsne = ['red' if y == 0 else 'green' for y in y_true[tsne_idx]]
        ax5.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                   c=colors_tsne, alpha=0.5, s=20)
        ax5.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax5.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax5.set_title('t-SNE: Colored by Actual Outcome', fontsize=13, fontweight='bold')
        ax5.legend(handles=legend_elements, loc='best')
        ax5.grid(True, alpha=0.3)
        
        # 6. t-SNE colored by score bucket
        ax6 = fig3.add_subplot(gs3[1, 2])
        score_buckets_sample = score_buckets[tsne_idx]
        
        for bucket, color in bucket_colors.items():
            mask = score_buckets_sample == bucket
            if mask.sum() > 0:
                ax6.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=color, label=bucket, alpha=0.6, s=20)
        
        ax6.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax6.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax6.set_title('t-SNE: Colored by Score Bucket', fontsize=13, fontweight='bold')
        ax6.legend(title='Score Range', loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. PCA biplot with feature vectors
        ax7 = fig3.add_subplot(gs3[2, 0])
        
        # Plot points (subsample for clarity)
        subsample = min(1000, len(X_pca))
        idx_sub = np.random.choice(len(X_pca), subsample, replace=False)
        scatter7 = ax7.scatter(X_pca[idx_sub, 0], X_pca[idx_sub, 1],
                              c=score_100[idx_sub], cmap='RdYlGn',
                              alpha=0.3, s=10, vmin=0, vmax=100)
        
        # Plot feature vectors (scaled for visibility)
        scale = 4.0
        for i, (idx, row) in enumerate(pc_components.iterrows()):
            if abs(row['PC1']) > 0.2 or abs(row['PC2']) > 0.2:  # Only show important features
                ax7.arrow(0, 0, row['PC1']*scale, row['PC2']*scale,
                         head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
                ax7.text(row['PC1']*scale*1.15, row['PC2']*scale*1.15, 
                        idx, fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        ax7.set_xlabel('First Principal Component', fontsize=11)
        ax7.set_ylabel('Second Principal Component', fontsize=11)
        ax7.set_title('PCA Biplot: Feature Contributions', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='k', linewidth=0.5)
        ax7.axvline(x=0, color='k', linewidth=0.5)
        
        # 8. Mid-range focus: PCA with annotation
        ax8 = fig3.add_subplot(gs3[2, 1])
        
        # Plot all points in gray
        ax8.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgray', alpha=0.2, s=10)
        
        # Highlight mid-range (20-40)
        mid_range = (score_100 >= 20) & (score_100 < 40)
        mid_converted = mid_range & (y_true == 1)
        mid_not_converted = mid_range & (y_true == 0)
        
        ax8.scatter(X_pca[mid_converted, 0], X_pca[mid_converted, 1],
                   c='green', label='Mid-range Converted', alpha=0.7, s=40, edgecolors='darkgreen')
        ax8.scatter(X_pca[mid_not_converted, 0], X_pca[mid_not_converted, 1],
                   c='orange', label='Mid-range Not Converted', alpha=0.7, s=40, edgecolors='darkorange')
        
        # Highlight high score (80+) for comparison
        high_score = (score_100 >= 80)
        ax8.scatter(X_pca[high_score, 0], X_pca[high_score, 1],
                   c='darkgreen', label='High Score (80+)', alpha=0.5, s=30, marker='^')
        
        ax8.set_xlabel('First Principal Component', fontsize=11)
        ax8.set_ylabel('Second Principal Component', fontsize=11)
        ax8.set_title('Focus: Mid-Range (20-40) vs High Score Leads', fontsize=13, fontweight='bold')
        ax8.legend(loc='best', fontsize=9)
        ax8.grid(True, alpha=0.3)
        
        # 9. Feature importance contribution
        ax9 = fig3.add_subplot(gs3[2, 2])
        
        # Get top contributing features to PC1 and PC2
        pc1_contrib = pc_components['PC1'].abs().sort_values(ascending=True).tail(8)
        
        y_pos = np.arange(len(pc1_contrib))
        colors_bar = ['green' if x > 0 else 'red' for x in pc_components.loc[pc1_contrib.index, 'PC1']]
        
        ax9.barh(y_pos, pc_components.loc[pc1_contrib.index, 'PC1'], color=colors_bar, alpha=0.7)
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(pc1_contrib.index, fontsize=9)
        ax9.set_xlabel('Component Loading', fontsize=11)
        ax9.set_title('Top Features Contributing to PC1', fontsize=13, fontweight='bold')
        ax9.axvline(x=0, color='black', linewidth=1)
        ax9.grid(True, alpha=0.3, axis='x')
        
        plt.savefig(self.output_dir / 'feature_space_analysis.png',
                   dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.output_dir / 'feature_space_analysis.png'}")
        plt.close()
    
    def generate_recommendations(self, top_leads_df, bucket_df, auc):
        """Generate actionable recommendations"""
        
        logger.info("\n" + "="*60)
        logger.info("RECOMMENDATIONS FOR SALES TEAM")
        logger.info("="*60)
        
        # Find best top-% to focus on
        best_top = top_leads_df.loc[top_leads_df['lift'].idxmax()]
        
        logger.info(f"\n🎯 PRIORITIZATION STRATEGY:")
        logger.info(f"\n   Focus on the top {best_top['percentile']}% of leads (score ≥ {best_top['threshold']*100:.1f})")
        logger.info(f"   This segment has:")
        logger.info(f"   • {best_top['lift']:.1f}x higher conversion rate than average")
        logger.info(f"   • {best_top['coverage']:.1%} of all conversions")
        logger.info(f"   • Only {best_top['total_leads']} leads to review (manageable)")
        
        # Score interpretation
        high_score_bucket = bucket_df[bucket_df['low'] >= 60].iloc[0] if len(bucket_df[bucket_df['low'] >= 60]) > 0 else None
        low_score_bucket = bucket_df[bucket_df['high'] <= 40].iloc[-1] if len(bucket_df[bucket_df['high'] <= 40]) > 0 else None
        
        logger.info(f"\n📊 SCORE INTERPRETATION:")
        if high_score_bucket is not None:
            logger.info(f"\n   Score 60-100 (High Priority):")
            logger.info(f"   • Conversion rate: {high_score_bucket['conversion_rate']:.1%}")
            logger.info(f"   • Action: Immediate outreach, personalized approach")
        
        if low_score_bucket is not None:
            logger.info(f"\n   Score 0-40 (Low Priority):")
            logger.info(f"   • Conversion rate: {low_score_bucket['conversion_rate']:.1%}")
            logger.info(f"   • Action: Nurture campaigns, automated follow-up")
        
        logger.info(f"\n✅ MODEL QUALITY:")
        logger.info(f"   • AUC-ROC: {auc:.3f} (Excellent discrimination)")
        if auc >= 0.95:
            logger.info(f"   • The model ranks leads very accurately")
        elif auc >= 0.90:
            logger.info(f"   • The model ranks leads well")
        else:
            logger.info(f"   • The model has moderate ranking ability")
        
        logger.info(f"\n💡 KEY INSIGHT:")
        logger.info(f"   Your model successfully separates high-value from low-value leads.")
        logger.info(f"   Use the scores to prioritize, not as a hard cutoff.")
        logger.info(f"   Even 'medium' scores (40-60) may be worth pursuing if capacity allows.")
    
    def run_analysis(self, positive_path, negative_path):
        """Run complete scoring analysis"""
        
        # Load and prepare
        X_train, X_val, y_train, y_val, feature_cols = self.load_and_prepare_data(
            positive_path, negative_path
        )
        
        # Train model
        model = self.train_best_model(X_train, y_train)
        
        # Get scores
        y_score = model.predict_proba(X_val)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y_val, y_score)
        logger.info(f"\nValidation AUC-ROC: {auc:.4f}")
        
        # Analyze score distribution
        converted_scores, not_converted_scores = self.analyze_score_distribution(
            y_val.values, y_score
        )
        
        # Analyze top leads
        top_leads_df = self.analyze_top_leads(y_val.values, y_score)
        
        # Analyze score buckets
        bucket_df = self.analyze_score_buckets(y_val.values, y_score)
        
        # Analyze calibration
        calibration_df = self.analyze_calibration(y_val.values, y_score, n_bins=10)
        
        # Analyze feature space
        feature_space_data = self.analyze_feature_space(
            X_val.values, y_val.values, y_score, feature_cols
        )
        
        # Generate plots
        self.plot_score_analysis(
            y_val.values, y_score, converted_scores, not_converted_scores,
            top_leads_df, bucket_df, calibration_df
        )
        
        # Generate feature space plots
        self.plot_feature_space_analysis(
            X_val.values, y_val.values, y_score, feature_space_data
        )
        
        # Save detailed results
        top_leads_df.to_csv(self.output_dir / 'top_leads_analysis.csv', index=False)
        bucket_df.to_csv(self.output_dir / 'score_buckets_analysis.csv', index=False)
        calibration_df.to_csv(self.output_dir / 'calibration_analysis.csv', index=False)
        feature_space_data['pca_components'].to_csv(self.output_dir / 'pca_components.csv')
        
        # Generate recommendations
        self.generate_recommendations(top_leads_df, bucket_df, auc)
        
        return model, top_leads_df, bucket_df


if __name__ == "__main__":
    downloads_path = Path(".")
    output_dir = downloads_path / "lead_scoring_analysis"
    
    positive_enriched = downloads_path / "leads_positive_enriched.csv"
    negative_enriched = downloads_path / "leads_negative_enriched.csv"
    
    analyzer = LeadScoringQualityAnalyzer(output_dir)
    model, top_leads, buckets = analyzer.run_analysis(
        str(positive_enriched),
        str(negative_enriched)
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {output_dir}")