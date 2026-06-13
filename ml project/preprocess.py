# ============================================================================
# GENETIC DISEASE PREDICTION - PREPROCESSING PIPELINE
# ============================================================================
# This script handles data loading, cleaning, feature engineering, and
# visualization for the genetic disease prediction ML project.
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seaborn style to whitegrid for professional visualizations
sns.set_style("whitegrid")

# ============================================================================
# STEP 1: CREATE OUTPUT FOLDER
# ============================================================================
output_folder = "outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"✓ Created output folder: {output_folder}/")

# ============================================================================
# STEP 2: LOAD DATASET
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATASET")
print("="*70)

dataset_file = "gene_sequence_dataset.csv"
try:
    # Load only first 2000 rows
    df = pd.read_csv(dataset_file, nrows=2000)
    print(f"✓ Dataset loaded successfully from {dataset_file}")
except FileNotFoundError:
    print(f"✗ Error: {dataset_file} not found!")
    exit()

# Display dataset information
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 Rows:")
print(df.head())
print(f"\nColumn Names:")
print(df.columns.tolist())
print(f"\nMissing Values Summary:")
print(df.isnull().sum())

# ============================================================================
# STEP 3: REMOVE SPECIFIED COLUMNS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: REMOVING SPECIFIED COLUMNS")
print("="*70)

columns_to_remove = ['amino_acid_change', 'zygosity', 'sequence_length', 
                     'mutation_type', 'mutation_detail', 'gene']

# Remove only columns that exist in the dataset
existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
if existing_columns_to_remove:
    df = df.drop(columns=existing_columns_to_remove)
    print(f"✓ Removed columns: {existing_columns_to_remove}")
else:
    print("✓ No specified columns found to remove")

# ============================================================================
# STEP 4: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*70)
print("STEP 3: HANDLING MISSING VALUES")
print("="*70)

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Fill missing categorical values with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].fillna("Unknown")
        df[col] = df[col].fillna(mode_value)
        print(f"✓ Filled missing values in '{col}' with mode")

# Fill missing numerical values with median
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"✓ Filled missing values in '{col}' with median")

# ============================================================================
# STEP 5: REMOVE DUPLICATE ROWS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: REMOVING DUPLICATES")
print("="*70)

duplicates_count = df.duplicated().sum()
if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicates_count} duplicate rows")
else:
    print("✓ No duplicate rows found")

print(f"\n✓ Cleaned dataset shape: {df.shape}")

# ============================================================================
# STEP 6: AUTO-DETECT DNA SEQUENCE AND TARGET COLUMNS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: AUTO-DETECTING DNA SEQUENCE AND TARGET COLUMNS")
print("="*70)

# Detect DNA sequence column (contains A, C, G, T characters)
dna_col = None
target_col = None

for col in df.columns:
    if df[col].dtype == 'object':
        sample_text = df[col].astype(str).iloc[0].upper()
        # Check if it looks like DNA sequence
        if all(c in 'ACGTN' for c in sample_text if c.isalpha()):
            dna_col = col
            print(f"✓ Detected DNA sequence column: '{dna_col}'")
            break

# Detect target column (specifically look for 'disease' column)
target_col = None
if 'disease' in df.columns:
    target_col = 'disease'
    unique_count = df['disease'].nunique()
    print(f"✓ Detected target column: '{target_col}' with {unique_count} unique values")
else:
    # Fallback to auto-detection if 'disease' not found
    for col in df.columns:
        if col != dna_col and df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count <= 10:  # Likely a target column
                target_col = col
                print(f"✓ Detected target column: '{target_col}' with {unique_count} unique values")
                break

if dna_col is None or target_col is None:
    print("✗ Warning: Could not auto-detect DNA sequence or target column")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# ============================================================================
# STEP 7: FEATURE ENGINEERING - K-MER EXTRACTION
# ============================================================================
print("\n" + "="*70)
print("STEP 6: FEATURE ENGINEERING - K-MER EXTRACTION")
print("="*70)

# Extract DNA sequences
X_dna = df[dna_col].astype(str)

# Apply CountVectorizer with k-mer extraction (k=3)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
X_features = vectorizer.fit_transform(X_dna)

# Convert sparse matrix to dense array for further processing
X_features = X_features.toarray()

print(f"✓ K-mer features extracted using CountVectorizer")
print(f"✓ Feature matrix shape: {X_features.shape}")
print(f"✓ Number of k-mers (features): {X_features.shape[1]}")

# ============================================================================
# STEP 8: TARGET ENCODING
# ============================================================================
print("\n" + "="*70)
print("STEP 7: TARGET ENCODING")
print("="*70)

# Extract target variable
y = df[target_col]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"✓ Target variable encoded")
print(f"✓ Classes: {label_encoder.classes_}")
print(f"✓ Target shape: {y_encoded.shape}")

# ============================================================================
# STEP 9: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("STEP 8: TRAIN-TEST SPLIT (80-20)")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded, 
    test_size=0.2, 
    stratify=y_encoded, 
    random_state=42
)

print(f"✓ Training set size: {X_train.shape[0]} samples")
print(f"✓ Testing set size: {X_test.shape[0]} samples")
print(f"✓ Number of features: {X_train.shape[1]}")

# ============================================================================
# STEP 10: FEATURE SCALING
# ============================================================================
print("\n" + "="*70)
print("STEP 9: FEATURE SCALING")
print("="*70)

# Initialize and fit StandardScaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")
print(f"✓ Training set mean: {X_train_scaled.mean():.6f}")
print(f"✓ Training set std: {X_train_scaled.std():.6f}")

# ============================================================================
# STEP 11: PCA - DIMENSIONALITY REDUCTION
# ============================================================================
print("\n" + "="*70)
print("STEP 10: PCA - DIMENSIONALITY REDUCTION (95% VARIANCE)")
print("="*70)

# Apply PCA retaining 95% variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✓ PCA applied successfully")
print(f"✓ Original features: {X_train_scaled.shape[1]}")
print(f"✓ PCA components: {X_train_pca.shape[1]}")
print(f"✓ Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# ============================================================================
# STEP 12: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 11: CREATING VISUALIZATIONS")
print("="*70)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Visualization 1: Histogram of feature values
ax1 = plt.subplot(2, 2, 1)
feature_means = X_train_scaled.mean(axis=0)
ax1.hist(feature_means, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Mean Feature Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Histogram of Feature Means (Scaled Data)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Visualization 2: Boxplot of scaled features
ax2 = plt.subplot(2, 2, 2)
sample_features = X_train_scaled[:, :10]  # Sample first 10 features
ax2.boxplot(sample_features, labels=[f'F{i+1}' for i in range(10)])
ax2.set_ylabel('Feature Value', fontsize=11, fontweight='bold')
ax2.set_title('Boxplot of First 10 Features (Scaled Data)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Visualization 3: Correlation Heatmap of PCA components
ax3 = plt.subplot(2, 2, 3)
pca_corr = np.corrcoef(X_train_pca.T[:10, :])  # First 10 PCA components
sns.heatmap(pca_corr, annot=False, cmap='coolwarm', center=0, 
            ax=ax3, cbar_kws={'label': 'Correlation'}, square=True)
ax3.set_title('Correlation Matrix of First 10 PCA Components', fontsize=12, fontweight='bold')
ax3.set_xlabel('PCA Component', fontsize=11, fontweight='bold')
ax3.set_ylabel('PCA Component', fontsize=11, fontweight='bold')

# Visualization 4: PCA Scatter Plot
ax4 = plt.subplot(2, 2, 4)
scatter = ax4.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                     c=y_train, cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
ax4.set_title('PCA Scatter Plot (First 2 Components)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Target Class', fontsize=10, fontweight='bold')
ax4.grid(alpha=0.3)

# Adjust layout and save
plt.tight_layout()
visualizations_path = os.path.join(output_folder, "visualizations.png")
plt.savefig(visualizations_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualizations saved to: {visualizations_path}")
plt.close()

# ============================================================================
# STEP 13: FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
print("="*70)

print(f"\n📊 SUMMARY:")
print(f"   • Original dataset shape: (2000, {df.shape[1]})")
print(f"   • Final dataset shape: {df.shape}")
print(f"   • DNA sequence column: '{dna_col}'")
print(f"   • Target column: '{target_col}'")
print(f"   • K-mer features extracted: {X_features.shape[1]}")
print(f"   • Training set: {X_train_pca.shape[0]} samples × {X_train_pca.shape[1]} features")
print(f"   • Testing set: {X_test_pca.shape[0]} samples × {X_test_pca.shape[1]} features")
print(f"   • PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

print(f"\n💾 OUTPUTS SAVED:")
print(f"   • Visualizations: {output_folder}/visualizations.png")

print("\n" + "="*70)
print("✓ All preprocessing steps completed successfully!")
print("="*70 + "\n")
