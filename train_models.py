# ============================================================================
# GENETIC DISEASE PREDICTION - MODEL TRAINING PIPELINE
# ============================================================================
# This script trains multiple machine learning models on preprocessed data,
# evaluates their performance, and saves the best model.
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seaborn style for professional visualizations
sns.set_style("whitegrid")

# ============================================================================
# STEP 1: CREATE OUTPUT FOLDERS
# ============================================================================
models_folder = "models"
outputs_folder = "outputs"

if not os.path.exists(models_folder):
    os.makedirs(models_folder)
    print(f"✓ Created models folder: {models_folder}/")

if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)
    print(f"✓ Created outputs folder: {outputs_folder}/")

# ============================================================================
# STEP 2: LOAD AND PREPROCESS DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING AND PREPROCESSING DATA")
print("="*70)

# Load dataset
dataset_file = "gene_sequence_dataset.csv"
try:
    df = pd.read_csv(dataset_file, nrows=2000)
    print(f"✓ Dataset loaded: {df.shape}")
except FileNotFoundError:
    print(f"✗ Error: {dataset_file} not found!")
    exit()

# Remove specified columns
columns_to_remove = ['amino_acid_change', 'zygosity', 'sequence_length',
                     'mutation_type', 'mutation_detail', 'gene']
existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
if existing_columns_to_remove:
    df = df.drop(columns=existing_columns_to_remove)

# Handle missing values
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].fillna("Unknown")
        df[col] = df[col].fillna(mode_value)

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

# Remove duplicates
df = df.drop_duplicates()

# Detect DNA sequence column
dna_col = None
for col in df.columns:
    if df[col].dtype == 'object':
        sample_text = df[col].astype(str).iloc[0].upper()
        if all(c in 'ACGTN' for c in sample_text if c.isalpha()):
            dna_col = col
            print(f"✓ Detected DNA sequence column: '{dna_col}'")
            break

# Set target column to 'disease'
target_col = 'disease'
if target_col not in df.columns:
    print(f"✗ Error: Target column '{target_col}' not found in dataset")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

unique_count = df[target_col].nunique()
print(f"✓ Using target column: '{target_col}' with {unique_count} unique values")
print(f"Target classes: {sorted(df[target_col].unique())}")

# K-mer extraction
X_dna = df[dna_col].astype(str)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
X_features = vectorizer.fit_transform(X_dna).toarray()

# Target encoding
y = df[target_col]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✓ Preprocessing completed: {X_train_pca.shape[0]} train × {X_train_pca.shape[1]} features")

# Save preprocessed data
preprocessed_data_path = os.path.join(outputs_folder, "preprocessed_data.pkl")
data_dict = {
    'X_train_pca': X_train_pca,
    'X_test_pca': X_test_pca,
    'y_train': y_train,
    'y_test': y_test,
    'label_encoder': label_encoder
}
joblib.dump(data_dict, preprocessed_data_path)
print(f"✓ Preprocessed data saved to: {preprocessed_data_path}")
# ============================================================================
# STEP 3: INITIALIZE MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: TRAINING MACHINE LEARNING MODELS")
print("="*70)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Dictionary to store trained models
trained_models = {}
predictions = {}
accuracies = {}

# Train each model
for model_name, model in models.items():
    print(f"\n   Training {model_name}...")
    model.fit(X_train_pca, y_train)
    trained_models[model_name] = model
    predictions[model_name] = model.predict(X_test_pca)
    print(f"   ✓ {model_name} trained successfully")

print(f"\n✓ All models trained successfully")

# ============================================================================
# STEP 4: EVALUATE MODELS
# ============================================================================
print("\n" + "="*70)
print("STEP 3: EVALUATING MODELS")
print("="*70)

# Calculate evaluation metrics for each model
results = []

for model_name, predictions_data in predictions.items():
    accuracy = accuracy_score(y_test, predictions_data)
    precision = precision_score(y_test, predictions_data, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions_data, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions_data, average='weighted', zero_division=0)
    
    accuracies[model_name] = accuracy
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{model_name}:")
    print(f"   • Accuracy:  {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall:    {recall:.4f}")
    print(f"   • F1-Score:  {f1:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results)

# ============================================================================
# STEP 5: SELECT BEST MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 4: SELECTING BEST MODEL")
print("="*70)

best_model_name = max(accuracies, key=accuracies.get)
best_model = trained_models[best_model_name]
best_accuracy = accuracies[best_model_name]

print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Best Accuracy: {best_accuracy:.4f}")

# ============================================================================
# STEP 6: GENERATE CONFUSION MATRIX
# ============================================================================
print("\n" + "="*70)
print("STEP 5: GENERATING CONFUSION MATRIX")
print("="*70)

# Get best model predictions
best_predictions = predictions[best_model_name]
cm = confusion_matrix(y_test, best_predictions)

# Create confusion matrix visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix - {best_model_name}\nAccuracy: {best_accuracy:.4f}', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
cm_path = os.path.join(outputs_folder, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {cm_path}")
plt.close()

# ============================================================================
# STEP 7: CREATE MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "="*70)
print("STEP 6: MODEL COMPARISON")
print("="*70)

print("\n" + results_df.to_string(index=False))

# Save results table as CSV
results_csv_path = os.path.join(outputs_folder, "model_comparison.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\n✓ Model comparison saved to: {results_csv_path}")

# ============================================================================
# STEP 8: CREATE ACCURACY COMPARISON BAR CHART
# ============================================================================
print("\n" + "="*70)
print("STEP 7: CREATING ACCURACY COMPARISON CHART")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart
models_list = list(accuracies.keys())
accuracies_list = list(accuracies.values())
colors = ['green' if model == best_model_name else 'steelblue' for model in models_list]

bars = ax.bar(models_list, accuracies_list, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
for bar, accuracy in zip(bars, accuracies_list):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{accuracy:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - Accuracy', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, max(accuracies_list) * 1.15)
ax.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.8, label='Best Model'),
                   Patch(facecolor='steelblue', alpha=0.8, label='Other Models')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
accuracy_chart_path = os.path.join(outputs_folder, "accuracy_comparison.png")
plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
print(f"✓ Accuracy comparison chart saved to: {accuracy_chart_path}")
plt.close()

# ============================================================================
# STEP 9: DISPLAY CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*70)
print("STEP 8: CLASSIFICATION REPORT - BEST MODEL")
print("="*70)

print(f"\n{best_model_name}:\n")
print(classification_report(y_test, best_predictions, 
                           target_names=label_encoder.classes_,
                           zero_division=0))

# ============================================================================
# STEP 10: SAVE BEST MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 9: SAVING BEST MODEL")
print("="*70)

best_model_path = os.path.join(models_folder, "best_model.pkl")
joblib.dump(best_model, best_model_path)
print(f"✓ Best model saved to: {best_model_path}")

# Also save model metadata including preprocessing artifacts
metadata = {
    'model_name': best_model_name,
    'accuracy': best_accuracy,
    'label_encoder': label_encoder,
    'model': best_model,
    'vectorizer': vectorizer,
    'scaler': scaler,
    'pca': pca
}
metadata_path = os.path.join(models_folder, "model_metadata.pkl")
joblib.dump(metadata, metadata_path)
print(f"✓ Model metadata saved to: {metadata_path}")

# Save preprocessing artifacts as separate files
vectorizer_path = os.path.join(models_folder, "vectorizer.pkl")
scaler_path = os.path.join(models_folder, "scaler.pkl")
pca_path = os.path.join(models_folder, "pca.pkl")

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(scaler, scaler_path)
joblib.dump(pca, pca_path)

print(f"✓ Vectorizer saved to: {vectorizer_path}")
print(f"✓ Scaler saved to: {scaler_path}")
print(f"✓ PCA saved to: {pca_path}")

# ============================================================================
# STEP 11: FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("="*70)

print(f"\n📊 TRAINING SUMMARY:")
print(f"   • Training samples: {X_train_pca.shape[0]}")
print(f"   • Testing samples: {X_test_pca.shape[0]}")
print(f"   • Features: {X_train_pca.shape[1]}")
print(f"   • Models trained: {len(trained_models)}")

print(f"\n🏆 BEST MODEL:")
print(f"   • Model: {best_model_name}")
print(f"   • Accuracy: {best_accuracy:.4f}")

print(f"\n📈 MODEL ACCURACIES:")
for model_name, accuracy in accuracies.items():
    marker = "⭐" if model_name == best_model_name else "  "
    print(f"   {marker} {model_name}: {accuracy:.4f}")

print(f"\n💾 OUTPUTS SAVED:")
print(f"   • Best model: {best_model_path}")
print(f"   • Model metadata: {metadata_path}")
print(f"   • Confusion matrix: {cm_path}")
print(f"   • Accuracy comparison: {accuracy_chart_path}")
print(f"   • Model comparison CSV: {results_csv_path}")

print("\n" + "="*70)
print("✓ All training steps completed successfully!")
print("="*70 + "\n")
