import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_curve, auc
)
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    'SVM':                '#6C63FF',
    'KNN':                '#FF6584',
    'Random Forest':      '#43B89C',
    'Logistic Regression':'#F4A261',
}
PALETTE  = list(COLORS.values())
BG       = '#0F1117'
PANEL    = '#1A1D27'
TEXT     = '#E8E8F0'
SUBTEXT  = '#9090A8'
GRID     = '#2A2D3E'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   TEXT,
    'xtick.color':       SUBTEXT,
    'ytick.color':       SUBTEXT,
    'text.color':        TEXT,
    'grid.color':        GRID,
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
})

# ── 1. Load & Prepare Data ─────────────────────────────────────────────────────
wine   = load_wine()
X, y   = wine.data, wine.target
labels = wine.target_names          # ['class_0', 'class_1', 'class_2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 2. Define Models ───────────────────────────────────────────────────────────
models = {
    'SVM':                 SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=7, metric='minkowski'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
}

results, trained = {}, {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    # scaled for SVM / KNN / LR; raw for RF
    Xtr = X_train_s if name != 'Random Forest' else X_train
    Xte = X_test_s  if name != 'Random Forest' else X_test
    clf.fit(Xtr, y_train)
    y_pred   = clf.predict(Xte)
    y_proba  = clf.predict_proba(Xte)
    cv_sc    = cross_val_score(clf, Xtr, y_train, cv=cv, scoring='f1_macro')
    results[name] = {
        'y_pred':   y_pred,
        'y_proba':  y_proba,
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_class': f1_score(y_test, y_pred, average=None),
        'cm':       confusion_matrix(y_test, y_pred),
        'cv_mean':  cv_sc.mean(),
        'cv_std':   cv_sc.std(),
        'report':   classification_report(y_test, y_pred, target_names=labels, output_dict=True),
    }
    trained[name] = (clf, Xte)

# ── 3. Figure 1: Confusion Matrices ───────────────────────────────────────────
fig1, axes = plt.subplots(1, 4, figsize=(20, 5))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Confusion Matrices — Wine Quality Dataset', fontsize=16,
              fontweight='bold', color=TEXT, y=1.02)

for ax, (name, res) in zip(axes, results.items()):
    cm   = res['cm']
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(norm, annot=cm, fmt='d', ax=ax,
                cmap=sns.light_palette(COLORS[name], as_cmap=True),
                linewidths=1, linecolor=BG,
                xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={'size': 12, 'color': TEXT})
    ax.set_title(name, fontsize=13, fontweight='bold', color=COLORS[name], pad=10)
    ax.set_xlabel('Predicted', labelpad=6)
    ax.set_ylabel('Actual', labelpad=6)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    f1 = res['f1_macro']
    ax.text(0.5, -0.22, f'Macro F1: {f1:.3f}', ha='center',
            transform=ax.transAxes, fontsize=11, color=COLORS[name], fontweight='bold')

plt.tight_layout(pad=2)
plt.savefig('1_confusion_matrices.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: confusion matrices")

# ── 4. Figure 2: F1-Score Breakdown ───────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(18, 6))
fig2.patch.set_facecolor(BG)
fig2.suptitle('F1-Score Analysis', fontsize=16, fontweight='bold', color=TEXT)

# 4a. Grouped bar — per-class F1
ax = axes[0]
x       = np.arange(len(labels))
width   = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]
for i, (name, res) in enumerate(results.items()):
    bars = ax.bar(x + offsets[i]*width, res['f1_class'], width,
                  label=name, color=COLORS[name], alpha=0.88, zorder=3)
    for bar, val in zip(bars, res['f1_class']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, color=TEXT)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel('F1-Score'); ax.set_title('Per-Class F1-Score', fontsize=13, color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
ax.yaxis.grid(True); ax.set_axisbelow(True)

# 4b. CV Macro-F1 with error bars
ax = axes[1]
names   = list(results.keys())
means   = [results[n]['cv_mean']  for n in names]
stds    = [results[n]['cv_std']   for n in names]
colors  = [COLORS[n] for n in names]
bars    = ax.bar(names, means, color=colors, alpha=0.88,
                 yerr=stds, capsize=6, error_kw={'ecolor': TEXT, 'linewidth': 1.5}, zorder=3)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{m:.3f}', ha='center', va='bottom', fontsize=11,
            color=TEXT, fontweight='bold')
ax.set_ylim(0.80, 1.02)
ax.set_ylabel('Macro F1 (5-Fold CV)')
ax.set_title('Cross-Validated Macro F1 ± Std', fontsize=13, color=TEXT)
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.tick_params(axis='x', rotation=12)

plt.tight_layout(pad=2)
plt.savefig('2_f1_scores.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: F1-score analysis")

# ── 5. Figure 3: ROC Curves (One-vs-Rest) ─────────────────────────────────────
y_bin = label_binarize(y_test, classes=[0, 1, 2])

fig3, axes = plt.subplots(1, 4, figsize=(22, 5))
fig3.patch.set_facecolor(BG)
fig3.suptitle('ROC Curves — One-vs-Rest (per class)', fontsize=16,
              fontweight='bold', color=TEXT, y=1.02)

class_colors = ['#6C63FF', '#FF6584', '#43B89C']

for ax, (name, res) in zip(axes, results.items()):
    y_proba = res['y_proba']
    for i, (cls, cc) in enumerate(zip(labels, class_colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cc, lw=2, label=f'{cls} (AUC={roc_auc:.2f})')
        ax.fill_between(fpr, tpr, alpha=0.06, color=cc)
    ax.plot([0,1],[0,1], '--', color=SUBTEXT, lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('FPR', labelpad=4); ax.set_ylabel('TPR', labelpad=4)
    ax.set_title(name, fontsize=13, fontweight='bold', color=COLORS[name], pad=10)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc='lower right')
    ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)

plt.tight_layout(pad=2)
plt.savefig('3_roc_curves.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: ROC curves")

# ── 6. Figure 4: Summary Dashboard ────────────────────────────────────────────
fig4 = plt.figure(figsize=(18, 7))
fig4.patch.set_facecolor(BG)
gs   = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.35)
fig4.suptitle('Model Comparison — Summary Dashboard', fontsize=16,
              fontweight='bold', color=TEXT)

# 6a. Radar chart
ax_rad = fig4.add_subplot(gs[0], polar=True)
metrics_names = ['Macro F1', 'CV Score', 'Class-0 F1', 'Class-1 F1', 'Class-2 F1']
N    = len(metrics_names)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
ax_rad.set_facecolor(PANEL)
ax_rad.set_theta_offset(np.pi / 2); ax_rad.set_theta_direction(-1)
ax_rad.set_rlim(0.7, 1.0)
ax_rad.set_rticks([0.75, 0.85, 0.95, 1.0])
ax_rad.set_rlabel_position(30)
ax_rad.set_thetagrids(np.degrees(angles[:-1]), metrics_names, color=SUBTEXT, fontsize=8)
ax_rad.grid(color=GRID, linewidth=0.7)
for name, res in results.items():
    vals = [res['f1_macro'], res['cv_mean']] + list(res['f1_class'])
    vals += vals[:1]
    ax_rad.plot(angles, vals, 'o-', lw=2, color=COLORS[name], label=name)
    ax_rad.fill(angles, vals, alpha=0.07, color=COLORS[name])
ax_rad.set_title('Performance Radar', fontsize=12, color=TEXT, pad=14)
ax_rad.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# 6b. Feature Importances (RF)
ax_fi = fig4.add_subplot(gs[1])
rf_clf = trained['Random Forest'][0]
importances = rf_clf.feature_importances_
feat_names  = wine.feature_names
top_idx     = np.argsort(importances)[::-1][:8]
colors_fi   = [COLORS['Random Forest']] * 8
ax_fi.barh(np.array(feat_names)[top_idx][::-1],
           importances[top_idx][::-1], color=colors_fi, alpha=0.88)
ax_fi.set_xlabel('Importance'); ax_fi.set_title('RF Feature Importances (Top 8)',
                                                   fontsize=12, color=TEXT)
ax_fi.xaxis.grid(True); ax_fi.set_axisbelow(True)
for i, (idx, val) in enumerate(zip(top_idx[::-1], importances[top_idx][::-1])):
    ax_fi.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8, color=TEXT)

# 6c. Score summary table
ax_tb = fig4.add_subplot(gs[2])
ax_tb.axis('off')
col_labels = ['Model', 'Test F1', 'CV F1', 'CV ±']
rows = []
for name, res in results.items():
    rows.append([name, f"{res['f1_macro']:.3f}",
                 f"{res['cv_mean']:.3f}", f"{res['cv_std']:.3f}"])
tbl = ax_tb.table(cellText=rows, colLabels=col_labels,
                  loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if r > 0 else '#252836')
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=TEXT if r > 0 else COLORS[rows[r-1][0]] if r > 0 else TEXT)
    if r == 0:
        cell.set_text_props(color=TEXT, fontweight='bold')
    elif rows[r-1][0] == 'Random Forest':
        cell.set_facecolor('#1E2E2A')
ax_tb.set_title('Score Summary', fontsize=12, color=TEXT, pad=10)

plt.tight_layout(pad=2)
plt.savefig('4_summary_dashboard.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: summary dashboard")

# ── 7. Print full classification reports ──────────────────────────────────────
print("\n" + "="*60)
for name, res in results.items():
    print(f"\n{'='*60}")
    print(f"  {name}  |  Test Macro-F1: {res['f1_macro']:.4f}  |  CV: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
    print(f"{'='*60}")
    clf, Xte = trained[name]
    Xtr_full = X_train_s if name != 'Random Forest' else X_train
    Xte_full = X_test_s  if name != 'Random Forest' else X_test
    print(classification_report(y_test, res['y_pred'], target_names=labels))