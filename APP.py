import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.optimize import minimize

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import warnings
warnings.filterwarnings("ignore")

# Optional libraries flags (non-fatal)
has_xgb = False
has_lgb = False
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    pass
try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    pass

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Modélisation des émissions NOx dans l'industrie cimentière", layout="wide")

# ---------------------------
# Theme (CSS injection) + header
# ---------------------------
st.sidebar.title("Paramètres de l'application")
app_name = st.sidebar.text_input("Nom de l'application", value="Modélisation des émissions NOx dans l'industrie cimentière")
theme_choice = st.sidebar.selectbox("Thème", options=["Clair", "Sombre"])

if theme_choice == "Sombre":
    st.markdown(
        """
        <style>
        .reportview-container { background: #0e1117; color: #e6eef8; }
        .sidebar .sidebar-content { background: #0b1220; }
        .stButton>button { background-color: #2b6cb0; color: white; }
        .stDownloadButton>button { background-color: #2b6cb0; color: white; }
        .st-bf { color: white; }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stButton>button { background-color: #0f6cb0; color: white; }
        .stDownloadButton>button { background-color: #0f6cb0; color: white; }
        </style>
        """, unsafe_allow_html=True
    )

# App title
st.title(f" {app_name}")
st.write("Importez un fichier CSV contenant vos mesures et lancez l'analyse complète (automatique).")

# --------------------------------------------------------------------
#  AJOUT DU CONTEXTE 

# --------------------------------------------------------------------
st.markdown("""
---

##  Contexte du projet

L’industrie cimentière constitue l’une des principales sources industrielles d’émissions de NOx, des polluants atmosphériques dangereux responsables d’irritations pulmonaires, d’affections chroniques et de phénomènes environnementaux tels que les pluies acides.  
La variabilité de ces émissions dépend fortement des conditions de combustion, de la température du four, du taux d’oxygène ainsi que de divers paramètres physico-chimiques du procédé.

Cette application a été développée afin de :

- automatiser l’analyse chimiométrique des données industrielles ;
- prétraiter les données et gérer les valeurs manquantes ;
- comparer plusieurs modèles d’apprentissage automatique dédiés à la prédiction des NOx ;
- visualiser les résultats (ACP, distributions, résidus, importances variables) ;
- optimiser les paramètres opérationnels pour réduire les émissions ;
- permettre l’export d’un rapport PDF complet et du modèle optimal.

Elle constitue un outil clé pour les ingénieurs procédés, responsables environnement et techniciens souhaitant suivre, prédire et minimiser les émissions de NOx dans un environnement industriel réel.

---

""")

# ---------------------------
# File uploader
# ---------------------------
uploaded_file = st.file_uploader("Importer votre fichier CSV", type=["csv"])
if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour démarrer.")
    st.stop()

# Create output temp folder
out_dir = os.path.join(tempfile.gettempdir(), "nox_app_results")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
try:
    df = pd.read_csv(uploaded_file, engine="python")
except Exception as e:
    st.error(f"Impossible de lire le CSV : {e}")
    st.stop()

st.subheader("Aperçu des données")
st.dataframe(df.head())

# ---------------------------
# Preprocessing
# ---------------------------
st.write(" Nettoyage automatique des données...")

df.dropna(axis=0, how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)
df = df.loc[:, ~df.columns.duplicated()]

def detect_target_column(df):
    for c in df.columns:
        if 'nox' in c.lower() or 'no2' in c.lower() or 'emiss' in c.lower():
            return c
    return df.columns[-1]

target_col = detect_target_column(df)
st.success(f"Colonne cible détectée : **{target_col}**")

y = df[target_col]
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    st.write("Encodage colonnes catégorielles :", ", ".join(cat_cols))
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

const_cols = [c for c in X.columns if X[c].nunique() <= 1]
if const_cols:
    st.write("Colonnes constantes supprimées :", ", ".join(const_cols))
    X.drop(columns=const_cols, inplace=True)

# ---------------------------
# Imputation iterative PCA (NIPALS)
# ---------------------------
st.write(" Imputation (PCA itérative) des valeurs manquantes...")

def iterative_pca_impute(X_df, n_components=5, max_iter=100, tol=1e-6):
    X = X_df.copy().astype(float)
    mask = X.isna()
    X_filled = X.fillna(X.mean())
    prev = X_filled.copy()
    n_components = min(n_components, X.shape[1])
    for it in range(max_iter):
        pca = PCA(n_components=n_components)
        pca.fit(X_filled)
        rec = pd.DataFrame(pca.inverse_transform(pca.transform(X_filled)), index=X.index, columns=X.columns)
        X_filled[mask] = rec[mask]
        diff = np.linalg.norm((X_filled - prev).values)
        if diff < tol:
            break
        prev = X_filled.copy()
    return X_filled, pca

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Aucune colonne numérique détectée pour l'analyse.")
    st.stop()

X_num = X[numeric_cols]
n_comp_impute = min(5, X_num.shape[1])
X_imputed, pca_impute = iterative_pca_impute(X_num, n_components=n_comp_impute, max_iter=100, tol=1e-5)

other_cols = [c for c in X.columns if c not in numeric_cols]
if other_cols:
    X_processed = pd.concat([X_imputed.reset_index(drop=True), X[other_cols].reset_index(drop=True)], axis=1)
else:
    X_processed = X_imputed.copy()

y = pd.to_numeric(y, errors='coerce')
mask_y = ~y.isna()
X_processed = X_processed[mask_y].reset_index(drop=True)
y = y[mask_y].reset_index(drop=True)

st.success("Imputation terminée.")

# ---------------------------
# Save images
# ---------------------------
saved_figs = []

# ---------------------------
# Distributions
# ---------------------------
st.subheader(" Distributions (exemples)")
n_show = min(12, len(X_processed.columns))
fig = plt.figure(figsize=(16,8))
for i, col in enumerate(X_processed.columns[:n_show], start=1):
    ax = plt.subplot((n_show+3)//4, 4, i)
    sns.histplot(X_processed[col].dropna(), bins=30, kde=False, ax=ax)
    ax.set_title(col)
plt.tight_layout()
dist_path = os.path.join(out_dir, "distributions.png")
fig.savefig(dist_path, dpi=150, bbox_inches='tight')
saved_figs.append(dist_path)
st.pyplot(fig)
plt.close(fig)

# ---------------------------
# PCA 3D
# ---------------------------
st.subheader(" ACP 3D (PC1-PC2-PC3) — coloré par NOx")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
n_comp = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
pca = PCA(n_components=n_comp)
pc_scores = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

if n_comp >= 3:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pc_scores[:,0], pc_scores[:,1], pc_scores[:,2], c=y, cmap='plasma', s=45, alpha=0.9)
    fig.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)")
    plt.title("ACP 3D — coloré par NOx")
    pca3d_path = os.path.join(out_dir, "pca_3d.png")
    fig.savefig(pca3d_path, dpi=150, bbox_inches='tight')
    saved_figs.append(pca3d_path)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("ACP 3D non disponible (moins de 3 composantes).")

# ---------------------------
# MODELLING
# ---------------------------
st.subheader(" Modélisation & comparaison")

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

models = {}
param_dist = {}

models['PLS'] = PLSRegression()
param_dist['PLS'] = {'n_components': list(range(1, min(10, X_train.shape[1])+1))}

models['SVR_RBF'] = SVR(kernel='rbf')
param_dist['SVR_RBF'] = {'C': [0.1,1,10], 'epsilon':[0.01,0.05,0.1], 'gamma':['scale','auto']}

models['RandomForest'] = RandomForestRegressor(random_state=0)
param_dist['RandomForest'] = {'n_estimators':[100,200], 'max_depth':[None,6,12]}

models['MLP'] = MLPRegressor(random_state=0, max_iter=1000)
param_dist['MLP'] = {'hidden_layer_sizes':[(50,),(100,),(50,50)], 'alpha':[1e-4,1e-3]}

models['GradientBoosting'] = GradientBoostingRegressor(random_state=0)
param_dist['GradientBoosting'] = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1], 'max_depth':[3,6]}

# fallback LightGBM / XGBoost
if has_lgb:
    models['LightGBM'] = lgb.LGBMRegressor(random_state=0)
    param_dist['LightGBM'] = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1]}
else:
    models['LightGBM'] = HistGradientBoostingRegressor(random_state=0)
    param_dist['LightGBM'] = {'max_iter':[100,200]}

if has_xgb:
    models['XGBoost'] = xgb.XGBRegressor(random_state=0, n_jobs=1, verbosity=0)
    param_dist['XGBoost'] = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1], 'max_depth':[3,6]}
else:
    models['XGBoost'] = HistGradientBoostingRegressor(random_state=1)
    param_dist['XGBoost'] = {'max_iter':[100,200]}

kf = KFold(n_splits=5, shuffle=True, random_state=0)
results = []
trained_models = {}

for name, model in models.items():
    st.markdown(f"###  {name}")
    use_scaled = name in ['SVR_RBF', 'PLS', 'MLP']
    Xtr = X_train_scaled if use_scaled else X_train.values
    Xte = X_test_scaled if use_scaled else X_test.values

    pdist = param_dist.get(name, {})
    best = None
    try:
        if pdist:
            rs = RandomizedSearchCV(model, pdist, n_iter=min(8, max(1, len(pdist)*2)), cv=kf,
                                    scoring='neg_root_mean_squared_error', random_state=0, n_jobs=-1)
            rs.fit(Xtr, y_train)
            best = rs.best_estimator_
            st.write("Meilleurs params (approx):", rs.best_params_)
        else:
            model.fit(Xtr, y_train)
            best = model
    except Exception as e:
        st.write("Recherche hyperparam échouée — utilisation paramètres par défaut.", str(e))
        try:
            model.fit(Xtr, y_train)
            best = model
        except Exception as e2:
            st.write("Impossible d'entraîner ce modèle :", e2)
            continue

    try:
        best.fit(Xtr, y_train)
    except Exception:
        pass

    trained_models[name] = {'model': best, 'use_scaled': use_scaled}

    y_pred = best.predict(Xte)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    try:
        from sklearn.model_selection import cross_val_score
        cv_vals = cross_val_score(best, Xtr, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        cv_rmse = -cv_vals.mean()
        cv_std = cv_vals.std()
    except Exception:
        cv_rmse = np.nan
        cv_std = np.nan

    st.write(f"- Test RMSE: **{rmse:.4f}**  |  MAE: {mae:.4f}  |  R2: {r2:.4f}")
    st.write(f"- CV RMSE (5-fold): {cv_rmse:.4f} ± {cv_std:.4f}")

    fig = plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Observé (NOx)")
    plt.ylabel("Prédit (NOx)")
    plt.title(f"{name} — Observé vs Prédit (RMSE={rmse:.3f})")
    plt.grid(True)
    obs_pred_path = os.path.join(out_dir, f"obs_vs_pred_{name}.png")
    fig.savefig(obs_pred_path, dpi=150, bbox_inches='tight')
    saved_figs.append(obs_pred_path)
    st.pyplot(fig)
    plt.close(fig)

    resid = y_test - y_pred
    fig = plt.figure(figsize=(6,3))
    sns.histplot(resid, bins=30, kde=True)
    plt.title(f"{name} — Résidus (mean={resid.mean():.3f})")
    resid_path = os.path.join(out_dir, f"resid_{name}.png")
    fig.savefig(resid_path, dpi=150, bbox_inches='tight')
    saved_figs.append(resid_path)
    st.pyplot(fig)
    plt.close(fig)

    try:
        if hasattr(best, 'feature_importances_'):
            fi = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            st.write("Top features:")
            st.dataframe(fi.head(10).to_frame("importance"))
            fig = fi.head(10).plot.bar(figsize=(8,4)).get_figure()
            feat_path = os.path.join(out_dir, f"feat_imp_{name}.png")
            fig.savefig(feat_path, dpi=150, bbox_inches='tight')
            saved_figs.append(feat_path)
            plt.close(fig)
        elif hasattr(best, 'coef_'):
            coef = pd.Series(best.coef_.ravel(), index=X_train.columns).abs().sort_values(ascending=False)
            st.write("Coefficients principaux :")
            st.dataframe(coef.head(10).to_frame("coef_abs"))
            fig = coef.head(10).plot.bar(figsize=(8,4)).get_figure()
            coef_path = os.path.join(out_dir, f"coef_{name}.png")
            fig.savefig(coef_path, dpi=150, bbox_inches='tight')
            saved_figs.append(coef_path)
            plt.close(fig)
    except:
        pass

    results.append({
        'model': name,
        'rmse_test': rmse,
        'mae_test': mae,
        'r2_test': r2,
        'cv_rmse': cv_rmse,
        'cv_std': cv_std
    })

# ---------------------------
# COMPARATIVE TABLE
# ---------------------------
st.subheader(" Résumé comparatif des modèles")
results_df = pd.DataFrame(results).sort_values('rmse_test')
st.dataframe(results_df.reset_index(drop=True))

if results_df.empty:
    st.error("Aucun modèle n’a pu être entraîné.")
    st.stop()

best_row = results_df.iloc[0]
best_model_name = best_row['model']
st.success(f" Meilleur modèle : **{best_model_name}** — RMSE={best_row['rmse_test']:.4f}")

best_info = trained_models.get(best_model_name)
if best_info is None:
    st.error("Impossible de récupérer le meilleur modèle.")
    st.stop()

best_model = best_info['model']
use_scaled_best = best_info['use_scaled']

X_final = X_scaled if use_scaled_best else X_processed.values
try:
    best_model.fit(X_final, y)
except:
    try:
        best_model = models[best_model_name].__class__()
        best_model.fit(X_final, y)
    except Exception as e:
        st.warning("Réentraînement impossible:", e)

# ---------------------------
# OPTIMISATION
# ---------------------------
st.subheader(" Optimisation des variables pour minimiser NOx")

X_min = np.percentile(X_processed, 5, axis=0)
X_max = np.percentile(X_processed, 95, axis=0)
x0 = X_processed.mean(axis=0).values

def objective(x):
    x = np.array(x).reshape(1, -1)
    if use_scaled_best:
        x = scaler_model.transform(x)
    return float(best_model.predict(x)[0])

bounds = list(zip(X_min, X_max))
try:
    res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B', options={'maxiter':200})
    optimal_values = pd.Series(res.x, index=X_processed.columns)
    predicted_nox_min = res.fun
    st.write("Valeurs optimales estimées :")
    st.dataframe(optimal_values.to_frame("valeur_opt"))
    st.success(f" NOx minimal estimé : **{predicted_nox_min:.3f}**")
except Exception as e:
    st.error("Optimisation impossible :", e)
    optimal_values = None
    predicted_nox_min = None

# ---------------------------
# DOWNLOAD MODEL
# ---------------------------
st.subheader("Téléchargements")
model_bytes = io.BytesIO()
try:
    pickle.dump(best_model, model_bytes)
    model_bytes.seek(0)
    st.download_button(label=" Télécharger le modèle optimal (pickle)",
                       data=model_bytes,
                       file_name=f"best_model_{best_model_name}.pkl")
except Exception as e:
    st.warning("Impossible d'exporter le modèle :", e)

# ---------------------------
# PDF GENERATION
# ---------------------------
st.write("Générer un rapport PDF complet :")
generate_pdf = st.button(" Générer le PDF")

def build_pdf(saved_figs, results_df, optimal_values, predicted_nox_min):
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    width, height = A4
    margin = 40
    y_pos = height - margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y_pos, app_name)
    y_pos -= 30
    c.setFont("Helvetica", 12)
    c.drawString(margin, y_pos, f"Rapport généré le: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y_pos -= 25
    c.drawString(margin, y_pos, f"Fichier analysé: {getattr(uploaded_file, 'name', 'uploaded_data.csv')}")
    y_pos -= 40

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_pos, "Résumé des modèles")
    y_pos -= 20
    c.setFont("Helvetica", 10)

    for idx, row in results_df.head(6).iterrows():
        txt = f"{idx+1}. {row['model']}: RMSE={row['rmse_test']:.4f}, MAE={row['mae_test']:.4f}, R2={row['r2_test']:.4f}"
        c.drawString(margin, y_pos, txt)
        y_pos -= 14
        if y_pos < 120:
            c.showPage()
            y_pos = height - margin

    y_pos -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y_pos, f"Meilleur modèle : {best_model_name}")
    y_pos -= 20

    if optimal_values is not None:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_pos, "Valeurs optimales estimées :")
        y_pos -= 16
        c.setFont("Helvetica", 10)
        for i, (k, v) in enumerate(optimal_values.items()):
            c.drawString(margin+10, y_pos, f"- {k}: {v:.4f}")
            y_pos -= 12
            if y_pos < 120:
                c.showPage()
                y_pos = height - margin

    for img_path in saved_figs:
        try:
            c.showPage()
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            max_w = width - 2*margin
            max_h = height - 2*margin
            scale = min(max_w/iw, max_h/ih)
            draw_w = iw * scale
            draw_h = ih * scale
            x_img = (width - draw_w) / 2
            y_img = (height - draw_h) / 2
            c.drawImage(img, x_img, y_img, draw_w, draw_h)
        except:
            pass

    c.save()
    bio.seek(0)
    return bio

if generate_pdf:
    try:
        pdf_bytes = build_pdf(saved_figs, results_df, optimal_values, predicted_nox_min)
        st.download_button(" Télécharger le rapport PDF", data=pdf_bytes, file_name="nox_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error("Erreur PDF :", e)

# ---------------------------
# Final message
# ---------------------------
st.info("Analyse terminée — Vous pouvez télécharger le modèle optimal et le rapport PDF.")
