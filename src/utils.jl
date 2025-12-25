"""
    utils.jl

Utilitaires pour le projet DRO: chargement de données, simulation de mécanismes
de manquement, et fonctions d'évaluation.

Auteur: Louck
Projet: IFT6512 - DRO avec données manquantes
"""

module Utils

using Statistics
using Random
using DelimitedFiles

export load_pima_diabetes, load_heart_disease, download_uci_data
export simulate_mcar, simulate_mar, simulate_mnar, simulate_mnar_threshold
export train_test_split, train_val_test_split, normalize_features, apply_normalization
export compute_metrics, print_results_table
export DatasetInfo

#==============================================================================
# Structure d'information sur les datasets
==============================================================================#

"""
    DatasetInfo

Structure contenant les informations d'un dataset.
"""
struct DatasetInfo
    X::Matrix{Float64}
    y::Vector{Float64}
    feature_names::Vector{String}
    costs::Vector{Float64}
    protected_cols::Vector{Int}  # Colonnes toujours observées (coût 0)
    name::String
end

#==============================================================================
# Chargement des données
==============================================================================#

"""
    load_pima_diabetes(; use_real_data::Bool=true, data_dir::String="data") -> DatasetInfo

Charge le dataset Pima Indians Diabetes.
Si use_real_data=true, tente de charger depuis un fichier CSV.
Sinon, génère des données simulées réalistes.
"""
function load_pima_diabetes(; use_real_data::Bool=true, data_dir::String="data")
    
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigree", "Age"]
    
    # Coûts simulés (basés sur le type d'examen médical)
    # 0 = gratuit (anamnèse), 2-8 = examen simple, 10-15 = test sanguin
    costs = [0.0, 10.0, 5.0, 8.0, 15.0, 2.0, 0.0, 0.0]
    
    # Colonnes toujours observées (coût 0)
    protected_cols = [1, 7, 8]
    
    # Essayer de charger les vraies données
    csv_path = joinpath(data_dir, "pima-indians-diabetes.csv")
    
    if use_real_data && isfile(csv_path)
        data = readdlm(csv_path, ',', Float64, skipstart=1)
        X = data[:, 1:8]
        y = data[:, 9]
        
        # Traiter les valeurs aberrantes (0 pour Glucose, BP, etc.)
        # Ces 0 sont des valeurs manquantes codées
        for j in [2, 3, 4, 5, 6]  # Glucose, BP, Skin, Insulin, BMI
            zero_mask = X[:, j] .== 0
            if any(zero_mask)
                # Remplacer par NaN pour l'instant (seront gérés comme manquants)
                col_mean = mean(X[.!zero_mask, j])
                X[zero_mask, j] .= col_mean  # Imputation simple pour l'entraînement
            end
        end
    else
        # Données simulées réalistes
        n = 768
        rng = MersenneTwister(42)
        
        X = zeros(n, 8)
        
        # Générer des caractéristiques réalistes
        X[:, 1] = round.(abs.(randn(rng, n)) .* 3)           # Pregnancies (0-17)
        X[:, 2] = 120 .+ randn(rng, n) .* 32                  # Glucose (~89-199)
        X[:, 3] = 69 .+ randn(rng, n) .* 19                   # BloodPressure (~24-122)
        X[:, 4] = max.(0, 20 .+ randn(rng, n) .* 16)          # SkinThickness (0-99)
        X[:, 5] = max.(0, 80 .+ randn(rng, n) .* 115)         # Insulin (0-846)
        X[:, 6] = 32 .+ randn(rng, n) .* 8                    # BMI (~18-67)
        X[:, 7] = max.(0.08, 0.47 .+ randn(rng, n) .* 0.33)   # DiabetesPedigree
        X[:, 8] = max.(21, 33 .+ randn(rng, n) .* 12)         # Age (21-81)
        
        # Générer y avec une relation réaliste
        logits = -8.0 .+ 
                 0.035 .* X[:, 2] .+    # Glucose (fort effet)
                 0.089 .* X[:, 6] .+    # BMI
                 0.042 .* X[:, 8] .+    # Age
                 1.82 .* X[:, 7] .+     # DiabetesPedigree
                 0.15 .* X[:, 1]        # Pregnancies
        
        probs = 1 ./ (1 .+ exp.(-logits))
        y = Float64.(rand(rng, n) .< probs)
        
        @info "Données Pima simulées générées" n=n prevalence=mean(y)
    end
    
    return DatasetInfo(X, y, feature_names, costs, protected_cols, "Pima Indians Diabetes")
end

"""
    load_heart_disease(; use_real_data::Bool=true, data_dir::String="data") -> DatasetInfo

Charge le dataset Heart Disease Cleveland.
"""
function load_heart_disease(; use_real_data::Bool=true, data_dir::String="data")
    
    feature_names = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
                    "Oldpeak", "ST_Slope", "NumMajorVessels", "Thal"]
    
    # Coûts basés sur le type d'examen
    costs = [0.0,   # Age - anamnèse
            0.0,   # Sex - observation
            5.0,   # ChestPainType - examen clinique
            5.0,   # RestingBP - tensiomètre
            12.0,  # Cholesterol - test sanguin
            8.0,   # FastingBS - test sanguin
            15.0,  # RestingECG - ECG
            10.0,  # MaxHR - test d'effort
            5.0,   # ExerciseAngina - test d'effort
            10.0,  # Oldpeak - ECG d'effort
            15.0,  # ST_Slope - ECG d'effort
            25.0,  # NumMajorVessels - fluoroscopie
            20.0]  # Thal - scintigraphie
    
    protected_cols = [1, 2]  # Age et Sex toujours observés
    
    csv_path = joinpath(data_dir, "heart-disease-cleveland.csv")
    
    if use_real_data && isfile(csv_path)
        data = readdlm(csv_path, ',', Float64, skipstart=1)
        X = data[:, 1:13]
        y = Float64.(data[:, 14] .> 0)  # Binariser (0 = pas de maladie, >0 = maladie)
    else
        # Données simulées
        n = 303
        rng = MersenneTwister(123)
        
        X = zeros(n, 13)
        
        X[:, 1] = 54 .+ randn(rng, n) .* 9           # Age
        X[:, 2] = Float64.(rand(rng, n) .< 0.68)    # Sex (68% hommes)
        X[:, 3] = rand(rng, 0:3, n)                  # ChestPainType
        X[:, 4] = 131 .+ randn(rng, n) .* 18        # RestingBP
        X[:, 5] = 246 .+ randn(rng, n) .* 52        # Cholesterol
        X[:, 6] = Float64.(rand(rng, n) .< 0.15)   # FastingBS
        X[:, 7] = rand(rng, 0:2, n)                  # RestingECG
        X[:, 8] = 150 .+ randn(rng, n) .* 23        # MaxHR
        X[:, 9] = Float64.(rand(rng, n) .< 0.33)   # ExerciseAngina
        X[:, 10] = max.(0, 1.0 .+ randn(rng, n) .* 1.2)  # Oldpeak
        X[:, 11] = rand(rng, 0:2, n)                # ST_Slope
        X[:, 12] = rand(rng, 0:3, n)                # NumMajorVessels
        X[:, 13] = rand(rng, [3, 6, 7], n)         # Thal
        
        # Générer y
        logits = -5.0 .+
                 0.05 .* X[:, 1] .+     # Age
                 0.8 .* X[:, 2] .+      # Sex
                 0.5 .* X[:, 3] .+      # ChestPainType
                 0.01 .* X[:, 5] .+     # Cholesterol
                 1.5 .* X[:, 9] .+      # ExerciseAngina
                 0.5 .* X[:, 10]        # Oldpeak
        
        probs = 1 ./ (1 .+ exp.(-logits))
        y = Float64.(rand(rng, n) .< probs)
        
        @info "Données Heart Disease simulées générées" n=n prevalence=mean(y)
    end
    
    return DatasetInfo(X, y, feature_names, costs, protected_cols, "Heart Disease Cleveland")
end

"""
    download_uci_data(data_dir::String="data")

Télécharge les datasets UCI si non présents.
Note: Nécessite une connexion internet.
"""
function download_uci_data(data_dir::String="data")
    isdir(data_dir) || mkdir(data_dir)
    
    # URLs des datasets (à adapter selon disponibilité)
    datasets = [
        ("pima-indians-diabetes.csv", 
         "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"),
        ("heart-disease-cleveland.csv",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
    ]
    
    for (filename, url) in datasets
        filepath = joinpath(data_dir, filename)
        if !isfile(filepath)
            @info "Téléchargement de $filename..."
            try
                download(url, filepath)
                @info "  Téléchargé avec succès"
            catch e
                @warn "  Échec du téléchargement: $e"
            end
        end
    end
end

#==============================================================================
# Simulation des mécanismes de manquement
==============================================================================#

"""
    simulate_mcar(X::Matrix{Float64}, p_miss::Float64; 
                 protected_cols::Vector{Int}=Int[], rng=Random.GLOBAL_RNG) -> Matrix{Bool}

Simule un mécanisme MCAR (Missing Completely At Random).
Chaque valeur est manquante avec probabilité p_miss.

Arguments:
- X: Matrice de données (n × d)
- p_miss: Probabilité de manquement
- protected_cols: Colonnes qui ne peuvent pas être manquantes
- rng: Générateur aléatoire

Retourne une matrice de masques (true = manquant).
"""
function simulate_mcar(X::Matrix{Float64}, p_miss::Float64;
                      protected_cols::Vector{Int}=Int[],
                      rng=Random.GLOBAL_RNG)
    
    n, d = size(X)
    mask = rand(rng, n, d) .< p_miss
    
    # Protéger certaines colonnes
    for j in protected_cols
        mask[:, j] .= false
    end
    
    return mask
end

"""
    simulate_mar(X::Matrix{Float64}, p_miss::Float64, obs_col::Int;
                α::Float64=-1.5, β::Float64=1.0,
                protected_cols::Vector{Int}=Int[], rng=Random.GLOBAL_RNG) -> Matrix{Bool}

Simule un mécanisme MAR (Missing At Random).
La probabilité de manquement dépend d'une variable observée (typiquement l'âge).

P(M_j = 1 | X_{obs_col}) = σ(α + β * (X_{obs_col} - μ) / σ)

Les paramètres α et β sont ajustés pour obtenir environ p_miss de manquement.
"""
function simulate_mar(X::Matrix{Float64}, p_miss::Float64, obs_col::Int;
                     α::Float64=-1.5, β::Float64=1.0,
                     protected_cols::Vector{Int}=Int[],
                     rng=Random.GLOBAL_RNG)
    
    n, d = size(X)
    mask = falses(n, d)
    
    # Variable conditionnante normalisée
    x_obs = X[:, obs_col]
    x_normalized = (x_obs .- mean(x_obs)) ./ (std(x_obs) + 1e-8)
    
    # Probabilité de base
    base_probs = 1 ./ (1 .+ exp.(-(α .+ β .* x_normalized)))
    
    # Ajuster pour avoir environ p_miss en moyenne
    current_mean = mean(base_probs)
    if current_mean > 0
        adjustment = p_miss / current_mean
        probs = clamp.(base_probs .* adjustment, 0.0, 0.95)
    else
        probs = fill(p_miss, n)
    end
    
    for j in 1:d
        if j == obs_col || j in protected_cols
            continue
        end
        mask[:, j] = rand(rng, n) .< probs
    end
    
    return mask
end

"""
    simulate_mnar(X::Matrix{Float64}, p_miss::Float64;
                 α::Float64=-1.5, β::Float64=0.8,
                 protected_cols::Vector{Int}=Int[], rng=Random.GLOBAL_RNG) -> Matrix{Bool}

Simule un mécanisme MNAR (Missing Not At Random).
La probabilité de manquement dépend de la valeur elle-même.

P(M_j = 1 | X_j) = σ(α + β * (X_j - μ_j) / σ_j)

Arguments:
- β > 0: les grandes valeurs sont plus souvent manquantes (MNAR+)
- β < 0: les petites valeurs sont plus souvent manquantes (MNAR-)
"""
function simulate_mnar(X::Matrix{Float64}, p_miss::Float64;
                      α::Float64=-1.5, β::Float64=0.8,
                      protected_cols::Vector{Int}=Int[],
                      rng=Random.GLOBAL_RNG)
    
    n, d = size(X)
    mask = falses(n, d)
    
    for j in 1:d
        if j in protected_cols
            continue
        end
        
        x_j = X[:, j]
        x_normalized = (x_j .- mean(x_j)) ./ (std(x_j) + 1e-8)
        base_probs = 1 ./ (1 .+ exp.(-(α .+ β .* x_normalized)))
        
        # Ajuster pour la proportion cible
        current_mean = mean(base_probs)
        if current_mean > 0
            adjustment = p_miss / current_mean
            probs = clamp.(base_probs .* adjustment, 0.0, 0.95)
        else
            probs = fill(p_miss, n)
        end
        
        mask[:, j] = rand(rng, n) .< probs
    end
    
    return mask
end

"""
    simulate_mnar_threshold(X::Matrix{Float64}, p_miss::Float64; 
                           threshold_quantile::Float64=0.7,
                           p_low::Float64=0.1, p_high::Float64=0.5,
                           protected_cols::Vector{Int}=Int[], 
                           rng=Random.GLOBAL_RNG) -> Matrix{Bool}

MNAR avec effet de seuil: les valeurs au-dessus du quantile sont plus souvent manquantes.
Modélise le cas où le médecin prescrit un test seulement si un score dépasse un seuil.
"""
function simulate_mnar_threshold(X::Matrix{Float64}, p_miss::Float64; 
                                threshold_quantile::Float64=0.7,
                                p_low::Float64=0.1, p_high::Float64=0.5,
                                protected_cols::Vector{Int}=Int[],
                                rng=Random.GLOBAL_RNG)
    
    n, d = size(X)
    mask = falses(n, d)
    
    for j in 1:d
        if j in protected_cols
            continue
        end
        
        x_j = X[:, j]
        threshold = quantile(x_j, threshold_quantile)
        
        # Probabilité élevée au-dessus du seuil
        probs = ifelse.(x_j .> threshold, p_high, p_low)
        
        mask[:, j] = rand(rng, n) .< probs
    end
    
    return mask
end

#==============================================================================
# Prétraitement
==============================================================================#

"""
    train_test_split(X::Matrix{Float64}, y::Vector{Float64}, test_ratio::Float64=0.2;
                    stratify::Bool=true, rng=Random.GLOBAL_RNG)

Sépare les données en ensembles d'entraînement et de test.
Si stratify=true, maintient les proportions de classes.
"""
function train_test_split(X::Matrix{Float64}, y::Vector{Float64}, 
                         test_ratio::Float64=0.2;
                         stratify::Bool=true,
                         rng=Random.GLOBAL_RNG)
    
    n = size(X, 1)
    
    if stratify
        # Stratification
        idx_pos = findall(y .== 1)
        idx_neg = findall(y .== 0)
        
        n_test_pos = round(Int, length(idx_pos) * test_ratio)
        n_test_neg = round(Int, length(idx_neg) * test_ratio)
        
        shuffle!(rng, idx_pos)
        shuffle!(rng, idx_neg)
        
        test_idx = vcat(idx_pos[1:n_test_pos], idx_neg[1:n_test_neg])
        train_idx = vcat(idx_pos[n_test_pos+1:end], idx_neg[n_test_neg+1:end])
        
        shuffle!(rng, test_idx)
        shuffle!(rng, train_idx)
    else
        indices = shuffle(rng, 1:n)
        n_test = round(Int, n * test_ratio)
        test_idx = indices[1:n_test]
        train_idx = indices[n_test+1:end]
    end
    
    return X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
end

"""
    train_val_test_split(X::Matrix{Float64}, y::Vector{Float64};
                        train_ratio::Float64=0.6, val_ratio::Float64=0.2,
                        stratify::Bool=true, rng=Random.GLOBAL_RNG)

Sépare les données en trois ensembles: train, validation, test.
"""
function train_val_test_split(X::Matrix{Float64}, y::Vector{Float64};
                             train_ratio::Float64=0.6, val_ratio::Float64=0.2,
                             stratify::Bool=true, rng=Random.GLOBAL_RNG)
    
    test_ratio = 1.0 - train_ratio - val_ratio
    @assert test_ratio > 0 "Les ratios doivent sommer à moins de 1"
    
    # Premier split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_ratio; stratify=stratify, rng=rng)
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, val_ratio_adjusted; stratify=stratify, rng=rng)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
end

"""
    normalize_features(X::Matrix{Float64}; return_params::Bool=false)

Normalise les features (moyenne 0, écart-type 1).
"""
function normalize_features(X::Matrix{Float64}; return_params::Bool=false)
    μ = vec(mean(X, dims=1))
    σ = vec(std(X, dims=1))
    σ[σ .< 1e-8] .= 1.0  # Éviter division par zéro
    
    X_norm = (X .- μ') ./ σ'
    
    if return_params
        return X_norm, μ, σ
    else
        return X_norm
    end
end

"""
    apply_normalization(X::Matrix{Float64}, μ::Vector{Float64}, σ::Vector{Float64})

Applique une normalisation avec des paramètres pré-calculés.
"""
function apply_normalization(X::Matrix{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    return (X .- μ') ./ σ'
end

#==============================================================================
# Métriques et affichage
==============================================================================#

"""
    compute_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64}) -> NamedTuple

Calcule les métriques de classification complètes.
"""
function compute_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})
    n = length(y_true)
    
    # Prédictions binaires
    y_class = Float64.(y_pred .> 0.5)
    
    # Accuracy
    accuracy = mean(y_class .== y_true)
    
    # Log-loss
    ε = 1e-15
    y_pred_clipped = clamp.(y_pred, ε, 1-ε)
    logloss = -mean(y_true .* log.(y_pred_clipped) .+ (1 .- y_true) .* log.(1 .- y_pred_clipped))
    
    # Confusion matrix
    tp = sum((y_class .== 1) .& (y_true .== 1))
    tn = sum((y_class .== 0) .& (y_true .== 0))
    fp = sum((y_class .== 1) .& (y_true .== 0))
    fn = sum((y_class .== 0) .& (y_true .== 1))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Specificity
    specificity = tn / (tn + fp + 1e-10)
    
    # AUC approximation (simple)
    auc = compute_auc(y_true, y_pred)
    
    # Brier score
    brier = mean((y_pred .- y_true).^2)
    
    return (
        accuracy = accuracy,
        logloss = logloss,
        precision = precision,
        recall = recall,
        f1 = f1,
        specificity = specificity,
        auc = auc,
        brier = brier,
        tp = tp, tn = tn, fp = fp, fn = fn
    )
end

"""
    compute_auc(y_true::Vector{Float64}, y_pred::Vector{Float64}) -> Float64

Calcule l'AUC-ROC de façon exacte.
"""
function compute_auc(y_true::Vector{Float64}, y_pred::Vector{Float64})
    n_pos = sum(y_true .== 1)
    n_neg = sum(y_true .== 0)
    
    if n_pos == 0 || n_neg == 0
        return 0.5
    end
    
    # Trier par score décroissant
    order = sortperm(y_pred, rev=true)
    y_sorted = y_true[order]
    
    # Compter les paires concordantes
    n_concordant = 0.0
    cumsum_neg = 0.0
    
    for i in 1:length(y_sorted)
        if y_sorted[i] == 0
            cumsum_neg += 1
        else
            n_concordant += cumsum_neg
        end
    end
    
    # AUC = P(score_pos > score_neg)
    # Attention: on veut les positifs AVANT les négatifs dans le tri décroissant
    auc = 1.0 - n_concordant / (n_pos * n_neg)
    
    return auc
end

"""
    print_results_table(results::Dict; methods::Vector{String}=String[])

Affiche un tableau formaté des résultats.
"""
function print_results_table(results::Dict; methods::Vector{String}=String[])
    if isempty(methods)
        methods = sort(collect(keys(results)))
    end
    
    println("\n" * "="^80)
    println("RÉSULTATS EXPÉRIMENTAUX")
    println("="^80)
    
    # Header
    println(@sprintf("%-20s %10s %10s %10s %10s %10s", 
                    "Méthode", "Accuracy", "LogLoss", "AUC", "Coût moy.", "N acquis"))
    println("-"^80)
    
    for method in methods
        if haskey(results, method)
            r = results[method]
            if haskey(r, :accuracy)
                println(@sprintf("%-20s %10.4f %10.4f %10.4f %10.2f %10.1f",
                       method, 
                       get(r, :accuracy, 0.0),
                       get(r, :logloss, 0.0),
                       get(r, :auc, 0.0),
                       get(r, :avg_cost, 0.0),
                       get(r, :avg_n_acquired, 0.0)))
            end
        end
    end
    
    println("="^80)
end

"""
    print_robustness_table(results_by_mechanism::Dict, methods::Vector{String})

Affiche un tableau de robustesse comparant les méthodes sous différents mécanismes.
"""
function print_robustness_table(results_by_mechanism::Dict, methods::Vector{String})
    mechanisms = sort(collect(keys(results_by_mechanism)))
    
    println("\n" * "="^90)
    println("ANALYSE DE ROBUSTESSE (Log-Loss par mécanisme)")
    println("="^90)
    
    # Header
    print(@sprintf("%-15s", "Méthode"))
    for mech in mechanisms
        print(@sprintf(" %12s", mech))
    end
    println(@sprintf(" %12s %12s", "Moyenne", "Écart"))
    println("-"^90)
    
    for method in methods
        print(@sprintf("%-15s", method))
        losses = Float64[]
        
        for mech in mechanisms
            if haskey(results_by_mechanism[mech], method)
                loss = results_by_mechanism[mech][method].logloss
                push!(losses, loss)
                print(@sprintf(" %12.4f", loss))
            else
                print(@sprintf(" %12s", "N/A"))
            end
        end
        
        if !isempty(losses)
            println(@sprintf(" %12.4f %12.4f", mean(losses), maximum(losses) - minimum(losses)))
        else
            println()
        end
    end
    
    println("="^90)
end

# Import Printf pour le formatage
using Printf

end # module
