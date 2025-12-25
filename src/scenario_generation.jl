"""
    scenario_generation.jl

Module pour la génération de scénarios des valeurs manquantes.
Implémente l'imputation conditionnelle gaussienne et MICE.

Auteur: Louck
Projet: IFT6512 - DRO avec données manquantes
"""

module ScenarioGeneration

using LinearAlgebra
using Statistics
using Distributions
using Random

export generate_gaussian_scenarios, generate_mice_scenarios, estimate_gaussian_params
export ConditionalGaussian, fit!, sample

#==============================================================================
# Structure pour l'imputation gaussienne conditionnelle
==============================================================================#

"""
    ConditionalGaussian

Structure pour stocker les paramètres d'une distribution gaussienne
et effectuer l'imputation conditionnelle.
"""
struct ConditionalGaussian
    μ::Vector{Float64}      # Moyenne globale
    Σ::Matrix{Float64}      # Covariance globale
    d::Int                  # Dimension
end

"""
    fit!(data::Matrix{Float64}) -> ConditionalGaussian

Estime les paramètres gaussiens à partir des données complètes.
Les lignes sont les observations, les colonnes sont les variables.
"""
function fit(data::Matrix{Float64})
    n, d = size(data)
    μ = vec(mean(data, dims=1))
    Σ = cov(data)
    
    # Régularisation pour éviter les matrices singulières
    Σ = Σ + 1e-6 * I(d)
    
    return ConditionalGaussian(μ, Σ, d)
end

"""
    conditional_params(cg::ConditionalGaussian, x_obs::Vector{Float64}, 
                       idx_obs::Vector{Int}, idx_miss::Vector{Int})

Calcule les paramètres conditionnels μ_cond et Σ_cond pour les valeurs manquantes
sachant les valeurs observées.

Retourne (μ_cond, Σ_cond, L_cond) où L_cond est la factorisation de Cholesky.
"""
function conditional_params(cg::ConditionalGaussian, 
                           x_obs::Vector{Float64},
                           idx_obs::Vector{Int}, 
                           idx_miss::Vector{Int})
    
    # Extraire les sous-vecteurs et sous-matrices
    μ_obs = cg.μ[idx_obs]
    μ_miss = cg.μ[idx_miss]
    
    Σ_oo = cg.Σ[idx_obs, idx_obs]
    Σ_mo = cg.Σ[idx_miss, idx_obs]
    Σ_mm = cg.Σ[idx_miss, idx_miss]
    
    # Calcul des paramètres conditionnels
    # μ_cond = μ_miss + Σ_mo * Σ_oo^{-1} * (x_obs - μ_obs)
    # Σ_cond = Σ_mm - Σ_mo * Σ_oo^{-1} * Σ_om
    
    Σ_oo_inv = inv(Σ_oo)
    μ_cond = μ_miss + Σ_mo * Σ_oo_inv * (x_obs - μ_obs)
    Σ_cond = Σ_mm - Σ_mo * Σ_oo_inv * Σ_mo'
    
    # Régularisation et Cholesky
    Σ_cond = Symmetric(Σ_cond)
    Σ_cond = Σ_cond + 1e-8 * I(length(idx_miss))
    L_cond = cholesky(Σ_cond).L
    
    return μ_cond, Σ_cond, L_cond
end

"""
    generate_gaussian_scenarios(cg::ConditionalGaussian, x_obs::Vector{Float64},
                                mask::Vector{Bool}, N::Int; 
                                rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Génère N scénarios pour les valeurs manquantes par imputation gaussienne conditionnelle.

Arguments:
- cg: Paramètres gaussiens estimés
- x_obs: Vecteur d'observation (valeurs observées aux positions mask .== false)
- mask: Masque de manquement (true = manquant)
- N: Nombre de scénarios à générer
- rng: Générateur de nombres aléatoires

Retourne une matrice N × d_miss où chaque ligne est un scénario.
"""
function generate_gaussian_scenarios(cg::ConditionalGaussian,
                                    x_obs::Vector{Float64},
                                    mask::Vector{Bool},
                                    N::Int;
                                    rng=Random.GLOBAL_RNG)
    
    idx_obs = findall(.!mask)
    idx_miss = findall(mask)
    d_miss = length(idx_miss)
    
    if d_miss == 0
        return zeros(N, 0)
    end
    
    # Extraire les valeurs observées
    x_obs_values = x_obs[idx_obs]
    
    # Calculer les paramètres conditionnels
    μ_cond, Σ_cond, L_cond = conditional_params(cg, x_obs_values, idx_obs, idx_miss)
    
    # Générer les scénarios
    scenarios = zeros(N, d_miss)
    for k in 1:N
        u = randn(rng, d_miss)
        scenarios[k, :] = μ_cond + L_cond * u
    end
    
    return scenarios
end

#==============================================================================
# Imputation par MICE (Multiple Imputation by Chained Equations)
==============================================================================#

"""
    mice_single_imputation(data::Matrix{Float64}, mask_matrix::Matrix{Bool},
                          n_iter::Int; rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Effectue une imputation MICE sur les données.

Arguments:
- data: Matrice de données (n × d) avec des NaN pour les valeurs manquantes
- mask_matrix: Matrice de masques (true = manquant)
- n_iter: Nombre d'itérations MICE
- rng: Générateur aléatoire

Retourne les données imputées.
"""
function mice_single_imputation(data::Matrix{Float64}, 
                               mask_matrix::Matrix{Bool},
                               n_iter::Int;
                               rng=Random.GLOBAL_RNG)
    
    n, d = size(data)
    imputed = copy(data)
    
    # Initialisation par la moyenne des valeurs observées
    for j in 1:d
        obs_vals = data[.!mask_matrix[:, j], j]
        if !isempty(obs_vals)
            mean_j = mean(obs_vals)
            imputed[mask_matrix[:, j], j] .= mean_j
        end
    end
    
    # Itérations MICE
    for iter in 1:n_iter
        for j in 1:d
            miss_idx = findall(mask_matrix[:, j])
            if isempty(miss_idx)
                continue
            end
            
            obs_idx = findall(.!mask_matrix[:, j])
            if isempty(obs_idx)
                continue
            end
            
            # Régression de X_j sur X_{-j}
            X_other = imputed[:, setdiff(1:d, j)]
            y_j = imputed[:, j]
            
            # Fit sur les observations complètes pour cette variable
            X_train = X_other[obs_idx, :]
            y_train = y_j[obs_idx]
            
            # Régression linéaire simple (avec régularisation ridge)
            X_train_aug = hcat(ones(length(obs_idx)), X_train)
            λ_ridge = 0.01
            β = (X_train_aug' * X_train_aug + λ_ridge * I) \ (X_train_aug' * y_train)
            
            # Prédiction pour les valeurs manquantes
            X_miss = X_other[miss_idx, :]
            X_miss_aug = hcat(ones(length(miss_idx)), X_miss)
            y_pred = X_miss_aug * β
            
            # Ajouter du bruit (variance résiduelle)
            residuals = y_train - X_train_aug * β
            σ_res = std(residuals)
            noise = randn(rng, length(miss_idx)) * σ_res
            
            imputed[miss_idx, j] = y_pred + noise
        end
    end
    
    return imputed
end

"""
    generate_mice_scenarios(data::Matrix{Float64}, x_obs::Vector{Float64},
                           mask::Vector{Bool}, N::Int, n_iter::Int=10;
                           rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Génère N scénarios par MICE pour une nouvelle observation.

Arguments:
- data: Données d'entraînement complètes
- x_obs: Nouvelle observation avec valeurs manquantes
- mask: Masque de manquement
- N: Nombre de scénarios
- n_iter: Nombre d'itérations MICE

Retourne une matrice N × d_miss.
"""
function generate_mice_scenarios(data::Matrix{Float64},
                                x_obs::Vector{Float64},
                                mask::Vector{Bool},
                                N::Int,
                                n_iter::Int=10;
                                rng=Random.GLOBAL_RNG)
    
    idx_miss = findall(mask)
    d_miss = length(idx_miss)
    
    if d_miss == 0
        return zeros(N, 0)
    end
    
    scenarios = zeros(N, d_miss)
    
    for k in 1:N
        # Créer un dataset augmenté avec la nouvelle observation
        data_aug = vcat(data, x_obs')
        mask_aug = vcat(falses(size(data, 1), length(mask)), mask')
        
        # Imputer
        imputed = mice_single_imputation(data_aug, mask_aug, n_iter; rng=rng)
        
        # Extraire les valeurs imputées pour la nouvelle observation
        scenarios[k, :] = imputed[end, idx_miss]
    end
    
    return scenarios
end

#==============================================================================
# Quasi-Monte Carlo (séquences de Sobol)
==============================================================================#

"""
    sobol_sequence(N::Int, d::Int) -> Matrix{Float64}

Génère une séquence de Sobol de N points en dimension d.
Implémentation simplifiée - pour une vraie application, utiliser Sobol.jl
"""
function sobol_sequence(N::Int, d::Int)
    # Implémentation simplifiée via Halton
    # Pour une vraie implémentation Sobol, utiliser le package Sobol.jl
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    points = zeros(N, d)
    for j in 1:d
        p = primes[mod(j-1, length(primes)) + 1]
        for i in 1:N
            points[i, j] = halton_element(i, p)
        end
    end
    
    return points
end

function halton_element(i::Int, base::Int)
    result = 0.0
    f = 1.0 / base
    n = i
    while n > 0
        result += f * (n % base)
        n = div(n, base)
        f /= base
    end
    return result
end

"""
    generate_qmc_gaussian_scenarios(cg::ConditionalGaussian, x_obs::Vector{Float64},
                                   mask::Vector{Bool}, N::Int) -> Matrix{Float64}

Génère N scénarios par quasi-Monte Carlo (séquences à faible discrépance).
"""
function generate_qmc_gaussian_scenarios(cg::ConditionalGaussian,
                                        x_obs::Vector{Float64},
                                        mask::Vector{Bool},
                                        N::Int)
    
    idx_obs = findall(.!mask)
    idx_miss = findall(mask)
    d_miss = length(idx_miss)
    
    if d_miss == 0
        return zeros(N, 0)
    end
    
    x_obs_values = x_obs[idx_obs]
    μ_cond, Σ_cond, L_cond = conditional_params(cg, x_obs_values, idx_obs, idx_miss)
    
    # Générer points QMC dans [0,1]^d_miss
    U = sobol_sequence(N, d_miss)
    
    # Transformer en gaussien via quantile
    Z = quantile.(Normal(), U)
    
    # Transformer en distribution conditionnelle
    scenarios = zeros(N, d_miss)
    for k in 1:N
        scenarios[k, :] = μ_cond + L_cond * Z[k, :]
    end
    
    return scenarios
end

#==============================================================================
# Utilitaires
==============================================================================#

"""
    estimate_gaussian_params(data::Matrix{Float64}) -> ConditionalGaussian

Wrapper pour estimer les paramètres gaussiens.
"""
function estimate_gaussian_params(data::Matrix{Float64})
    return fit(data)
end

"""
    imputation_mean(cg::ConditionalGaussian, x_obs::Vector{Float64},
                   mask::Vector{Bool}) -> Vector{Float64}

Retourne l'imputation par la moyenne conditionnelle (point estimate).
"""
function imputation_mean(cg::ConditionalGaussian,
                        x_obs::Vector{Float64},
                        mask::Vector{Bool})
    
    idx_obs = findall(.!mask)
    idx_miss = findall(mask)
    
    if isempty(idx_miss)
        return Float64[]
    end
    
    x_obs_values = x_obs[idx_obs]
    μ_cond, _, _ = conditional_params(cg, x_obs_values, idx_obs, idx_miss)
    
    return μ_cond
end

end # module
