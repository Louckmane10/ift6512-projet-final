"""
    dro_objective.jl

Module pour le calcul de l'objectif DRO (Distributionally Robust Optimization).
Implémente la reformulation duale avec ambiguïté de Wasserstein.

Auteur: Louck
Projet: IFT6512 - DRO avec données manquantes
"""

module DROObjective

using LinearAlgebra
using Statistics
using Optim

export evaluate_dro_objective, evaluate_dro_objective_detailed, evaluate_standard_objective
export LogisticModel, fit_logistic!, predict, logloss
export lipschitz_constant

#==============================================================================
# Modèle de régression logistique
==============================================================================#

"""
    LogisticModel

Structure pour un modèle de régression logistique.
"""
mutable struct LogisticModel
    θ::Vector{Float64}      # Coefficients (incluant le biais en position 1)
    d::Int                  # Dimension des features (sans biais)
    fitted::Bool
end

LogisticModel(d::Int) = LogisticModel(zeros(d + 1), d, false)

"""
    sigmoid(z)

Fonction sigmoïde avec protection contre l'overflow.
"""
function sigmoid(z::Real)
    if z >= 0
        return 1.0 / (1.0 + exp(-z))
    else
        ez = exp(z)
        return ez / (1.0 + ez)
    end
end

sigmoid(z::AbstractVector) = sigmoid.(z)

"""
    predict(model::LogisticModel, X::Matrix{Float64}) -> Vector{Float64}

Prédit les probabilités pour les observations X (n × d).
"""
function predict(model::LogisticModel, X::Matrix{Float64})
    n = size(X, 1)
    X_aug = hcat(ones(n), X)  # Ajouter colonne de 1 pour le biais
    return sigmoid.(X_aug * model.θ)
end

function predict(model::LogisticModel, x::Vector{Float64})
    x_aug = vcat(1.0, x)
    return sigmoid(dot(x_aug, model.θ))
end

"""
    logloss(y_true::Vector, y_pred::Vector) -> Float64

Calcule la log-loss moyenne.
"""
function logloss(y_true::AbstractVector, y_pred::AbstractVector)
    ε = 1e-15  # Pour éviter log(0)
    y_pred_clipped = clamp.(y_pred, ε, 1 - ε)
    return -mean(y_true .* log.(y_pred_clipped) .+ (1 .- y_true) .* log.(1 .- y_pred_clipped))
end

function logloss(y_true::Real, y_pred::Real)
    ε = 1e-15
    y_pred_clipped = clamp(y_pred, ε, 1 - ε)
    return -(y_true * log(y_pred_clipped) + (1 - y_true) * log(1 - y_pred_clipped))
end

"""
    fit_logistic!(model::LogisticModel, X::Matrix{Float64}, y::Vector{Float64};
                  λ::Float64=0.01, max_iter::Int=1000)

Entraîne le modèle par descente de gradient avec régularisation L2.
"""
function fit_logistic!(model::LogisticModel, X::Matrix{Float64}, y::Vector{Float64};
                       λ::Float64=0.01, max_iter::Int=1000, tol::Float64=1e-6)
    
    n, d = size(X)
    @assert d == model.d "Dimension mismatch"
    
    X_aug = hcat(ones(n), X)
    
    # Fonction objectif (log-loss + régularisation L2)
    function objective(θ)
        p = sigmoid.(X_aug * θ)
        ll = -mean(y .* log.(p .+ 1e-15) .+ (1 .- y) .* log.(1 .- p .+ 1e-15))
        reg = 0.5 * λ * sum(θ[2:end].^2)  # Ne pas régulariser le biais
        return ll + reg
    end
    
    # Gradient
    function gradient!(G, θ)
        p = sigmoid.(X_aug * θ)
        G .= X_aug' * (p - y) / n
        G[2:end] .+= λ * θ[2:end]  # Gradient de la régularisation
    end
    
    # Optimisation
    result = optimize(objective, gradient!, model.θ, LBFGS(),
                     Optim.Options(iterations=max_iter, g_tol=tol))
    
    model.θ = Optim.minimizer(result)
    model.fitted = true
    
    return model
end

#==============================================================================
# Calcul de la constante de Lipschitz
==============================================================================#

"""
    lipschitz_constant(model::LogisticModel, X_bounds::Tuple{Vector{Float64}, Vector{Float64}})

Estime la constante de Lipschitz de la log-loss composée avec le modèle logistique
sur un domaine borné [X_min, X_max].

Pour la log-loss, le gradient par rapport à x est:
∂ℓ/∂x = (p - y) * θ
où p = σ(θᵀx). La norme est maximale quand |p - y| est maximal (= 1) et
quand on utilise la norme de θ.
"""
function lipschitz_constant(model::LogisticModel; 
                           p_min::Float64=0.01, p_max::Float64=0.99)
    
    # Simplification: pour la log-loss avec régression logistique,
    # la constante de Lipschitz est approximativement ||θ|| / (p_min * (1-p_min))
    # où p_min est la plus petite probabilité prédite.
    
    θ_norm = norm(model.θ[2:end])  # Exclure le biais
    
    # Le gradient de la log-loss est borné par:
    # ||∇_x ℓ|| ≤ ||θ|| * max(1/p, 1/(1-p))
    L = θ_norm * max(1/p_min, 1/(1-p_max))
    
    return L
end

#==============================================================================
# Évaluation de l'objectif DRO
==============================================================================#

"""
    evaluate_dro_objective(model::LogisticModel, scenarios::Matrix{Float64},
                          x_obs::Vector{Float64}, mask::Vector{Bool},
                          z::Vector{Bool}, y::Real, ε::Float64;
                          p::Int=1) -> Float64

Évalue l'objectif DRO pour une décision d'acquisition z donnée.

Arguments:
- model: Modèle de prédiction
- scenarios: Scénarios des valeurs manquantes (N × d_miss)
- x_obs: Observation originale
- mask: Masque de manquement
- z: Décision d'acquisition (z[j] = true si on acquiert j)
- y: Vraie étiquette
- ε: Rayon de l'ensemble d'ambiguïté de Wasserstein
- p: Ordre de la distance de Wasserstein (défaut: 1)

Retourne l'estimation SAA de l'objectif DRO.
"""
function evaluate_dro_objective(model::LogisticModel,
                               scenarios::Matrix{Float64},
                               x_obs::Vector{Float64},
                               mask::Vector{Bool},
                               z::Vector{Bool},
                               y::Real,
                               ε::Float64;
                               p::Int=1)
    
    N = size(scenarios, 1)
    idx_miss = findall(mask)
    d_miss = length(idx_miss)
    
    # Calculer la constante de Lipschitz
    L = lipschitz_constant(model)
    
    # Pour λ ≥ L, le supremum dans la reformulation duale est atteint en ξ' = ξ
    # Donc: R_DRO = L * ε^p + (1/N) * Σ ℓ(y, f(x̃(z, ξ^k)))
    
    total_loss = 0.0
    
    for k in 1:N
        # Construire le vecteur reconstruit x̃
        x_tilde = copy(x_obs)
        
        for (local_idx, global_idx) in enumerate(idx_miss)
            if z[global_idx]
                # Acquis: on utilise la vraie valeur (simulée par le scénario)
                x_tilde[global_idx] = scenarios[k, local_idx]
            else
                # Imputé: on utilise la moyenne des scénarios
                x_tilde[global_idx] = mean(scenarios[:, local_idx])
            end
        end
        
        # Calculer la perte
        y_pred = predict(model, x_tilde)
        total_loss += logloss(y, y_pred)
    end
    
    avg_loss = total_loss / N
    
    # Objectif DRO = terme de robustesse + perte moyenne
    dro_objective = L * ε^p + avg_loss
    
    return dro_objective
end

"""
    evaluate_dro_objective_detailed(model::LogisticModel, scenarios::Matrix{Float64},
                                   x_obs::Vector{Float64}, mask::Vector{Bool},
                                   z::Vector{Bool}, y::Real, ε::Float64;
                                   p::Int=1) -> NamedTuple

Version détaillée qui retourne aussi les composantes de l'objectif.
"""
function evaluate_dro_objective_detailed(model::LogisticModel,
                                        scenarios::Matrix{Float64},
                                        x_obs::Vector{Float64},
                                        mask::Vector{Bool},
                                        z::Vector{Bool},
                                        y::Real,
                                        ε::Float64;
                                        p::Int=1)
    
    N = size(scenarios, 1)
    idx_miss = findall(mask)
    
    L = lipschitz_constant(model)
    
    losses = zeros(N)
    
    for k in 1:N
        x_tilde = copy(x_obs)
        
        for (local_idx, global_idx) in enumerate(idx_miss)
            if z[global_idx]
                x_tilde[global_idx] = scenarios[k, local_idx]
            else
                x_tilde[global_idx] = mean(scenarios[:, local_idx])
            end
        end
        
        y_pred = predict(model, x_tilde)
        losses[k] = logloss(y, y_pred)
    end
    
    avg_loss = mean(losses)
    robustness_term = L * ε^p
    dro_objective = robustness_term + avg_loss
    
    return (
        dro_objective = dro_objective,
        avg_loss = avg_loss,
        robustness_term = robustness_term,
        lipschitz = L,
        loss_std = std(losses),
        losses = losses
    )
end

"""
    evaluate_standard_objective(model::LogisticModel, scenarios::Matrix{Float64},
                               x_obs::Vector{Float64}, mask::Vector{Bool},
                               z::Vector{Bool}, y::Real) -> Float64

Évalue l'objectif stochastique standard (non robuste, ε = 0).
"""
function evaluate_standard_objective(model::LogisticModel,
                                    scenarios::Matrix{Float64},
                                    x_obs::Vector{Float64},
                                    mask::Vector{Bool},
                                    z::Vector{Bool},
                                    y::Real)
    
    return evaluate_dro_objective(model, scenarios, x_obs, mask, z, y, 0.0)
end

#==============================================================================
# Évaluation avec acquisition révélée (pour le test)
==============================================================================#

"""
    evaluate_with_true_values(model::LogisticModel, x_true::Vector{Float64},
                             x_obs::Vector{Float64}, mask::Vector{Bool},
                             z::Vector{Bool}, imputed_values::Vector{Float64},
                             y::Real) -> Float64

Évalue la perte avec les vraies valeurs révélées pour les caractéristiques acquises.
Utilisé pour l'évaluation finale après la décision d'acquisition.
"""
function evaluate_with_true_values(model::LogisticModel,
                                  x_true::Vector{Float64},
                                  x_obs::Vector{Float64},
                                  mask::Vector{Bool},
                                  z::Vector{Bool},
                                  imputed_values::Vector{Float64},
                                  y::Real)
    
    idx_miss = findall(mask)
    x_tilde = copy(x_obs)
    
    for (local_idx, global_idx) in enumerate(idx_miss)
        if z[global_idx]
            # Acquis: vraie valeur
            x_tilde[global_idx] = x_true[global_idx]
        else
            # Imputé
            x_tilde[global_idx] = imputed_values[local_idx]
        end
    end
    
    y_pred = predict(model, x_tilde)
    return logloss(y, y_pred)
end

end # module
