"""
    baselines.jl

Module contenant les méthodes de comparaison (baselines) pour le projet DRO.

Auteur: Louck
Projet: IFT6512 - DRO avec données manquantes
"""

module Baselines

using Statistics
using Random

export BaselineResult
export no_acquisition_baseline, random_acquisition_baseline
export full_acquisition_baseline, oracle_baseline
export evaluate_all_baselines

#==============================================================================
# Structure de résultat
==============================================================================#

"""
    BaselineResult

Structure pour stocker les résultats d'une baseline.
"""
struct BaselineResult
    z::Vector{Bool}           # Décision d'acquisition
    prediction::Float64       # Prédiction (probabilité)
    loss::Float64             # Log-loss
    cost::Float64             # Coût d'acquisition
    n_acquired::Int           # Nombre de caractéristiques acquises
    correct::Bool             # Classification correcte?
end

#==============================================================================
# Baseline 1: Pas d'acquisition (imputation pure)
==============================================================================#

"""
    no_acquisition_baseline(model, x_obs::Vector{Float64}, mask::Vector{Bool},
                           imputed_values::Vector{Float64}, y::Real,
                           costs::Vector{Float64}) -> BaselineResult

Baseline sans acquisition: impute toutes les valeurs manquantes.
"""
function no_acquisition_baseline(model, 
                                x_obs::Vector{Float64}, 
                                mask::Vector{Bool},
                                imputed_values::Vector{Float64}, 
                                y::Real,
                                costs::Vector{Float64})
    
    d = length(x_obs)
    z = falses(d)  # Pas d'acquisition
    
    # Construire le vecteur avec imputation
    x_final = copy(x_obs)
    idx_miss = findall(mask)
    for (local_idx, global_idx) in enumerate(idx_miss)
        x_final[global_idx] = imputed_values[local_idx]
    end
    
    # Prédiction
    y_pred = predict_prob(model, x_final)
    loss = logloss_single(y, y_pred)
    correct = (y_pred > 0.5) == (y > 0.5)
    
    return BaselineResult(z, y_pred, loss, 0.0, 0, correct)
end

#==============================================================================
# Baseline 2: Acquisition aléatoire
==============================================================================#

"""
    random_acquisition_baseline(model, x_obs::Vector{Float64}, x_true::Vector{Float64},
                               mask::Vector{Bool}, imputed_values::Vector{Float64},
                               y::Real, costs::Vector{Float64}, budget::Float64;
                               rng=Random.GLOBAL_RNG) -> BaselineResult

Baseline avec acquisition aléatoire jusqu'au budget.
"""
function random_acquisition_baseline(model, 
                                    x_obs::Vector{Float64},
                                    x_true::Vector{Float64},
                                    mask::Vector{Bool},
                                    imputed_values::Vector{Float64},
                                    y::Real,
                                    costs::Vector{Float64},
                                    budget::Float64;
                                    rng=Random.GLOBAL_RNG)
    
    d = length(x_obs)
    z = falses(d)
    
    # Indices des caractéristiques manquantes, permutés aléatoirement
    idx_miss = findall(mask)
    shuffle!(rng, idx_miss)
    
    # Acquérir dans l'ordre aléatoire jusqu'au budget
    remaining_budget = budget
    for j in idx_miss
        if costs[j] <= remaining_budget
            z[j] = true
            remaining_budget -= costs[j]
        end
    end
    
    # Construire le vecteur final
    x_final = copy(x_obs)
    for (local_idx, global_idx) in enumerate(findall(mask))
        if z[global_idx]
            x_final[global_idx] = x_true[global_idx]  # Vraie valeur
        else
            x_final[global_idx] = imputed_values[local_idx]  # Imputé
        end
    end
    
    # Prédiction
    y_pred = predict_prob(model, x_final)
    loss = logloss_single(y, y_pred)
    correct = (y_pred > 0.5) == (y > 0.5)
    total_cost = sum(costs[j] for j in 1:d if z[j]; init=0.0)
    
    return BaselineResult(z, y_pred, loss, total_cost, sum(z), correct)
end

#==============================================================================
# Baseline 3: Acquisition complète
==============================================================================#

"""
    full_acquisition_baseline(model, x_obs::Vector{Float64}, x_true::Vector{Float64},
                             mask::Vector{Bool}, y::Real, costs::Vector{Float64},
                             budget::Float64) -> BaselineResult

Baseline avec acquisition complète (toutes les caractéristiques manquantes si budget le permet).
Si le budget est insuffisant, acquiert les moins chères d'abord.
"""
function full_acquisition_baseline(model, 
                                  x_obs::Vector{Float64},
                                  x_true::Vector{Float64},
                                  mask::Vector{Bool},
                                  y::Real,
                                  costs::Vector{Float64},
                                  budget::Float64)
    
    d = length(x_obs)
    z = falses(d)
    
    # Indices manquants triés par coût croissant
    idx_miss = findall(mask)
    sort!(idx_miss, by=j -> costs[j])
    
    # Acquérir par ordre de coût croissant
    remaining_budget = budget
    for j in idx_miss
        if costs[j] <= remaining_budget
            z[j] = true
            remaining_budget -= costs[j]
        end
    end
    
    # Construire le vecteur final
    x_final = copy(x_obs)
    for global_idx in idx_miss
        if z[global_idx]
            x_final[global_idx] = x_true[global_idx]
        else
            # Si pas acquis, utiliser la moyenne (simple)
            x_final[global_idx] = mean(x_true)  # Approximation
        end
    end
    
    # Prédiction
    y_pred = predict_prob(model, x_final)
    loss = logloss_single(y, y_pred)
    correct = (y_pred > 0.5) == (y > 0.5)
    total_cost = sum(costs[j] for j in 1:d if z[j]; init=0.0)
    
    return BaselineResult(z, y_pred, loss, total_cost, sum(z), correct)
end

#==============================================================================
# Baseline 4: Oracle (toutes les vraies valeurs)
==============================================================================#

"""
    oracle_baseline(model, x_true::Vector{Float64}, y::Real) -> BaselineResult

Oracle qui connaît toutes les vraies valeurs (borne supérieure de performance).
"""
function oracle_baseline(model, x_true::Vector{Float64}, y::Real)
    
    y_pred = predict_prob(model, x_true)
    loss = logloss_single(y, y_pred)
    correct = (y_pred > 0.5) == (y > 0.5)
    
    d = length(x_true)
    return BaselineResult(falses(d), y_pred, loss, 0.0, 0, correct)
end

#==============================================================================
# Baseline 5: Glouton par ratio coût (sans robustesse)
==============================================================================#

"""
    greedy_cost_ratio_baseline(model, gaussian_params, x_obs::Vector{Float64},
                              x_true::Vector{Float64}, mask::Vector{Bool},
                              scenarios::Matrix{Float64}, y::Real,
                              costs::Vector{Float64}, budget::Float64) -> BaselineResult

Baseline gloutonne basée sur le ratio valeur/coût estimé, sans robustesse (ε=0).
"""
function greedy_cost_ratio_baseline(model, 
                                   x_obs::Vector{Float64},
                                   x_true::Vector{Float64},
                                   mask::Vector{Bool},
                                   imputed_values::Vector{Float64},
                                   y::Real,
                                   costs::Vector{Float64},
                                   budget::Float64)
    
    d = length(x_obs)
    z = falses(d)
    idx_miss = findall(mask)
    
    remaining_budget = budget
    
    while remaining_budget > 0
        # Candidats
        candidates = [j for j in idx_miss if !z[j] && costs[j] <= remaining_budget]
        
        if isempty(candidates)
            break
        end
        
        # Calculer la valeur marginale pour chaque candidat
        # Valeur = réduction de perte espérée
        best_j = -1
        best_ratio = -Inf
        
        x_current = copy(x_obs)
        for (local_idx, global_idx) in enumerate(idx_miss)
            if z[global_idx]
                x_current[global_idx] = x_true[global_idx]
            else
                x_current[global_idx] = imputed_values[local_idx]
            end
        end
        loss_current = logloss_single(y, predict_prob(model, x_current))
        
        for j in candidates
            # Perte si on acquiert j
            x_with_j = copy(x_current)
            x_with_j[j] = x_true[j]
            loss_with_j = logloss_single(y, predict_prob(model, x_with_j))
            
            value = loss_current - loss_with_j
            ratio = value / costs[j]
            
            if ratio > best_ratio
                best_ratio = ratio
                best_j = j
            end
        end
        
        # Si pas d'amélioration, arrêter
        if best_ratio <= 0
            break
        end
        
        z[best_j] = true
        remaining_budget -= costs[best_j]
    end
    
    # Construire le vecteur final
    x_final = copy(x_obs)
    for (local_idx, global_idx) in enumerate(idx_miss)
        if z[global_idx]
            x_final[global_idx] = x_true[global_idx]
        else
            x_final[global_idx] = imputed_values[local_idx]
        end
    end
    
    y_pred = predict_prob(model, x_final)
    loss = logloss_single(y, y_pred)
    correct = (y_pred > 0.5) == (y > 0.5)
    total_cost = sum(costs[j] for j in 1:d if z[j]; init=0.0)
    
    return BaselineResult(z, y_pred, loss, total_cost, sum(z), correct)
end

#==============================================================================
# Utilitaires
==============================================================================#

"""
Fonction de prédiction générique (à adapter selon le modèle).
"""
function predict_prob(model, x::Vector{Float64})
    # Suppose que le modèle a une méthode predict
    # Pour un modèle logistique simple:
    if hasfield(typeof(model), :θ)
        x_aug = vcat(1.0, x)
        z = dot(x_aug, model.θ)
        return 1.0 / (1.0 + exp(-z))
    else
        # Fallback
        return 0.5
    end
end

using LinearAlgebra: dot

"""
Log-loss pour une seule observation.
"""
function logloss_single(y_true::Real, y_pred::Real)
    ε = 1e-15
    y_pred_clipped = clamp(y_pred, ε, 1 - ε)
    return -(y_true * log(y_pred_clipped) + (1 - y_true) * log(1 - y_pred_clipped))
end

#==============================================================================
# Évaluation groupée
==============================================================================#

"""
    evaluate_all_baselines(model, X_test::Matrix{Float64}, y_test::Vector{Float64},
                          masks::Matrix{Bool}, imputation_func::Function,
                          costs::Vector{Float64}, budget::Float64;
                          n_eval::Int=100, rng=Random.GLOBAL_RNG) -> Dict

Évalue toutes les baselines sur un ensemble de test.
"""
function evaluate_all_baselines(model,
                               X_test::Matrix{Float64},
                               y_test::Vector{Float64},
                               masks::Matrix{Bool},
                               imputation_func::Function,
                               costs::Vector{Float64},
                               budget::Float64;
                               n_eval::Union{Int, Nothing}=nothing,
                               rng=Random.GLOBAL_RNG)
    
    n_test = size(X_test, 1)
    n_eval = isnothing(n_eval) ? n_test : min(n_eval, n_test)
    
    # Accumulateurs pour chaque méthode
    methods = ["NoAcq", "Random", "Full", "Greedy", "Oracle"]
    results = Dict(m => (
        accuracies = Float64[],
        losses = Float64[],
        costs = Float64[],
        n_acquired = Int[]
    ) for m in methods)
    
    for i in 1:n_eval
        x_obs = X_test[i, :]
        x_true = X_test[i, :]  # En test, on révèle les vraies valeurs
        y = y_test[i]
        mask = masks[i, :]
        
        # Imputation
        imputed = imputation_func(x_obs, mask)
        
        # Baselines
        r_no = no_acquisition_baseline(model, x_obs, mask, imputed, y, costs)
        push!(results["NoAcq"].accuracies, r_no.correct ? 1.0 : 0.0)
        push!(results["NoAcq"].losses, r_no.loss)
        push!(results["NoAcq"].costs, r_no.cost)
        push!(results["NoAcq"].n_acquired, r_no.n_acquired)
        
        r_rand = random_acquisition_baseline(model, x_obs, x_true, mask, imputed, 
                                            y, costs, budget; rng=rng)
        push!(results["Random"].accuracies, r_rand.correct ? 1.0 : 0.0)
        push!(results["Random"].losses, r_rand.loss)
        push!(results["Random"].costs, r_rand.cost)
        push!(results["Random"].n_acquired, r_rand.n_acquired)
        
        r_full = full_acquisition_baseline(model, x_obs, x_true, mask, y, costs, budget)
        push!(results["Full"].accuracies, r_full.correct ? 1.0 : 0.0)
        push!(results["Full"].losses, r_full.loss)
        push!(results["Full"].costs, r_full.cost)
        push!(results["Full"].n_acquired, r_full.n_acquired)
        
        r_greedy = greedy_cost_ratio_baseline(model, x_obs, x_true, mask, imputed,
                                             y, costs, budget)
        push!(results["Greedy"].accuracies, r_greedy.correct ? 1.0 : 0.0)
        push!(results["Greedy"].losses, r_greedy.loss)
        push!(results["Greedy"].costs, r_greedy.cost)
        push!(results["Greedy"].n_acquired, r_greedy.n_acquired)
        
        r_oracle = oracle_baseline(model, x_true, y)
        push!(results["Oracle"].accuracies, r_oracle.correct ? 1.0 : 0.0)
        push!(results["Oracle"].losses, r_oracle.loss)
        push!(results["Oracle"].costs, 0.0)
        push!(results["Oracle"].n_acquired, 0)
    end
    
    # Calculer les moyennes
    summary = Dict{String, NamedTuple}()
    for m in methods
        r = results[m]
        summary[m] = (
            accuracy = mean(r.accuracies),
            accuracy_std = std(r.accuracies),
            logloss = mean(r.losses),
            logloss_std = std(r.losses),
            avg_cost = mean(r.costs),
            avg_n_acquired = mean(r.n_acquired),
            n_evaluated = n_eval
        )
    end
    
    return summary
end

end # module
