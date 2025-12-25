"""
    saa_dro.jl

Module principal pour l'algorithme SAA-DRO d'acquisition de caractéristiques
avec robustesse distributionnelle.

Auteur: Louck
Projet: IFT6512 - DRO avec données manquantes
"""

module SAADRO

include("scenario_generation.jl")
include("dro_objective.jl")
include("acquisition_solver.jl")

using .ScenarioGeneration
using .DROObjective
using .AcquisitionSolver
using Statistics
using Random

# Re-export from submodules
export DROConfig, DROResult, solve_dro_acquisition
export fit_and_solve!, evaluate_policy, cross_validate_epsilon
export LogisticModel, fit_logistic!, predict, logloss
export ConditionalGaussian, generate_gaussian_scenarios, estimate_gaussian_params
export solve_acquisition, solve_acquisition_exact, solve_acquisition_greedy
export evaluate_dro_objective, evaluate_dro_objective_detailed

#==============================================================================
# Structures de configuration et résultats
==============================================================================#

"""
    DROConfig

Configuration pour l'algorithme SAA-DRO.
"""
Base.@kwdef struct DROConfig
    # Paramètres DRO
    ε::Float64 = 0.1                    # Rayon d'ambiguïté Wasserstein
    p::Int = 1                          # Ordre de Wasserstein
    
    # Paramètres SAA
    n_scenarios::Int = 500              # Nombre de scénarios
    use_qmc::Bool = false               # Utiliser quasi-Monte Carlo
    
    # Paramètres d'optimisation
    exact_threshold::Int = 15           # Seuil pour énumération exacte
    
    # Paramètres du modèle
    λ_reg::Float64 = 0.01               # Régularisation L2 pour logistique
    
    # Options
    verbose::Bool = false
    seed::Union{Int, Nothing} = nothing
end

"""
    DROResult

Résultats de l'algorithme SAA-DRO.
"""
struct DROResult
    z_optimal::Vector{Bool}             # Décision d'acquisition optimale
    objective::Float64                  # Valeur de l'objectif DRO
    acquisition_cost::Float64           # Coût total d'acquisition
    method::Symbol                      # Méthode utilisée (:exact ou :greedy)
    n_acquired::Int                     # Nombre de caractéristiques acquises
    acquired_indices::Vector{Int}       # Indices des caractéristiques acquises
    computation_time::Float64           # Temps de calcul (secondes)
    
    # Détails supplémentaires
    avg_loss::Float64                   # Perte moyenne (sans terme robustesse)
    robustness_term::Float64            # Terme de robustesse L*ε^p
    lipschitz::Float64                  # Constante de Lipschitz estimée
end

#==============================================================================
# Algorithme principal
==============================================================================#

"""
    solve_dro_acquisition(model::LogisticModel, gaussian_params::ConditionalGaussian,
                         x_obs::Vector{Float64}, mask::Vector{Bool}, y::Real,
                         costs::Vector{Float64}, budget::Float64,
                         config::DROConfig) -> DROResult

Résout le problème d'acquisition DRO pour une observation donnée.

Arguments:
- model: Modèle logistique pré-entraîné
- gaussian_params: Paramètres pour l'imputation gaussienne
- x_obs: Vecteur d'observation (avec NaN pour les manquants)
- mask: Masque de manquement
- y: Vraie étiquette
- costs: Coûts d'acquisition par caractéristique
- budget: Budget total
- config: Configuration de l'algorithme

Retourne un DROResult.
"""
function solve_dro_acquisition(model::LogisticModel,
                              gaussian_params::ConditionalGaussian,
                              x_obs::Vector{Float64},
                              mask::Vector{Bool},
                              y::Real,
                              costs::Vector{Float64},
                              budget::Float64,
                              config::DROConfig)
    
    start_time = time()
    
    # Initialiser RNG si seed spécifiée
    rng = isnothing(config.seed) ? Random.GLOBAL_RNG : MersenneTwister(config.seed)
    
    if config.verbose
        println("=== SAA-DRO Acquisition ===")
        println("Caractéristiques manquantes: $(sum(mask))")
        println("Budget: $budget")
        println("Rayon ε: $(config.ε)")
        println("Nombre de scénarios: $(config.n_scenarios)")
    end
    
    # Générer les scénarios
    if config.use_qmc
        scenarios = generate_qmc_gaussian_scenarios(gaussian_params, x_obs, mask, 
                                                   config.n_scenarios)
    else
        scenarios = generate_gaussian_scenarios(gaussian_params, x_obs, mask,
                                               config.n_scenarios; rng=rng)
    end
    
    if config.verbose
        println("Scénarios générés: $(size(scenarios))")
    end
    
    # Définir la fonction d'évaluation
    function eval_func(z::Vector{Bool})
        return evaluate_dro_objective(model, scenarios, x_obs, mask, z, y, 
                                     config.ε; p=config.p)
    end
    
    # Résoudre le problème d'acquisition
    z_opt, obj_opt, method = solve_acquisition(eval_func, mask, costs, budget;
                                               exact_threshold=config.exact_threshold,
                                               verbose=config.verbose)
    
    # Calculer les détails
    details = evaluate_dro_objective_detailed(model, scenarios, x_obs, mask, 
                                             z_opt, y, config.ε; p=config.p)
    
    elapsed = time() - start_time
    
    # Construire le résultat
    result = DROResult(
        z_opt,
        obj_opt,
        AcquisitionSolver.acquisition_cost(z_opt, costs),
        method,
        sum(z_opt),
        findall(z_opt),
        elapsed,
        details.avg_loss,
        details.robustness_term,
        details.lipschitz
    )
    
    if config.verbose
        println("\n=== Résultat ===")
        println("Objectif DRO: $(round(obj_opt, digits=4))")
        println("Caractéristiques acquises: $(result.n_acquired)")
        println("Coût: $(result.acquisition_cost)")
        println("Méthode: $(method)")
        println("Temps: $(round(elapsed, digits=2))s")
    end
    
    return result
end

#==============================================================================
# Pipeline complet: entraînement + résolution
==============================================================================#

"""
    fit_and_solve!(X_train::Matrix{Float64}, y_train::Vector{Float64},
                  x_test::Vector{Float64}, mask::Vector{Bool}, y_test::Real,
                  costs::Vector{Float64}, budget::Float64,
                  config::DROConfig) -> (DROResult, LogisticModel, ConditionalGaussian)

Pipeline complet: entraîne le modèle, estime les paramètres d'imputation,
et résout le problème d'acquisition.

Retourne le résultat, le modèle entraîné, et les paramètres gaussiens.
"""
function fit_and_solve!(X_train::Matrix{Float64}, 
                       y_train::Vector{Float64},
                       x_test::Vector{Float64}, 
                       mask::Vector{Bool}, 
                       y_test::Real,
                       costs::Vector{Float64}, 
                       budget::Float64,
                       config::DROConfig)
    
    d = size(X_train, 2)
    
    if config.verbose
        println("=== Phase d'entraînement ===")
    end
    
    # Entraîner le modèle logistique
    model = LogisticModel(d)
    fit_logistic!(model, X_train, y_train; λ=config.λ_reg)
    
    if config.verbose
        train_pred = predict(model, X_train)
        train_acc = mean((train_pred .> 0.5) .== y_train)
        println("Accuracy entraînement: $(round(train_acc, digits=3))")
    end
    
    # Estimer les paramètres gaussiens
    gaussian_params = estimate_gaussian_params(X_train)
    
    if config.verbose
        println("Paramètres gaussiens estimés")
    end
    
    # Résoudre le problème d'acquisition
    result = solve_dro_acquisition(model, gaussian_params, x_test, mask, y_test,
                                  costs, budget, config)
    
    return result, model, gaussian_params
end

#==============================================================================
# Évaluation d'une politique
==============================================================================#

"""
    evaluate_policy(model::LogisticModel, gaussian_params::ConditionalGaussian,
                   X_test::Matrix{Float64}, y_test::Vector{Float64},
                   masks::Matrix{Bool}, costs::Vector{Float64}, budget::Float64,
                   config::DROConfig; n_eval::Int=100) -> NamedTuple

Évalue une politique d'acquisition sur un ensemble de test.

Retourne les métriques moyennes: accuracy, log-loss, coût moyen, etc.
"""
function evaluate_policy(model::LogisticModel,
                        gaussian_params::ConditionalGaussian,
                        X_test::Matrix{Float64},
                        y_test::Vector{Float64},
                        masks::Matrix{Bool},
                        costs::Vector{Float64},
                        budget::Float64,
                        config::DROConfig;
                        n_eval::Union{Int, Nothing}=nothing)
    
    n_test = size(X_test, 1)
    n_eval = isnothing(n_eval) ? n_test : min(n_eval, n_test)
    
    accuracies = Float64[]
    losses = Float64[]
    total_costs = Float64[]
    n_acquired_list = Int[]
    
    for i in 1:n_eval
        x_obs = X_test[i, :]
        y = y_test[i]
        mask = masks[i, :]
        
        # Résoudre pour cette observation
        result = solve_dro_acquisition(model, gaussian_params, x_obs, mask, y,
                                      costs, budget, config)
        
        # Évaluer avec les vraies valeurs
        x_true = X_test[i, :]  # Dans un vrai scénario, on révélerait les valeurs
        imputed = imputation_mean(gaussian_params, x_obs, mask)
        
        loss = evaluate_with_true_values(model, x_true, x_obs, mask, 
                                        result.z_optimal, imputed, y)
        
        # Prédiction finale
        x_final = copy(x_obs)
        idx_miss = findall(mask)
        for (local_idx, global_idx) in enumerate(idx_miss)
            if result.z_optimal[global_idx]
                x_final[global_idx] = x_true[global_idx]
            else
                x_final[global_idx] = imputed[local_idx]
            end
        end
        
        y_pred = predict(model, x_final)
        correct = (y_pred > 0.5) == (y > 0.5)
        
        push!(accuracies, correct ? 1.0 : 0.0)
        push!(losses, loss)
        push!(total_costs, result.acquisition_cost)
        push!(n_acquired_list, result.n_acquired)
    end
    
    return (
        accuracy = mean(accuracies),
        accuracy_std = std(accuracies),
        logloss = mean(losses),
        logloss_std = std(losses),
        avg_cost = mean(total_costs),
        avg_n_acquired = mean(n_acquired_list),
        n_evaluated = n_eval
    )
end

#==============================================================================
# Validation croisée pour ε
==============================================================================#

"""
    cross_validate_epsilon(X::Matrix{Float64}, y::Vector{Float64},
                          masks::Matrix{Bool}, costs::Vector{Float64}, 
                          budget::Float64, epsilon_grid::Vector{Float64};
                          n_folds::Int=5, config_base::DROConfig=DROConfig()) -> NamedTuple

Sélectionne le rayon ε optimal par validation croisée.

Retourne (ε_optimal, scores_par_epsilon, détails).
"""
function cross_validate_epsilon(X::Matrix{Float64}, 
                               y::Vector{Float64},
                               masks::Matrix{Bool}, 
                               costs::Vector{Float64}, 
                               budget::Float64,
                               epsilon_grid::Vector{Float64};
                               n_folds::Int=5,
                               config_base::DROConfig=DROConfig())
    
    n = size(X, 1)
    fold_size = n ÷ n_folds
    
    scores = Dict{Float64, Vector{Float64}}()
    
    for ε in epsilon_grid
        scores[ε] = Float64[]
    end
    
    for fold in 1:n_folds
        # Indices de validation
        val_start = (fold - 1) * fold_size + 1
        val_end = fold == n_folds ? n : fold * fold_size
        val_idx = val_start:val_end
        train_idx = setdiff(1:n, val_idx)
        
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val = X[val_idx, :]
        y_val = y[val_idx]
        masks_val = masks[val_idx, :]
        
        # Entraîner le modèle une fois
        d = size(X, 2)
        model = LogisticModel(d)
        fit_logistic!(model, X_train, y_train; λ=config_base.λ_reg)
        gaussian_params = estimate_gaussian_params(X_train)
        
        # Évaluer chaque ε
        for ε in epsilon_grid
            config = DROConfig(
                ε=ε,
                p=config_base.p,
                n_scenarios=config_base.n_scenarios,
                use_qmc=config_base.use_qmc,
                exact_threshold=config_base.exact_threshold,
                λ_reg=config_base.λ_reg,
                verbose=false
            )
            
            metrics = evaluate_policy(model, gaussian_params, X_val, y_val,
                                     masks_val, costs, budget, config)
            
            push!(scores[ε], metrics.logloss)
        end
    end
    
    # Calculer les scores moyens
    mean_scores = Dict(ε => mean(s) for (ε, s) in scores)
    std_scores = Dict(ε => std(s) for (ε, s) in scores)
    
    # Trouver le meilleur ε
    ε_optimal = argmin(mean_scores)
    
    return (
        ε_optimal = ε_optimal,
        mean_scores = mean_scores,
        std_scores = std_scores,
        all_scores = scores
    )
end

#==============================================================================
# Comparaison avec baselines
==============================================================================#

"""
    compare_with_baselines(model::LogisticModel, gaussian_params::ConditionalGaussian,
                          X_test::Matrix{Float64}, y_test::Vector{Float64},
                          masks::Matrix{Bool}, costs::Vector{Float64}, budget::Float64,
                          config::DROConfig) -> Dict

Compare DRO avec plusieurs baselines:
- no_acquisition: Imputation pure
- random: Acquisition aléatoire
- full: Acquisition complète (si budget le permet)
- greedy_nonrobust: Glouton sans robustesse (ε=0)
"""
function compare_with_baselines(model::LogisticModel,
                               gaussian_params::ConditionalGaussian,
                               X_test::Matrix{Float64},
                               y_test::Vector{Float64},
                               masks::Matrix{Bool},
                               costs::Vector{Float64},
                               budget::Float64,
                               config::DROConfig)
    
    results = Dict{String, Any}()
    
    # DRO
    results["dro"] = evaluate_policy(model, gaussian_params, X_test, y_test,
                                    masks, costs, budget, config)
    
    # Non-robuste (ε = 0)
    config_nonrobust = DROConfig(
        ε=0.0,
        n_scenarios=config.n_scenarios,
        exact_threshold=config.exact_threshold,
        verbose=false
    )
    results["nonrobust"] = evaluate_policy(model, gaussian_params, X_test, y_test,
                                          masks, costs, budget, config_nonrobust)
    
    # TODO: Implémenter les autres baselines (random, no_acquisition, full)
    
    return results
end

end # module
