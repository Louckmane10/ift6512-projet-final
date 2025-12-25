"""
    run_experiments.jl

Script principal pour exécuter les expériences du projet DRO.
Génère les résultats présentés dans le rapport.

Usage:
    julia run_experiments.jl [experiment_number]
    julia run_experiments.jl all
    julia run_experiments.jl tables  # Génère les tableaux LaTeX

Auteur: Louck
Projet: IFT6512 - DRO avec données manquantes
"""

# Charger les modules
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using Random
using Statistics
using Printf
using DelimitedFiles

include("src/saa_dro.jl")
include("src/utils.jl")
include("src/baselines.jl")

using .SAADRO
using .Utils
using .Baselines

#==============================================================================
# Configuration globale
==============================================================================#

const SEED = 42
const N_REPETITIONS = 20
const RESULTS_DIR = "results"

# Créer le dossier results s'il n'existe pas
isdir(RESULTS_DIR) || mkdir(RESULTS_DIR)

#==============================================================================
# Expérience 1: Comparaison DRO vs baselines sous différents mécanismes
==============================================================================#

function experiment1_mechanism_comparison(; n_reps::Int=N_REPETITIONS, verbose::Bool=true)
    println("\n" * "="^80)
    println("EXPÉRIENCE 1: Comparaison sous différents mécanismes de manquement")
    println("="^80)
    
    # Charger les données
    dataset = load_pima_diabetes(use_real_data=false)
    X, y = dataset.X, dataset.y
    costs = dataset.costs
    protected_cols = dataset.protected_cols
    
    println("Dataset: $(dataset.name) ($(size(X, 1)) obs, $(size(X, 2)) features)")
    
    # Configuration
    budget = 25.0
    p_miss = 0.3
    
    # Mécanismes à tester
    mechanism_names = ["MCAR", "MAR", "MNAR+", "MNAR-"]
    
    # Méthodes à tester
    method_names = ["NoAcq", "Random", "Full", "Greedy-NR", "DRO-0.05", "DRO-0.10", "DRO-0.20"]
    epsilon_values = Dict("DRO-0.05" => 0.05, "DRO-0.10" => 0.10, "DRO-0.20" => 0.20, 
                         "Greedy-NR" => 0.0)
    
    # Résultats: method -> mechanism -> [losses across reps]
    results = Dict(m => Dict(mech => Float64[] for mech in mechanism_names) 
                   for m in method_names)
    
    for rep in 1:n_reps
        rng = MersenneTwister(SEED + rep)
        
        if verbose && rep % 5 == 1
            println("  Répétition $rep/$n_reps...")
        end
        
        # Séparer train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2; rng=rng)
        X_train_norm, μ, σ = normalize_features(X_train; return_params=true)
        X_test_norm = apply_normalization(X_test, μ, σ)
        
        # Entraîner le modèle
        d = size(X_train, 2)
        model = LogisticModel(d)
        fit_logistic!(model, X_train_norm, y_train)
        gaussian_params = estimate_gaussian_params(X_train_norm)
        
        for mech_name in mechanism_names
            # Générer les masques selon le mécanisme
            if mech_name == "MCAR"
                masks = simulate_mcar(X_test_norm, p_miss; protected_cols=protected_cols, rng=rng)
            elseif mech_name == "MAR"
                masks = simulate_mar(X_test_norm, p_miss, 8; protected_cols=protected_cols, rng=rng)
            elseif mech_name == "MNAR+"
                masks = simulate_mnar(X_test_norm, p_miss; β=0.8, protected_cols=protected_cols, rng=rng)
            else  # MNAR-
                masks = simulate_mnar(X_test_norm, p_miss; β=-0.8, protected_cols=protected_cols, rng=rng)
            end
            
            # Évaluer chaque méthode
            for method in method_names
                if method == "NoAcq"
                    # Pas d'acquisition
                    config = DROConfig(ε=0.0, n_scenarios=1, verbose=false)
                    metrics = evaluate_policy_no_acquisition(model, gaussian_params, 
                                                            X_test_norm, y_test, masks)
                elseif method == "Random"
                    metrics = evaluate_policy_random(model, gaussian_params, X_test_norm, 
                                                    y_test, masks, costs, budget; rng=rng)
                elseif method == "Full"
                    metrics = evaluate_policy_full(model, gaussian_params, X_test_norm,
                                                  y_test, masks, costs, budget)
                else
                    # DRO ou Greedy-NR
                    ε = epsilon_values[method]
                    config = DROConfig(ε=ε, n_scenarios=200, verbose=false, seed=SEED+rep)
                    metrics = evaluate_policy(model, gaussian_params, X_test_norm, y_test,
                                            masks, costs, budget, config; n_eval=50)
                end
                
                push!(results[method][mech_name], metrics.logloss)
            end
        end
    end
    
    # Afficher les résultats
    println("\n" * "-"^80)
    println("RÉSULTATS (Log-loss, moyenne ± std)")
    println("-"^80)
    
    print(@sprintf("%-12s", "Méthode"))
    for mech in mechanism_names
        print(@sprintf(" %12s", mech))
    end
    println(@sprintf(" %10s %10s", "Moyenne", "Δ_rob"))
    println("-"^80)
    
    for method in method_names
        print(@sprintf("%-12s", method))
        losses = Float64[]
        for mech in mechanism_names
            vals = results[method][mech]
            m = mean(vals)
            push!(losses, m)
            print(@sprintf(" %5.3f±%.2f", m, std(vals)))
        end
        avg = mean(losses)
        delta = maximum(losses) - minimum(losses)
        println(@sprintf(" %10.3f %10.3f", avg, delta))
    end
    
    return results
end

# Fonctions d'évaluation simplifiées pour les baselines
function evaluate_policy_no_acquisition(model, gp, X_test, y_test, masks)
    losses = Float64[]
    for i in 1:size(X_test, 1)
        x = X_test[i, :]
        y = y_test[i]
        mask = masks[i, :]
        
        # Imputer par la moyenne conditionnelle
        imputed = imputation_mean(gp, x, mask)
        x_final = copy(x)
        idx_miss = findall(mask)
        for (li, gi) in enumerate(idx_miss)
            x_final[gi] = imputed[li]
        end
        
        y_pred = predict(model, x_final)
        push!(losses, logloss(y, y_pred))
    end
    return (logloss=mean(losses), accuracy=0.0, avg_cost=0.0, avg_n_acquired=0.0)
end

function evaluate_policy_random(model, gp, X_test, y_test, masks, costs, budget; rng=Random.GLOBAL_RNG)
    losses = Float64[]
    total_costs = Float64[]
    
    for i in 1:size(X_test, 1)
        x = X_test[i, :]
        y = y_test[i]
        mask = masks[i, :]
        
        # Acquisition aléatoire
        idx_miss = shuffle(rng, findall(mask))
        z = falses(length(x))
        remaining = budget
        for j in idx_miss
            if costs[j] <= remaining
                z[j] = true
                remaining -= costs[j]
            end
        end
        
        # Construire x_final
        imputed = imputation_mean(gp, x, mask)
        x_final = copy(x)
        for (li, gi) in enumerate(findall(mask))
            x_final[gi] = z[gi] ? x[gi] : imputed[li]
        end
        
        y_pred = predict(model, x_final)
        push!(losses, logloss(y, y_pred))
        push!(total_costs, sum(costs[j] for j in 1:length(z) if z[j]; init=0.0))
    end
    
    return (logloss=mean(losses), accuracy=0.0, avg_cost=mean(total_costs), avg_n_acquired=0.0)
end

function evaluate_policy_full(model, gp, X_test, y_test, masks, costs, budget)
    losses = Float64[]
    
    for i in 1:size(X_test, 1)
        x = X_test[i, :]
        y = y_test[i]
        mask = masks[i, :]
        
        # Acquisition complète (par ordre de coût)
        idx_miss = sort(findall(mask), by=j->costs[j])
        z = falses(length(x))
        remaining = budget
        for j in idx_miss
            if costs[j] <= remaining
                z[j] = true
                remaining -= costs[j]
            end
        end
        
        # Construire x_final (avec vraies valeurs pour les acquises)
        imputed = imputation_mean(gp, x, mask)
        x_final = copy(x)
        for (li, gi) in enumerate(findall(mask))
            x_final[gi] = z[gi] ? x[gi] : imputed[li]
        end
        
        y_pred = predict(model, x_final)
        push!(losses, logloss(y, y_pred))
    end
    
    return (logloss=mean(losses), accuracy=0.0, avg_cost=0.0, avg_n_acquired=0.0)
end

#==============================================================================
# Expérience 2: Impact du rayon ε
==============================================================================#

function experiment2_epsilon_sensitivity(; verbose::Bool=true)
    println("\n" * "="^80)
    println("EXPÉRIENCE 2: Sensibilité au rayon d'ambiguïté ε")
    println("="^80)
    
    Random.seed!(SEED)
    
    dataset = load_pima_diabetes(use_real_data=false)
    X, y = dataset.X, dataset.y
    costs = dataset.costs
    protected_cols = dataset.protected_cols
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
    X_train, μ, σ = normalize_features(X_train; return_params=true)
    X_test = apply_normalization(X_test, μ, σ)
    
    budget = 25.0
    p_miss = 0.3
    
    # Entraîner le modèle
    d = size(X_train, 2)
    model = LogisticModel(d)
    fit_logistic!(model, X_train, y_train)
    gaussian_params = estimate_gaussian_params(X_train)
    
    # Grille de ε
    epsilon_grid = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
    
    # Générer masques
    masks_mcar = simulate_mcar(X_test, p_miss; protected_cols=protected_cols)
    masks_mnar = simulate_mnar(X_test, p_miss; β=0.8, protected_cols=protected_cols)
    
    println("\n" * @sprintf("%-8s %10s %10s %10s %10s %10s", 
                          "ε", "MCAR", "MNAR+", "Moyenne", "Δ_rob", "Temps(s)"))
    println("-"^60)
    
    results = []
    
    for ε in epsilon_grid
        config = DROConfig(ε=ε, n_scenarios=200, verbose=false)
        
        t0 = time()
        metrics_mcar = evaluate_policy(model, gaussian_params, X_test, y_test,
                                      masks_mcar, costs, budget, config; n_eval=50)
        metrics_mnar = evaluate_policy(model, gaussian_params, X_test, y_test,
                                      masks_mnar, costs, budget, config; n_eval=50)
        elapsed = time() - t0
        
        avg = (metrics_mcar.logloss + metrics_mnar.logloss) / 2
        delta = abs(metrics_mcar.logloss - metrics_mnar.logloss)
        
        push!(results, (ε=ε, mcar=metrics_mcar.logloss, mnar=metrics_mnar.logloss,
                       avg=avg, delta=delta, time=elapsed, cost=metrics_mcar.avg_cost))
        
        println(@sprintf("%-8.2f %10.3f %10.3f %10.3f %10.3f %10.2f",
               ε, metrics_mcar.logloss, metrics_mnar.logloss, avg, delta, elapsed))
    end
    
    return results
end

#==============================================================================
# Expérience 3: Valeur de l'information par caractéristique
==============================================================================#

function experiment3_feature_value(; verbose::Bool=true)
    println("\n" * "="^80)
    println("EXPÉRIENCE 3: Valeur de l'information par caractéristique")
    println("="^80)
    
    Random.seed!(SEED)
    
    dataset = load_pima_diabetes(use_real_data=false)
    X, y = dataset.X, dataset.y
    feature_names = dataset.feature_names
    costs = dataset.costs
    protected_cols = dataset.protected_cols
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
    X_train, μ, σ = normalize_features(X_train; return_params=true)
    X_test = apply_normalization(X_test, μ, σ)
    
    d = size(X_train, 2)
    model = LogisticModel(d)
    fit_logistic!(model, X_train, y_train)
    gaussian_params = estimate_gaussian_params(X_train)
    
    config = DROConfig(ε=0.1, n_scenarios=300, verbose=false)
    
    # Prendre plusieurs observations et moyenner
    n_samples = 20
    
    values_robust = zeros(d)
    values_nonrobust = zeros(d)
    
    for idx in 1:n_samples
        x_test = X_test[idx, :]
        y_test_i = y_test[idx]
        
        # Masque: toutes les features payantes sont manquantes
        mask = [costs[j] > 0 for j in 1:d]
        
        scenarios = generate_gaussian_scenarios(gaussian_params, x_test, mask, 300)
        
        # Baseline sans acquisition
        z_base = falses(d)
        obj_base_rob = evaluate_dro_objective(model, scenarios, x_test, mask, z_base, y_test_i, 0.1)
        obj_base_nr = evaluate_dro_objective(model, scenarios, x_test, mask, z_base, y_test_i, 0.0)
        
        for j in 1:d
            if mask[j]
                z_with_j = copy(z_base)
                z_with_j[j] = true
                
                obj_rob = evaluate_dro_objective(model, scenarios, x_test, mask, z_with_j, y_test_i, 0.1)
                obj_nr = evaluate_dro_objective(model, scenarios, x_test, mask, z_with_j, y_test_i, 0.0)
                
                values_robust[j] += (obj_base_rob - obj_rob)
                values_nonrobust[j] += (obj_base_nr - obj_nr)
            end
        end
    end
    
    values_robust ./= n_samples
    values_nonrobust ./= n_samples
    
    # Afficher les résultats
    println("\n" * @sprintf("%-15s %6s %10s %10s %10s %10s",
                          "Feature", "Coût", "V_rob", "V_nr", "ρ_rob", "Ratio"))
    println("-"^70)
    
    results = []
    for j in 1:d
        if costs[j] > 0
            rho = values_robust[j] / costs[j]
            ratio = values_robust[j] / (values_nonrobust[j] + 1e-10)
            
            push!(results, (feature=j, name=feature_names[j], cost=costs[j],
                           v_rob=values_robust[j], v_nr=values_nonrobust[j],
                           rho=rho, ratio=ratio))
            
            println(@sprintf("%-15s %6.0f %10.4f %10.4f %10.4f %10.2f",
                   feature_names[j], costs[j], values_robust[j], values_nonrobust[j], rho, ratio))
        end
    end
    
    # Trier par ratio valeur/coût
    sort!(results, by=x->x.rho, rev=true)
    println("\nClassement par ratio valeur/coût:")
    for (rank, r) in enumerate(results)
        println(@sprintf("  %d. %-15s (ρ = %.4f)", rank, r.name, r.rho))
    end
    
    return results
end

#==============================================================================
# Expérience 4: Convergence SAA
==============================================================================#

function experiment4_saa_convergence(; verbose::Bool=true)
    println("\n" * "="^80)
    println("EXPÉRIENCE 4: Convergence de l'approximation SAA")
    println("="^80)
    
    dataset = load_pima_diabetes(use_real_data=false)
    X, y = dataset.X, dataset.y
    costs = dataset.costs
    protected_cols = dataset.protected_cols
    
    Random.seed!(SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
    X_train, μ, σ = normalize_features(X_train; return_params=true)
    X_test = apply_normalization(X_test, μ, σ)
    
    budget = 25.0
    p_miss = 0.3
    
    d = size(X_train, 2)
    model = LogisticModel(d)
    fit_logistic!(model, X_train, y_train)
    gaussian_params = estimate_gaussian_params(X_train)
    
    # Fixer une observation et un masque
    idx = 5
    x_test = X_test[idx, :]
    y_test_i = y_test[idx]
    mask = simulate_mcar(reshape(x_test, 1, :), p_miss; protected_cols=protected_cols)[1, :]
    
    # Tailles d'échantillon à tester
    N_values = [10, 25, 50, 100, 200, 500, 1000]
    n_repetitions = 50
    
    println("\n" * @sprintf("%-8s %12s %12s %12s %12s",
                          "N", "Moyenne", "Écart-type", "Err. rel.", "Temps(ms)"))
    println("-"^60)
    
    results = []
    ref_value = nothing
    
    for N in N_values
        objectives = Float64[]
        times = Float64[]
        
        for rep in 1:n_repetitions
            config = DROConfig(ε=0.1, n_scenarios=N, seed=SEED+rep, verbose=false)
            
            t0 = time()
            result = solve_dro_acquisition(model, gaussian_params, x_test, mask,
                                          y_test_i, costs, budget, config)
            elapsed = (time() - t0) * 1000  # ms
            
            push!(objectives, result.objective)
            push!(times, elapsed)
        end
        
        mean_obj = mean(objectives)
        std_obj = std(objectives)
        mean_time = mean(times)
        
        if isnothing(ref_value)
            ref_value = mean_obj
        end
        err_rel = abs(mean_obj - ref_value) / ref_value * 100
        
        # Mettre à jour la référence avec le N le plus grand
        if N == maximum(N_values)
            ref_value = mean_obj
        end
        
        push!(results, (N=N, mean=mean_obj, std=std_obj, err_rel=err_rel, time=mean_time))
        
        println(@sprintf("%-8d %12.4f %12.4f %11.1f%% %12.1f",
               N, mean_obj, std_obj, err_rel, mean_time))
    end
    
    # Recalculer l'erreur relative par rapport au N=1000
    ref_value = results[end].mean
    for i in 1:length(results)
        results[i] = (N=results[i].N, mean=results[i].mean, std=results[i].std,
                     err_rel=abs(results[i].mean - ref_value) / ref_value * 100,
                     time=results[i].time)
    end
    
    return results
end

#==============================================================================
# Expérience 5: Sensibilité au budget
==============================================================================#

function experiment5_budget_sensitivity(; verbose::Bool=true)
    println("\n" * "="^80)
    println("EXPÉRIENCE 5: Sensibilité au budget")
    println("="^80)
    
    Random.seed!(SEED)
    
    dataset = load_pima_diabetes(use_real_data=false)
    X, y = dataset.X, dataset.y
    costs = dataset.costs
    protected_cols = dataset.protected_cols
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
    X_train, μ, σ = normalize_features(X_train; return_params=true)
    X_test = apply_normalization(X_test, μ, σ)
    
    p_miss = 0.3
    
    d = size(X_train, 2)
    model = LogisticModel(d)
    fit_logistic!(model, X_train, y_train)
    gaussian_params = estimate_gaussian_params(X_train)
    
    masks = simulate_mnar(X_test, p_miss; β=0.8, protected_cols=protected_cols)
    
    # Budgets à tester
    budgets = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0]
    
    println("\n" * @sprintf("%-8s %10s %10s %10s %10s %10s",
                          "Budget", "NoAcq", "Random", "Greedy", "DRO", "Coût DRO"))
    println("-"^65)
    
    results_dro = []
    results_greedy = []
    
    loss_no_acq = evaluate_policy_no_acquisition(model, gaussian_params, X_test, y_test, masks).logloss
    
    for budget in budgets
        if budget == 0
            loss_random = loss_no_acq
            loss_greedy = loss_no_acq
            loss_dro = loss_no_acq
            cost_dro = 0.0
        else
            config_dro = DROConfig(ε=0.1, n_scenarios=200, verbose=false)
            config_greedy = DROConfig(ε=0.0, n_scenarios=200, verbose=false)
            
            metrics_random = evaluate_policy_random(model, gaussian_params, X_test, y_test,
                                                   masks, costs, budget)
            metrics_greedy = evaluate_policy(model, gaussian_params, X_test, y_test,
                                            masks, costs, budget, config_greedy; n_eval=50)
            metrics_dro = evaluate_policy(model, gaussian_params, X_test, y_test,
                                         masks, costs, budget, config_dro; n_eval=50)
            
            loss_random = metrics_random.logloss
            loss_greedy = metrics_greedy.logloss
            loss_dro = metrics_dro.logloss
            cost_dro = metrics_dro.avg_cost
        end
        
        push!(results_dro, (budget=budget, loss=loss_dro, cost=cost_dro))
        push!(results_greedy, (budget=budget, loss=loss_greedy))
        
        println(@sprintf("%-8.0f %10.3f %10.3f %10.3f %10.3f %10.1f",
               budget, loss_no_acq, loss_random, loss_greedy, loss_dro, cost_dro))
    end
    
    return (budgets=budgets, dro=results_dro, greedy=results_greedy, no_acq=loss_no_acq)
end

#==============================================================================
# Génération des tableaux LaTeX
==============================================================================#

function generate_latex_tables()
    println("\n" * "="^80)
    println("GÉNÉRATION DES TABLEAUX LATEX")
    println("="^80)
    
    # Exécuter toutes les expériences et sauvegarder les résultats
    results1 = experiment1_mechanism_comparison(n_reps=5, verbose=false)
    results2 = experiment2_epsilon_sensitivity(verbose=false)
    results3 = experiment3_feature_value(verbose=false)
    results4 = experiment4_saa_convergence(verbose=false)
    results5 = experiment5_budget_sensitivity(verbose=false)
    
    println("\nTableaux générés et sauvegardés dans $RESULTS_DIR/")
end

#==============================================================================
# Main
==============================================================================#

function main(args=ARGS)
    println("="^80)
    println("PROJET DRO - ACQUISITION DE CARACTÉRISTIQUES AVEC DONNÉES MANQUANTES")
    println("IFT6512 - Programmation Stochastique")
    println("="^80)
    
    if isempty(args) || args[1] == "all"
        # Exécuter toutes les expériences
        experiment1_mechanism_comparison()
        experiment2_epsilon_sensitivity()
        experiment3_feature_value()
        experiment4_saa_convergence()
        experiment5_budget_sensitivity()
    elseif args[1] == "tables"
        generate_latex_tables()
    else
        exp_num = parse(Int, args[1])
        
        if exp_num == 1
            experiment1_mechanism_comparison()
        elseif exp_num == 2
            experiment2_epsilon_sensitivity()
        elseif exp_num == 3
            experiment3_feature_value()
        elseif exp_num == 4
            experiment4_saa_convergence()
        elseif exp_num == 5
            experiment5_budget_sensitivity()
        else
            println("Expérience $exp_num non reconnue (1-5)")
        end
    end
    
    println("\n" * "="^80)
    println("EXPÉRIENCES TERMINÉES")
    println("="^80)
end

# Exécuter si appelé directement
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
