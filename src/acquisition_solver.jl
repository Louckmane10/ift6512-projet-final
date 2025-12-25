"""
    acquisition_solver.jl

Module pour la résolution du problème combinatoire d'acquisition de caractéristiques.
Implémente l'énumération exacte et l'heuristique gloutonne.

Auteur: Louck
Projet: IFT6512 - DRO avec données manquantes
"""

module AcquisitionSolver

using Combinatorics

export solve_acquisition, solve_acquisition_exact, solve_acquisition_greedy
export enumerate_feasible_solutions, marginal_value, acquisition_cost

#==============================================================================
# Énumération des solutions admissibles
==============================================================================#

"""
    enumerate_feasible_solutions(mask::Vector{Bool}, costs::Vector{Float64}, 
                                budget::Float64) -> Vector{Vector{Bool}}

Énumère toutes les solutions z admissibles satisfaisant:
- z[j] ≤ mask[j] (on ne peut acquérir que ce qui est manquant)
- Σ costs[j] * z[j] ≤ budget

Retourne un vecteur de vecteurs booléens.
"""
function enumerate_feasible_solutions(mask::Vector{Bool}, 
                                     costs::Vector{Float64}, 
                                     budget::Float64)
    
    d = length(mask)
    idx_miss = findall(mask)
    d_miss = length(idx_miss)
    
    feasible = Vector{Vector{Bool}}()
    
    # Solution vide toujours admissible
    z_empty = falses(d)
    push!(feasible, collect(z_empty))
    
    # Énumérer tous les sous-ensembles des indices manquants
    for k in 1:d_miss
        for subset in combinations(idx_miss, k)
            # Vérifier la contrainte de budget
            total_cost = sum(costs[j] for j in subset)
            if total_cost <= budget
                z = falses(d)
                for j in subset
                    z[j] = true
                end
                push!(feasible, collect(z))
            end
        end
    end
    
    return feasible
end

"""
    count_feasible_solutions(mask::Vector{Bool}, costs::Vector{Float64}, 
                            budget::Float64) -> Int

Compte le nombre de solutions admissibles sans les énumérer.
Utile pour décider si l'énumération est tractable.
"""
function count_feasible_solutions(mask::Vector{Bool}, 
                                 costs::Vector{Float64}, 
                                 budget::Float64)
    
    idx_miss = findall(mask)
    d_miss = length(idx_miss)
    
    count = 1  # Solution vide
    
    for k in 1:d_miss
        for subset in combinations(idx_miss, k)
            total_cost = sum(costs[j] for j in subset)
            if total_cost <= budget
                count += 1
            end
        end
    end
    
    return count
end

#==============================================================================
# Résolution exacte par énumération
==============================================================================#

"""
    solve_acquisition_exact(eval_func::Function, mask::Vector{Bool},
                           costs::Vector{Float64}, budget::Float64) -> (Vector{Bool}, Float64)

Résout le problème d'acquisition de façon exacte par énumération.

Arguments:
- eval_func: Fonction z -> objectif (à minimiser)
- mask: Masque de manquement
- costs: Coûts d'acquisition
- budget: Budget total

Retourne (z_optimal, objectif_optimal).
"""
function solve_acquisition_exact(eval_func::Function,
                                mask::Vector{Bool},
                                costs::Vector{Float64},
                                budget::Float64;
                                verbose::Bool=false)
    
    feasible = enumerate_feasible_solutions(mask, costs, budget)
    
    if verbose
        println("Nombre de solutions admissibles: $(length(feasible))")
    end
    
    best_z = feasible[1]
    best_obj = eval_func(best_z)
    
    for (i, z) in enumerate(feasible)
        obj = eval_func(z)
        
        if verbose && i % 100 == 0
            println("  Évaluation $i/$(length(feasible)), meilleur obj: $best_obj")
        end
        
        if obj < best_obj
            best_obj = obj
            best_z = z
        end
    end
    
    return best_z, best_obj
end

#==============================================================================
# Heuristique gloutonne
==============================================================================#

"""
    marginal_value(eval_func::Function, z_current::Vector{Bool}, j::Int) -> Float64

Calcule la valeur marginale de l'acquisition de la caractéristique j:
V_j(z) = objectif(z) - objectif(z + e_j)

Une valeur positive signifie que l'acquisition améliore l'objectif.
"""
function marginal_value(eval_func::Function, z_current::Vector{Bool}, j::Int)
    obj_current = eval_func(z_current)
    
    z_with_j = copy(z_current)
    z_with_j[j] = true
    obj_with_j = eval_func(z_with_j)
    
    return obj_current - obj_with_j  # Positif si amélioration
end

"""
    solve_acquisition_greedy(eval_func::Function, mask::Vector{Bool},
                            costs::Vector{Float64}, budget::Float64;
                            verbose::Bool=false) -> (Vector{Bool}, Float64)

Résout le problème d'acquisition par heuristique gloutonne.
À chaque itération, sélectionne la caractéristique avec le meilleur ratio valeur/coût.

Arguments:
- eval_func: Fonction z -> objectif (à minimiser)
- mask: Masque de manquement
- costs: Coûts d'acquisition
- budget: Budget total

Retourne (z_solution, objectif).
"""
function solve_acquisition_greedy(eval_func::Function,
                                 mask::Vector{Bool},
                                 costs::Vector{Float64},
                                 budget::Float64;
                                 verbose::Bool=false)
    
    d = length(mask)
    z = falses(d)
    remaining_budget = budget
    
    idx_miss = findall(mask)
    
    iteration = 0
    while remaining_budget > 0
        iteration += 1
        
        # Trouver les candidats (non encore acquis, coût ≤ budget restant)
        candidates = [j for j in idx_miss if !z[j] && costs[j] <= remaining_budget]
        
        if isempty(candidates)
            break
        end
        
        # Calculer le ratio valeur/coût pour chaque candidat
        best_j = -1
        best_ratio = -Inf
        best_value = 0.0
        
        for j in candidates
            value = marginal_value(eval_func, z, j)
            ratio = value / costs[j]
            
            if ratio > best_ratio
                best_ratio = ratio
                best_j = j
                best_value = value
            end
        end
        
        # Si aucune amélioration possible, arrêter
        if best_value <= 0
            if verbose
                println("Iteration $iteration: Pas d'amélioration, arrêt")
            end
            break
        end
        
        # Sélectionner la meilleure caractéristique
        z[best_j] = true
        remaining_budget -= costs[best_j]
        
        if verbose
            println("Iteration $iteration: Acquérir feature $best_j, " *
                   "ratio=$(round(best_ratio, digits=4)), " *
                   "budget restant=$(round(remaining_budget, digits=2))")
        end
    end
    
    final_obj = eval_func(z)
    
    return z, final_obj
end

#==============================================================================
# Résolution adaptative (choisit automatiquement la méthode)
==============================================================================#

"""
    solve_acquisition(eval_func::Function, mask::Vector{Bool},
                     costs::Vector{Float64}, budget::Float64;
                     exact_threshold::Int=15, verbose::Bool=false) -> (Vector{Bool}, Float64, Symbol)

Résout le problème d'acquisition en choisissant automatiquement la méthode:
- Énumération exacte si d_miss ≤ exact_threshold
- Heuristique gloutonne sinon

Retourne (z_solution, objectif, methode).
"""
function solve_acquisition(eval_func::Function,
                          mask::Vector{Bool},
                          costs::Vector{Float64},
                          budget::Float64;
                          exact_threshold::Int=15,
                          verbose::Bool=false)
    
    d_miss = sum(mask)
    
    # Estimer le nombre de solutions
    if d_miss <= exact_threshold
        n_feasible = count_feasible_solutions(mask, costs, budget)
        
        if verbose
            println("d_miss = $d_miss, solutions admissibles ≈ $n_feasible")
        end
        
        # Si tractable, utiliser l'énumération exacte
        if n_feasible <= 2^exact_threshold
            z, obj = solve_acquisition_exact(eval_func, mask, costs, budget; verbose=verbose)
            return z, obj, :exact
        end
    end
    
    # Sinon, utiliser l'heuristique gloutonne
    z, obj = solve_acquisition_greedy(eval_func, mask, costs, budget; verbose=verbose)
    return z, obj, :greedy
end

#==============================================================================
# Utilitaires pour l'analyse
==============================================================================#

"""
    acquisition_cost(z::Vector{Bool}, costs::Vector{Float64}) -> Float64

Calcule le coût total d'une décision d'acquisition.
"""
function acquisition_cost(z::Vector{Bool}, costs::Vector{Float64})
    return sum((costs[j] for j in 1:length(z) if z[j]), init=0.0)
end

"""
    acquired_features(z::Vector{Bool}) -> Vector{Int}

Retourne les indices des caractéristiques acquises.
"""
function acquired_features(z::Vector{Bool})
    return findall(z)
end

"""
    acquisition_summary(z::Vector{Bool}, costs::Vector{Float64}, 
                       feature_names::Vector{String}=String[])

Affiche un résumé de la décision d'acquisition.
"""
function acquisition_summary(z::Vector{Bool}, costs::Vector{Float64};
                            feature_names::Vector{String}=String[])
    
    acquired = findall(z)
    total_cost = acquisition_cost(z, costs)
    
    println("=== Résumé de l'acquisition ===")
    println("Caractéristiques acquises: $(length(acquired))")
    println("Coût total: $total_cost")
    println()
    
    if !isempty(acquired)
        println("Détail:")
        for j in acquired
            name = isempty(feature_names) ? "Feature $j" : feature_names[j]
            println("  - $name (coût: $(costs[j]))")
        end
    else
        println("Aucune acquisition (imputation pure)")
    end
end

end # module
