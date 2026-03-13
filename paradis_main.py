import os
import logging
import timeit
from datetime import datetime
import argparse
import random
import numpy as np
import csv

import torch

from naslib.utils import utils
from naslib.utils.logging import setup_logger
from naslib.utils.get_dataset_api import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import get_search_space
from naslib.predictors.zerocost import ZeroCost

def main(args):
    exp_name = "/adv_genetic_regparam_{:}_{:}_{:}_{:}".format(args.search_space, args.dataset, args.metric1, datetime.now().strftime('%Y-%m-%d_%H-%M'))

    logger = setup_logger(args.log_path + exp_name + '.log')
    logger.setLevel(logging.INFO)
    logger.info('Ablation type 2')

    utils.set_seed(args.seed)
    logger.info('Seed {:}'.format(args.seed))

    search_space = get_search_space(args.search_space, args.dataset)
    dataset_api = get_dataset_api(args.search_space, args.dataset)
    args.data = "{}/data".format(utils.get_project_root())

    train_loader, _, _, _, _ = utils.get_train_val_loaders(args)

    predictor1 = ZeroCost(method_type=args.metric1)

    def count_params(model):
        return(np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name))

    def get_regularized_params(n_params, mu, sigma):

        reg = torch.exp(-torch.pow(torch.as_tensor(n_params) - mu, 2)/sigma**2).item()
        return(reg)

    values_params = []

    '''
    def array_in(array1, array_list):

        for a in array_list:
            if all((array1 == a)):
                return(True)
        return(False)
    '''

    def is_in_population(arch, population):

        population_archs = [v[0].op_indices for k, v in population.items()]
        return(arch.op_indices in population_archs)
    
    tau_schedule = list(np.linspace(args.tau_min, args.tau_max, args.n_generations))
    tau = 0.

    population = {}
    start_time = timeit.default_timer()
    for i in range(args.pop_size):
        graph = search_space.clone()
        graph.sample_random_architecture()
        graph.parse()
        while len(population)!=0 and is_in_population(graph, population):
            graph = search_space.clone()
            graph.sample_random_architecture()
            graph.parse()
        score_1 = predictor1.query(graph, train_loader)
        score_2 = 0
        n_params = count_params(graph)
        population[i] = (graph, score_1, score_2, n_params)
        values_params.append(n_params)
    end_time = timeit.default_timer()
    logger.info('Generated initial population with time {:}'.format(end_time-start_time))

    params_average = np.mean(values_params)
    params_std = np.std(values_params)
    logger.info('Parameter distribution of initial population : mean {:} | std {:}'.format(params_average, params_std))

    start_time = timeit.default_timer()
    for i in range(args.pop_size):
        g = population[i]
        population[i] = (g[0], g[1], get_regularized_params(g[3], params_average + tau*params_std, params_std), g[3])
    end_time = timeit.default_timer()
    logger.info('Recalibrated intial population with time {:}'.format(end_time-start_time))

    with open(args.output_path + exp_name + '.csv', 'a+', newline='') as out:
        writer = csv.writer(out)
        header = ['step', 'arch', 'score_metric1', 'score_metric2', 'n_params', 'sum_scores', 'true_score']
        writer.writerow(header)
        for i, g in population.items():
            row = [0, g[0].op_indices, g[1], g[2], g[3], g[1] + g[2], g[0].query(Metric.VAL_ACCURACY, args.dataset, dataset_api=dataset_api)]
            writer.writerow(row)

    with open(args.output_path + exp_name + '_pophist.csv', 'a+', newline='') as out:
        writer = csv.writer(out)
        header = ['step', 'arch', 'score_metric1', 'score_metric2', 'n_params', 'sum_scores', 'true_score']
        writer.writerow(header)
        for i, g in population.items():
            row = [0, g[0].op_indices, g[1], g[2], g[3], g[1] + g[2], g[0].query(Metric.VAL_ACCURACY, args.dataset, dataset_api=dataset_api)]
            writer.writerow(row)

    for n in range(args.n_generations):
        logger.info('Start of generation {:}'.format(n))
        start_time = timeit.default_timer()
        
        tau = tau_schedule[-(n+1)]
        logger.info('Tau : {:}'.format(tau))

        #Accumulate number of parameters data until end of generation
        for s in range(args.steps_per_gen):

            def get_random_population_subset(population, subset_size):

                subset_indices = random.sample(list(population.keys()), k=subset_size)

                sub_population = {}
                for idx in subset_indices:
                    sub_population[idx] = population[idx]

                return(sub_population)

            def get_pop_complement(u_population, sub_population):

                complement = {}
                for p in u_population.keys():
                    if not is_in_population(u_population[p][0], sub_population):
                        complement[p] = u_population[p]

                return(complement)
            
            #Sort population by discriminator fitness
            sorted_indices_disc = sorted(list(population.keys()), key=lambda i: population[i][2])
            #Get 2 worst architectures
            index_worst1, index_worst2 = sorted_indices_disc[0], sorted_indices_disc[1]
            if args.crossover:
                #Get 3rd worst architecture
                index_worst3 = sorted_indices_disc[2]
            #Get a subpopulation of size subset_size
            sub_population = get_random_population_subset(population, args.subset_size)
            #Split 2 containing architectures that were not sampled into the subpopulation
            remaining_pop = get_pop_complement(population, sub_population)
            #Get best architecture in subpop
            index_best1 = max(list(sub_population.keys()), key=lambda i: sub_population[i][1])
            #Get best architecture in remaining pop
            index_best2 = max(list(remaining_pop.keys()), key=lambda i: remaining_pop[i][1])

            #Give priority to discriminator metric in case of ties to avoid getting stuck in a local minima
            if index_best1 == index_worst1 or index_best1 == index_worst2 or index_best1 == index_worst3:
                pop_onlybest = {index_worst1 : population[index_worst1],
                                index_worst2 : population[index_worst2],
                                index_worst3 : population[index_worst3]}
                pop_nobest = get_pop_complement(sub_population, pop_onlybest)

                index_best1 = max(list(pop_nobest.keys()), key=lambda i: pop_nobest[i][1])

            if index_best2 == index_worst1 or index_best2 == index_worst2 or index_best2 == index_worst3:
                pop_onlybest = {index_worst1 : population[index_worst1],
                                index_worst2 : population[index_worst2],
                                index_worst3 : population[index_worst3]}
                pop_nobest = get_pop_complement(remaining_pop, pop_onlybest)

                index_best2 = max(list(pop_nobest.keys()), key=lambda i: pop_nobest[i][1])

            best_graph_1 = population[index_best1][0]
            best_graph_2 = population[index_best2][0]

            #Child 1 created from mutating split 1 winner
            failsafe_flag_1 = False
            child_graph_1 = search_space.clone()
            child_graph_1.mutate(parent=best_graph_1)
            child_graph_1.parse()
            f = 0
            while len(population)!=0 and f < args.failsafe_size and is_in_population(child_graph_1, population):
                child_graph_1 = search_space.clone()
                child_graph_1.mutate(parent=best_graph_1)
                child_graph_1.parse()
                f+=1
            if f == args.failsafe_size:
                failsafe_flag_1 = True
            child_score_1_1 = predictor1.query(child_graph_1, train_loader)
            child_params_1 = count_params(child_graph_1)
            child_score_1_2 = get_regularized_params(child_params_1, params_average + tau*params_std, params_std)

            #Child 2 created from mutating split 2 winner
            failsafe_flag_2 = False
            child_graph_2 = search_space.clone()
            child_graph_2.mutate(parent=best_graph_2)
            child_graph_2.parse()
            f = 0
            while len(population)!=0 and f < args.failsafe_size and is_in_population(child_graph_2, population):
                child_graph_2 = search_space.clone()
                child_graph_2.mutate(parent=best_graph_2)
                child_graph_2.parse()
                f+=1
            if f == args.failsafe_size:
                failsafe_flag_2 = True
            child_score_2_1 = predictor1.query(child_graph_2, train_loader)
            child_params_2 = count_params(child_graph_2)
            child_score_2_2 = get_regularized_params(child_params_2, params_average + tau*params_std, params_std)

            if args.crossover:
                #Child 3 created from exchanging genetic material between best1 and best2
                failsafe_flag_3 = False
                child_graph_3 = search_space.clone()
                child_graph_3.crossover(parent0=best_graph_1, parent1=best_graph_2)
                child_graph_3.parse()
                f = 0
                while len(population)!=0 and f < args.failsafe_size and is_in_population(child_graph_3, population):
                    child_graph_3 = search_space.clone()
                    child_graph_3.crossover(parent0=best_graph_1, parent1=best_graph_2)
                    child_graph_3.parse()
                    f+=1
                if f == args.failsafe_size:
                    failsafe_flag_3 = True
                child_score_3_1 = predictor1.query(child_graph_3, train_loader)
                child_params_3 = count_params(child_graph_3)
                child_score_3_2 = get_regularized_params(child_params_3, params_average + tau*params_std, params_std)

            #Replace split 1 loser with the child of split 1 winner
            if not failsafe_flag_1:
                population[index_worst1] = (child_graph_1, child_score_1_1, child_score_1_2, child_params_1)
            #Replace split 2 loser with the child of split 2 winner
            if not failsafe_flag_2:
                population[index_worst2] = (child_graph_2, child_score_2_1, child_score_2_2, child_params_2)
            if args.crossover:
                #Replace split 3 loser with the child of split 1 and 2 winners
                if not failsafe_flag_3:
                    population[index_worst3] = (child_graph_3, child_score_3_1, child_score_3_2, child_params_3)

            if args.crossover:
                out_idx = [index_worst1, index_worst2, index_worst3]
            else:
                out_idx = [index_worst1, index_worst2]
            
            for i in out_idx:
                with open(args.output_path + exp_name + '.csv', 'a+', newline='') as out:
                    writer = csv.writer(out)
                    g = population[i]
                    row = [n*args.steps_per_gen + s + 1, g[0].op_indices, g[1], g[2], g[3], g[1]+g[2], g[0].query(Metric.VAL_ACCURACY, args.dataset, dataset_api=dataset_api)]
                    writer.writerow(row)

        end_time = timeit.default_timer()
        logger.info('Completed generation with time {:}'.format(end_time-start_time))

        if n+1 != args.n_generations:
            logger.info('Updating scores for population')
            values_params = []

            start_time = timeit.default_timer()
            for i in range(len(population)):
                values_params.append(population[i][3])
            params_average = np.mean(values_params)
            params_std = np.std(values_params)
            logger.info('Parameter distribution of population at generation {:}: mean {:} | std {:}'.format(n, params_average, params_std))
            for i in range(len(population)):
                graph = population[i][0]
                score_1 = population[i][1]
                n_params = population[i][3]
                population[i] = (graph, score_1, get_regularized_params(n_params, params_average + tau*params_std, params_std), n_params)
            end_time = timeit.default_timer()
            logger.info('Recomputed population scores with time {:}'.format(end_time-start_time))

        with open(args.output_path + exp_name + '_pophist.csv', 'a+', newline='') as out:
            writer = csv.writer(out)
            for i, g in population.items():
                row = [n, g[0].op_indices, g[1], g[2], g[3], g[1] + g[2], g[0].query(Metric.VAL_ACCURACY, args.dataset, dataset_api=dataset_api)]
                writer.writerow(row)

    logger.info('Finished search')
    logger.info('Calculating true scores for final population')
    for i, g in population.items():
        logger.info('Architecture {:} : #params {:} | final score {:} | true score {:}'.format(g[0].op_indices, g[3], g[1], g[0].query(Metric.VAL_ACCURACY, args.dataset, dataset_api=dataset_api)))

if __name__=='__main__':
    parser = argparse.ArgumentParser('')

    parser.add_argument('--search_space', type=str, choices=['nasbench201', 'nasbench301', 'transbench101_macro', 'transbench101_micro'], help='Search space to search in.')
    parser.add_argument('--dataset', type=str, help='Task to evaluate for in search space.')
    parser.add_argument('--metric1', type=str, help='First predictor to use (should have better general correlation)')
    
    parser.add_argument('--pop_size', type=int, help='Number of architecture in the population.')
    parser.add_argument('--n_generations', type=int, help='Number of generations (temperature updated at the end of generations)')
    parser.add_argument('--steps_per_gen', type=int, help='Number of steps per generation. A new architecture may be generated via mutation at each step.')
    parser.add_argument('--subset_size', type=int, help='Size of population subsets considered at each evolution step.')
    parser.add_argument('--crossover', action='store_true', help='Wether to use crossover.')
    parser.add_argument('--tau_min', type=float, help='Initial value of Tau schedule.')
    parser.add_argument('--tau_max', type=float, help='Final value of Tau schedule.')
    parser.add_argument('--failsafe_size', type=int, default=48, help='This affects the number of loops the algorithm makes before assuming it is stuck in a mutation loop.')

    parser.add_argument('--train_portion', type=float, default=0.7, help='Portion of dataset to use as training data.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')

    parser.add_argument('--log_path', type=str, default='.', help='Location of generated log file.')
    parser.add_argument('--output_path', type=str, default='.', help='Location of generated output csv file.')
    parser.add_argument('--seed', type=int, default=678, help='Random seed')
    args = parser.parse_args()

    main(args)
