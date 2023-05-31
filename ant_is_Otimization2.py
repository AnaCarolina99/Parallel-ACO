import multiprocessing

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from typing import *
import random
import math
import time
from joblib import Parallel, delayed
from multiprocessing import Process, Manager
from threading import Thread


class AntIS():
    def get_pairwise_distance(self, matrix: np.ndarray) -> np.ndarray:
        return euclidean_distances(matrix)


    def get_visibility_rates_by_distances(self, distances: np.ndarray) -> np.ndarray:
        # Evite divisão por zero, preenchendo com um valor muito grande
        distances = np.where(distances == 0, 1e-9, distances)

        # Calcula todas as visibilidades de uma vez
        visibilities = 1 / distances

        # Define a diagonal principal como zero para evitar divisão por zero
        np.fill_diagonal(visibilities, 0)

        return visibilities


    def create_colony(self, num_ants):
        return np.full((num_ants, num_ants), -1)


    def create_pheromone_trails(self, search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
        num_cities = search_space.shape[0]
        trails = np.ones((num_cities, num_cities)) * initial_pheromone
        np.fill_diagonal(trails, 0)
        return trails


    def get_pheromone_deposit(self, ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
        tour_length = 0
        for path in ant_choices:
            tour_length += distances[path[0], path[1]]

        if tour_length == 0:
            return 0

        if math.isinf(tour_length):
            print('ERROR! Length of the tour is infinity.')

        return deposit_factor / tour_length


    # Carol Otimizou esta função
    def get_probabilities_paths_ordered(self, ant: np.array, visib_rates: np.array, phe_trails: np.array) \
            -> Tuple[Tuple[int, Any]]:
        available_instances = np.nonzero(ant < 0)[0]

        # Calculate the sum of pheromones over all available paths
        smell = np.sum(phe_trails[available_instances] * visib_rates[available_instances])

        # Calculate the probabilities for all available paths in one go
        path_smell = phe_trails[available_instances] * visib_rates[available_instances]
        probabilities = np.zeros((len(available_instances), 2))
        probabilities[:, 0] = available_instances
        probabilities[:, 1] = np.divide(path_smell, smell, out=np.zeros_like(path_smell), where=path_smell != 0)

        # Sort the probabilities in descending order
        sorted_probabilities = probabilities[probabilities[:, 1].argsort()][::-1]
        return tuple([(int(i[0]), i[1]) for i in sorted_probabilities])


    def get_best_solution(self, ant_solutions: np.ndarray, X, Y) -> np.array:
        def evaluate_solution(solution):
            instances_selected = np.nonzero(solution)[0]
            X_train = X[instances_selected, :]
            Y_train = Y[instances_selected]
            classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)

            Y_pred = classifier_1nn.predict(X)
            return accuracy_score(Y, Y_pred)

        num_cores = 8  # adjust this to the number of cores on your machine
        accuracies = Parallel(n_jobs=num_cores)(delayed(evaluate_solution)(solution) for solution in ant_solutions)
        best_solution_index = np.argmax(accuracies)
        best_solution = ant_solutions[best_solution_index]

        # print(f"The winner is ant {best_solution_index} with accucarcy {accuracies[best_solution_index]}")
        return best_solution

    def has_instance_to_aval(self, vet):
        result = False
        for item in vet:
            if item == -1:
                return True
        return result

    def ant_search_solution(self, i, ant, Q, evaporation_rate):
        #print(type(ant))
        while -1 in ant:
            last_choice = self.ant_choices[i][-1]
            ant_pos = last_choice[1]
            choices = self.get_probabilities_paths_ordered(
                ant,
                self.visibility_rates[ant_pos, :],
                self.pheromone_trails[ant_pos, :])

            for choice in choices:
                next_instance = choice[0]
                probability = choice[1]

                ajk = random.randint(0, 1)
                final_probability = probability * ajk
                if final_probability != 0:
                    self.ant_choices[i].append((ant_pos, next_instance))
                    self.the_colony[i, next_instance] = 1
                    ant[next_instance] = 1
                    # Ant deposits the pheromones
                    ant_deposit = self.get_pheromone_deposit(self.ant_choices[i], self.distances, Q)
                    self.pheromone_trails[ant_pos, next_instance] += ant_deposit
                    break
                else:
                    self.the_colony[i, next_instance] = 0
                    ant[next_instance] = 0

            '''
            # Ant deposits the pheromones
            ant_deposit = self.get_pheromone_deposit(self.ant_choices[i], self.distances, Q)
            for path in self.ant_choices[i][1:]:  # Never deposit in pheromone on i == j!
                self.pheromone_trails[path[0], path[1]] += ant_deposit
            '''

    def evaporation(self):
        # Pheromones evaporation
        while True:
            time.sleep(0.2)
            print('Evaporation...')
            for i in range(self.pheromone_trails.shape[0]):
                for j in range(self.pheromone_trails.shape[1]):
                    self.pheromone_trails[i, j] = (1 - self.evaporation_rate) * self.pheromone_trails[i, j]


    def run_colony(self, X, Y, initial_pheromone, evaporation_rate, Q):
        self.distances = self.get_pairwise_distance(X)
        self.visibility_rates = self.get_visibility_rates_by_distances(self.distances)
        self.the_colony = self.create_colony(X.shape[0])
        self.evaporation_rate = evaporation_rate
        self.flag = True
        for i in range(X.shape[0]):
            self.the_colony[i, i] = 1

        self.ant_choices = [[(i, i)] for i in range(len(self.the_colony))]
        self.pheromone_trails = self.create_pheromone_trails(self.distances, initial_pheromone)

        process_list = []

        for i, ant in enumerate(self.the_colony):
            #self.ant_search_solution(i, ant, Q, evaporation_rate)
            #p = multiprocessing.Process(target=self.ant_search_solution, args=(i, ant, Q, evaporation_rate))
            p = Thread(target=self.ant_search_solution, args=(i, ant, Q, evaporation_rate))
            p.start()
            process_list.append(p)

        #num_cores = 8  # adjust this to the number of cores on your machine
        #Parallel(n_jobs=8)(delayed(self.ant_search_solution)(i, ant, Q, evaporation_rate) for i, ant in enumerate(self.the_colony))

        #ev = multiprocessing.Process(target=self.evaporation)
        #ev.start()

        for p in process_list:
            p.join()
        #ev.terminate()

        instances_selected = np.nonzero(self.get_best_solution(self.the_colony, X, Y))[0]
        return instances_selected


if __name__ == '__main__':
    start_time = time.time()
    original_df = pd.read_csv("tic-tac-toe-endgame.csv", sep=';')
    dataframe = pd.read_csv("tic-tac-toe-endgame.csv", sep=';')
    classes = dataframe["V10"]
    dataframe = dataframe.drop(columns=['V10'])
    initial_pheromone = 1
    Q = 1
    evaporation_rate = 0.1
    antis = AntIS()
    print('Starting search')
    indices_selected = antis.run_colony(dataframe.to_numpy(), classes.to_numpy(), initial_pheromone, evaporation_rate, Q)
    print('End Search')
    print(len(indices_selected))
    # print(indices_selected)
    reduced_dataframe = original_df.iloc[indices_selected]
    reduced_dataframe.to_csv('Mushrooms_otimizado_reduzido.csv', index=False)
    print("Execution finished")
    print("--- %s Hours ---" % ((time.time() - start_time) // 3600))
    print("--- %s Minutes ---" % ((time.time() - start_time) // 60))
    print("--- %s Seconds ---" % (time.time() - start_time))
