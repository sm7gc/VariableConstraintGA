import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir)) 
import random 
import math 
import numpy.random as npr

from GeneticAlgorithmInterface import VariableConstraintGA
from ProblemSpaceInterface import ProblemSpace 

def roulette_selection(population):
    small = min([c[0] for c in population]) # make all the fitnesses positive 
    if small < 0:
        add = -small 
    else:
        add = 0 
    m = sum([c[0] + add for c in population])
    if m == 0:
        selection_probs = [1 / len(population) for c in population]
    else:
        selection_probs = [(c[0] + add) / m for c in population]
    return population[npr.choice(len(population), p=selection_probs)]

def decide(rate):
    return random.random() < rate

class DynamicShuffling(VariableConstraintGA):
    def __init__(self, problem_space: ProblemSpace, number_generations, population_size, max_memory, cross_over_rate, mutation_rate, user, update_interval, infeasible_rate = 0.5, elitism = 0.3):
        self.infeasible_rate = infeasible_rate 
        self.elitism = elitism
        super().__init__(problem_space, number_generations, population_size, max_memory, cross_over_rate, mutation_rate, user, update_interval)
    
    def _sort_pop(self, pop):
        pop.sort(key=lambda i: i[0], reverse=True)
    
    def select(self):
        # select from feasible 
        if decide((self.num_feasible * 2) / (self.num_feasible + len(self.infeasible_pop))):

            # randomly select a bin with children 
            bi = random.choice(range(len(self.bins)))
            while len(self.bins[bi]) == 0:
                bi = random.choice(range(len(self.bins))) 
            
            # select from bin using roulette selection 
            return roulette_selection(self.bins[bi])
        # select from infeasible 
        else:
            return roulette_selection(self.infeasible_pop)


    def place_in_bin(self, ind, infeasible_pop):
        # determine if feasible 
        fes = True 
        constraints_sat = 0 
        for con in self.problem_space.get_constant_constraints() + self.variable_constraints:
            if not con.apply(ind):
                fes = False 
            else:
                constraints_sat += 1 
                 
        
        # if feasible put in bin 
        if fes:
            b = self.problem_space.place_in_bin(ind)

            # check if there is room
            if len(self.bins[b]) < self.inds_per_bin:
                self.bins[b].append((self.problem_space.fitness(ind), ind))
                self._sort_pop(self.bins[b])
                self.num_feasible += 1 
            # if individual is better then current ones, replace it 
            elif self.problem_space.fitness(ind) >= self.bins[b][-1][0]: 
                self.bins[b].pop(-1)
                self.bins[b].append((self.problem_space.fitness(ind), ind))
                self._sort_pop(self.bins[b])
                self.num_feasible += 1 
        # otherwise put in the infeasible population 
        else:
            # if there is still room in the infeasible fitness add it 
            if len(infeasible_pop) < self.infeasible_pop_size:
                
                infeasible_pop.append((constraints_sat, ind))
    
    def re_shuffle(self):
        # First get all children from feasible and infeasible pop 
        all_children = self.infeasible_pop[:]
        all_children += [el for li in self.bins for el in li]

        # then re-set all populations 
        new_infeasible = [] 
        self.set_up()

        #then re-add all children based on new cons 
        for c in all_children:
            self.place_in_bin(c[1], new_infeasible)
        
        self.infeasible_pop = new_infeasible 


    def set_up(self):
        self.infeasible_pop_size = self.max_memory * self.infeasible_rate 
        self.elitism_num = round(self.infeasible_pop_size * self.elitism)
        self.feasible_pop_size = self.max_memory - self.infeasible_pop_size 
        self.inds_per_bin = math.floor(self.feasible_pop_size / self.problem_space.get_num_bins()) 
        self.bins = []
        self.num_feasible = 0 
        self.infeasible_pop = [] 
        for i in range(self.problem_space.get_num_bins()):
            self.bins.append([])
        
        # generate initial population 
        for i in range(self.population_size):
            indv = self.problem_space.generate_random_individual()
            self.place_in_bin(indv, self.infeasible_pop)
        

    def run_one_generation(self, made_change): 

        # if the constraints have been change, reshuffle population 
        if made_change:
            self.re_shuffle()
        self._sort_pop(self.infeasible_pop)
        new_infeasible = self.infeasible_pop[:self.elitism_num]
        for i in range(math.floor(self.population_size / 2)):
            # select 
            child1 = self.select()[1]
            child2 = self.select()[1]

            # cross over 
            if decide(self.cross_over_rate):
                child1, child2 = self.problem_space.cross_over(child1, child2)
            
            # mutate 
            child1 = self.problem_space.mutate(child1, self.mutation_rate)
            child2 = self.problem_space.mutate(child2, self.mutation_rate)

            # update population 
            self.place_in_bin(child1, new_infeasible)
            self.place_in_bin(child2, new_infeasible)
        
        self.infeasible_pop = new_infeasible 

        return self.bins

    