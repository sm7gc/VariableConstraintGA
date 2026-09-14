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
    return npr.random() < rate

class DynamicShuffling(VariableConstraintGA):
    def _sort_pop(self, pop):
        pop.sort(key=lambda i: i[0], reverse=True)
    
    def select_random(self):

        # randomly select individual
        if decide((self.num_feasible * 2) / (self.num_feasible + len(self.infeasible_pop))):
            return self.select_feasible()
        else:
            return self.select_infeasible()

    def select_feasible(self):
        # randomly select a bin with children 
        bi = npr.choice(range(len(self.bins)))
        while len(self.bins[bi]) == 0:
            bi = npr.choice(range(len(self.bins)))  
        # select from bin using roulette selection 
        return roulette_selection(self.bins[bi])

    def select_infeasible(self):
        return roulette_selection(self.infeasible_pop)
    
    def select_near(self):
        return roulette_selection([x for x in self.infeasible_pop if x[0] >= self.near_sat])

    def place_in_bin(self, ind, infeasible_pop):

        satisfaction = self.satisfaction(ind)
        fes = (satisfaction == 1)

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
            # if there is still room in the infeasible pop add it 
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((satisfaction, ind))

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
        self.num_near = len([x for x in self.infeasible_pop if x[0] >= self.near_sat])

    def satisfaction(self, ind):
        applicable_constraints = self.problem_space.get_constant_constraints() + self.variable_constraints
        constraints_sat = sum([con.apply(ind) for con in applicable_constraints])
        return constraints_sat/len(applicable_constraints)

    def set_up(self): 
        self.infeasible_rate = 0.5
        self.elitism = 0.3
        self.near_sat = 0.6

        self.infeasible_pop_size = self.max_memory * self.infeasible_rate
        self.elitism_num = round(self.infeasible_pop_size * self.elitism)
        self.feasible_pop_size = self.max_memory - self.infeasible_pop_size
        self.inds_per_bin = self.feasible_pop_size // self.problem_space.get_num_bins()
        self.bins = []
        self.num_feasible = 0
        self.infeasible_pop = []
        self.num_near = 0

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

        for i in range(self.population_size // 2):
            
            # dynamic selection based on constraint satisfaction
            sat, child1 = self.select_random()
            # if feasible, select feasible
            if sat == 1:
                child2 = self.select_feasible()[1]
            # if near feasible, select feasible or near feasible
            elif sat >= self.near_sat:
                child2 = self.select_near()[1] if decide(self.num_near / len(self.infeasible_pop)) else self.select_feasible()[1]
            # if infeasible, select infeasible or feasible
            else:
                child2 = self.select_infeasible()[1] if decide(len(self.infeasible_pop) / self.max_memory) else self.select_feasible()[1]

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
        self.num_near = len([x for x in self.infeasible_pop if x[0] >= self.near_sat])

        return self.bins