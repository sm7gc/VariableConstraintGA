from GeneticAlgorithmInterface import VariableConstraintGA 

class YouAlgorithm(VariableConstraintGA):
    def set_up(self): 
        """
        Insert all your set up code here 

        You can generation an initial population 
        of individuals. However you may only 
        generate self.population_size in this method, and 
        can only store up to self.max_memory individuals  
        in total   
        
        We provide the useful functions 
        and values available for you to use here 
        """

        self.population_size # the max number of individuals you can generate per generation 
        self.max_memory # the max number of individuals you can store at any time (always > then pop size)
        self.mutation_rate # the rate of mutation to give mutation function 
        self.cross_over_rate # the ratio of time you should preform the cross over function 
        self.variable_constraints # the current list of variable constraints 

        ind1 = self.problem_space.generate_random_individual() # randomly generate a new individual 
        ind2 = self.problem_space.generate_random_individual() 

        fit = self.problem_space.fitness(ind1) # quality value of individual 
        ind3 = self.problem_space.mutate(ind1, self.mutation_rate) # preform mutation 
        child1, child2 = self.problem_space.cross_over(ind1, ind2) # preform cross over 

        cons = self.problem_space.get_constant_constraints() # list of static constraints 
        self.problem_space.get_num_bins() # number of diversity bins in problem space 
        self.problem_space.place_in_bin(ind1) # get the index of bin ind should be placed in 
        
        # you can check if individuals satisfy a constant through the apply function 
        cons[0].apply(ind1) # returns true if constraint is satisfied 


    def run_one_generation(self, made_change): 
        """
        Complete a single generation of the algorithm

        Returns the population of valid (by both constant and variable constraints)
        individuals that are shorted in bins. Each individual should be stored as a tuple
        with the first value being the fitness and the second being the object 

        EX: [[(fit1, obj1)], [], [(fit2, obj2), (fit3, obj3)], .... ] 
        
        """
        pop = [] 
        for i in range(self.problem_space.get_num_bins()):
            pop.append([])
        return pop 