from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from scipy.spatial.distance import pdist, squareform 
from sklearn.decomposition import PCA
import os

from tsptw_with_ortools import Solver
from config import get_config, print_config


# Compute a sequence's reward
def reward(tsptw_sequence,speed):
    # Convert sequence to tour (end=start)
    tour = np.concatenate((tsptw_sequence,np.expand_dims(tsptw_sequence[0],0)))
    # Compute tour length
    inter_city_distances = np.sqrt(np.sum(np.square(tour[:-1,:2]-tour[1:,:2]),axis=1))
    distance = np.sum(inter_city_distances)
    # Compute develiry times at each city and count late cities
    elapsed_time = -10
    late_cities = 0
    for i in range(tsptw_sequence.shape[0]-1):
        travel_time = inter_city_distances[i]/speed
        tw_open = tour[i+1,2]
        tw_close = tour[i+1,3]
        elapsed_time += travel_time
        if elapsed_time <= tw_open:
            elapsed_time = tw_open
        elif elapsed_time > tw_close:
            late_cities += 1
    # Reward
    return distance + 100000000*late_cities

# Swap city[i] with city[j] in sequence
def swap2opt(tsptw_sequence,i,j):
    new_tsptw_sequence = np.copy(tsptw_sequence)
    new_tsptw_sequence[i:j+1] = np.flip(tsptw_sequence[i:j+1], axis=0) # flip or swap ?
    return new_tsptw_sequence

# One step of 2opt  = one double loop and return first improved sequence
def step2opt(tsptw_sequence,speed):
    seq_length = tsptw_sequence.shape[0]
    distance = reward(tsptw_sequence,speed)
    for i in range(1,seq_length-1):
        for j in range(i+1,seq_length):
            new_tsptw_sequence = swap2opt(tsptw_sequence,i,j)
            new_distance = reward(new_tsptw_sequence,speed)
            if new_distance < distance:
                return new_tsptw_sequence
    return tsptw_sequence



class DataGenerator(object):

    # Initialize a DataGenerator
    def __init__(self,config):
        self.batch_size = config.batch_size
        self.dimension = config.input_dimension
        self.max_length = config.max_length
        self.speed = config.speed
        self.kNN = config.kNN # int for random k_nearest_neighbor
        self.width = config.width_mean, config.width_std # time window width gaussian distribution [mean,std]
        self.pretrain = config.pretrain
        # Create Solver and Data Generator
        self.solver = Solver(self.max_length, self.speed) # reference solver for TSP-TW (Google_OR_tools)

    # Solve an instance with reference solver
    def solve_instance(self, sequence, tw_open, tw_close):
        # Calculate distance matrix
        precision = 1 #20 #int(self.speed)
        dist_array = pdist(sequence)
        dist_matrix = squareform(dist_array)
        # Call OR Tools to solve instance
        demands = np.zeros(tw_open.size)
        tour, tour_length, delivery_time = self.solver.run(precision*dist_matrix, demands, precision*(1+tw_open), precision*(1+tw_close)) # a tour is a permutation + start_index    # Rq: +1 for depot offset
        tour_length= tour_length/precision
        delivery_time = np.asarray(delivery_time)/precision - 1 # offset -1 because depot opens at -1
        return tour[:-1], tour_length, delivery_time


    # Iterate step2opt max_iter times
    def loop2opt(self, tsptw_sequence, max_iter=2000, speed=1.):
        best_reward = reward(tsptw_sequence,speed)
        new_tsptw_sequence = np.copy(tsptw_sequence)
        for _ in range(max_iter): 
            new_tsptw_sequence = step2opt(new_tsptw_sequence,speed)
            new_reward = reward(new_tsptw_sequence,speed)
            if new_reward < best_reward:
                best_reward = new_reward
            else:
                break
        return new_tsptw_sequence, best_reward

    def get_tour_length(self, sequence):
        # Convert sequence to tour (end=start)
        tour = np.concatenate((sequence,np.expand_dims(sequence[0],0)))
        # Compute tour length
        inter_city_distances = np.sqrt(np.sum(np.square(tour[:-1]-tour[1:]),axis=1))
        return np.sum(inter_city_distances)

    # Reorder sequence with random k NN (TODO: Less computations)
    def k_nearest_neighbor(self, sequence):
        # Calculate dist_matrix
        dist_array = pdist(sequence)
        dist_matrix = squareform(dist_array)
        # Construct tour
        new_sequence = [sequence[0]]
        current_city = 0
        visited_cities = [0]
        for i in range(1,len(sequence)):
            j = np.random.randint(0,min(len(sequence)-i,self.kNN))
            next_city = [index for index in dist_matrix[current_city].argsort() if index not in visited_cities][j]
            visited_cities.append(next_city)
            new_sequence.append(sequence[next_city])
            current_city = next_city
        return np.asarray(new_sequence)


    # Generate random TSP-TW instance
    def gen_instance(self, test_mode=True, seed=0):
        if seed!=0: np.random.seed(seed)
        # Randomly generate (max_length+1) city integer coordinates in [0,100[      # Rq: +1 for depot
        sequence = np.random.randint(100, size=(self.max_length+1, self.dimension))
        if self.pretrain == False:
            sequence = self.k_nearest_neighbor(sequence) # k nearest neighbour tour (reverse order - depot end)
        # Principal Component Analysis to center & rotate coordinates
        pca = PCA(n_components=self.dimension)
        sequence_ = pca.fit_transform(sequence)
        # TW constraint 1 (open time)
        if self.pretrain == True:
            tw_open = np.random.randint(100, size=(self.max_length, 1)) # t_open random integer in [0,100[
            tw_open = np.concatenate((tw_open,[[-1]]), axis=0) # depot opens at -1
            tw_open[::-1].sort(axis=0) # sort cities by TW open constraint (reverse order)
        else: # Open time defined by kNN tour
            ordered_seq = sequence[::-1]
            inter_city_distances = np.sqrt(np.sum(np.square(ordered_seq[1:]-ordered_seq[:-1]),axis=1))
            time_at_cities = np.cumsum(inter_city_distances/self.speed,axis=0)
            time_at_cities = np.expand_dims(np.floor(time_at_cities).astype(int),axis=1)
            tw_open = np.concatenate(([[-1]],time_at_cities), axis=0) # depot opens at -1
            tw_open = tw_open[::-1] # TW open constraint sorted (reverse order)   Rq: depot = tw_open[-1], tw_width[-1] and sequence[-1]
        # TW constraint 2 (time width): Gaussian or uniform distribution
        tw_width = np.abs(np.random.normal(loc=self.width[0], scale=self.width[1], size=(self.max_length, 1)))  # gaussian distribution
        tw_width = np.concatenate((tw_width,[[1]]), axis=0) # depot opened for 1
        tw_width = np.ceil(tw_width).astype(int)
        tw_close = tw_open+tw_width
        # TW feature 1 = Centered mean time (invariance)
        tw_mean_ = (tw_open+tw_close)/2
        tw_mean_ -= np.mean(tw_mean_)
        # TW feature 2 = Width
        tw_width_ = tw_width
        print(tw_width_)
        # Concatenate input (sorted by time) and scale to [0,1[
        input_ = np.concatenate((sequence_,tw_mean_,tw_width_), axis=1)/100
        if test_mode == True:
            return input_, sequence, tw_open, tw_close
        else:
            return input_



    # Generate random batch for training procedure
    def train_batch(self):
        input_batch = []
        for _ in range(self.batch_size):
            # Generate random TSP-TW instance
            input_ = self.gen_instance(test_mode=False)
            # Store batch
            input_batch.append(input_)
        return input_batch


    # Generate random batch for testing procedure
    def test_batch(self, seed=0):
        # Generate random TSP-TW instance
        input_, or_sequence, tw_open, tw_close = self.gen_instance(test_mode=True, seed=seed)
        # Store batch
        input_batch = np.tile(input_,(self.batch_size,1,1))
        return input_batch, or_sequence, tw_open, tw_close


    # Plot a tour
    def visualize_2D_trip(self,trip,tw_open,tw_close):
        plt.figure(figsize=(30,30))
        rcParams.update({'font.size': 22})
        # Plot cities
        colors = ['red'] # Depot is first city
        for i in range(len(tw_open)-1):
            colors.append('blue')
        plt.scatter(trip[:,0], trip[:,1], color=colors, s=200)
        # Plot tour
        tour=np.array(list(range(len(trip))) + [0])
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--", markersize=100)
        # Annotate cities with TW
        tw_open = np.rint(tw_open)
        tw_close = np.rint(tw_close)
        time_window = np.concatenate((tw_open,tw_close),axis=1)
        for tw, (x, y) in zip(time_window,(zip(X,Y))):
            plt.annotate(tw,xy=(x, y))  
        plt.xlim(0,60)
        plt.ylim(0,60)
        plt.show()


    # Heatmap of permutations (x=cities; y=steps)
    def visualize_sampling(self,permutations):
        max_length = len(permutations[0])
        grid = np.zeros([max_length,max_length]) # initialize heatmap grid to 0
        transposed_permutations = np.transpose(permutations)
        for t, cities_t in enumerate(transposed_permutations): # step t, cities chosen at step t
            city_indices, counts = np.unique(cities_t,return_counts=True,axis=0)
            for u,v in zip(city_indices, counts):
                grid[t][u]+=v # update grid with counts from the batch of permutations
        # plot heatmap
        fig = plt.figure()
        rcParams.update({'font.size': 22})
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(grid, interpolation='nearest', cmap='gray')
        plt.colorbar()
        plt.title('Sampled permutations')
        plt.ylabel('Time t')
        plt.xlabel('City i')
        plt.show()

    # Heatmap of attention (x=cities; y=steps)
    def visualize_attention(self,attention):
        # plot heatmap
        fig = plt.figure()
        rcParams.update({'font.size': 22})
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(attention, interpolation='nearest', cmap='hot')
        plt.colorbar()
        plt.title('Attention distribution')
        plt.ylabel('Step t')
        plt.xlabel('Attention_t')
        plt.show()

    def load_Dumas(self,dir_='n20w100'):
        dataset = {}
        shrinkage = 80
        for file_name in os.listdir('benchmark/'+dir_):
            if 'solution' in file_name: continue
            # Gather data
            data = open('benchmark/'+dir_+'/'+file_name, 'r')
            x,y,t_open,t_close = [],[],[],[]
            for i,line in enumerate(data):
                if i>5:
                    line = line.split()
                    if line[0]!='999':
                        x.append(int(float(line[1])))
                        y.append(int(float(line[2])))
                        t_open.append(int(float(line[4])))
                        t_close.append(int(float(line[5])))
            # TW constraint 1 (open time)
            t_open = np.asarray(t_open) # open time
            sorted_index = np.argsort(t_open)[::-1] # sort cities by TW open constraint (reverse order)
            tw_open = t_open[sorted_index]
            tw_open = np.expand_dims(tw_open,axis=1)
            tw_open_ = shrinkage*tw_open/tw_open[0]-1  # scale open time in [0,100[ (depot opens at -1)
            # TW constraint 2 (close time) ############################### RESCALE ??
            t_close = np.asarray(t_close)-1 ############################### depot ?
            tw_close = t_close[sorted_index]
            tw_close = np.expand_dims(tw_close,axis=1)
            tw_close_ = shrinkage*tw_close/tw_open[0]-1 # scale close time
            tw_close_[-1] = 0 # depot open till 0
            # Coordinates    
            seq = np.stack((x,y),axis=1) # city integer coordinates in [0,100[ ############################### RESCALE ??
            sequence = seq[sorted_index]
            pca = PCA(n_components=self.dimension) # Principal Component Analysis to center & rotate coordinates
            sequence_ = pca.fit_transform(sequence)
            sequence_ = self.speed*shrinkage*sequence_/tw_open[0] # scale sequence
            # TW feature 1 = Centered mean time (invariance)
            tw_mean_ = (tw_open_+tw_close_)/2
            tw_mean_ -= np.mean(tw_mean_)
            # TW feature 2 = Width
            tw_width_ = tw_close_-tw_open_
            # Concatenate input (sorted by time) and scale to [0,1[
            input_ = np.concatenate((sequence_,tw_mean_,tw_width_), axis=1)/100
            # Gather solution
            solution = open('benchmark/'+dir_+'/'+file_name+'.solution', 'r')
            for i,line in enumerate(solution):
                if i==0: opt_permutation = np.asarray(line.split()).astype(int)-1
                if i==1: opt_length = int(line.split()[0])
            opt_length = self.get_tour_length(seq[opt_permutation])/100
            # Save data
            dataset[file_name]={'input_': input_, 'sequence':sequence, 'tw_open':tw_open, 'tw_close':tw_close, 'optimal_sequence':seq[opt_permutation],
                                'optimal_tw_open':np.expand_dims(t_open[opt_permutation],axis=1), 'optimal_tw_close':np.expand_dims(t_close[opt_permutation],axis=1), 'optimal_length':opt_length}

        return dataset









if __name__ == "__main__":

    # Config
    config, _ = get_config()
    dataset = DataGenerator(config)

    # Generate some data
    #input_batch = dataset.train_batch()
    input_batch, or_sequence, tw_open, tw_close = dataset.test_batch(seed=0)
    print()

    # Some print
    #print('Input batch: \n',100*input_batch)
    #print(np.rint(np.mean(100*input_batch,1)))

    # 2D plot for coord batch
    #dataset.visualize_2D_trip(or_sequence[::-1], tw_open[::-1], tw_close[::-1])

    # Solve to optimality and plot solution    
    #or_permutation, or_tour_length, or_delivery_time = dataset.solve_instance(or_sequence, tw_open, tw_close)
    #print('Solver tour length: \n', or_tour_length/100)
    #print('Time var: \n', or_delivery_time)
    #dataset.visualize_2D_trip(or_sequence[or_permutation], tw_open[or_permutation], tw_close[or_permutation])
    #dataset.visualize_sampling([or_permutation])

    dataset.load_Dumas()