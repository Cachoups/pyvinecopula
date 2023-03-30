import itertools
import numpy as np
import pyvinecopulib as pv


import itertools

class VinecopSearch:

  #----------------------------------------#
  # This class is for building a Vine copula using Kendall criteria for constuction
  #----------------------------------------#

  # Initiate class with :
  # d : number of dimension
  # order : array variable order 
  # controls : controls for bicop fitting
  # pair_copula : pair copula of our copula
  # structure : structure of our copula (R vine structure)
  # bicop_init : all bicop model each possible pairs from tree 0
  # all_couple_init : its couple associated
    def __init__(self, d):
        self.d = d
        self.order = np.array(range(1,d+1))
        self.controls = pv.FitControlsBicop(num_threads = 32)
        self.pair_copula = []
        self.structure = np.zeros((d,d))
        self.bicop_init = []
        self.all_couple_init = []
        self.n = 0
        
        
    # Generate all possible pairs
    # There are (d-t)!/(2!(d-t-2)!) possibilities
    # t : tree level
    def pairs(self, t):
        all_couple = []
        for pair in itertools.combinations(range(1,self.d-t+1),2):
            all_couple.append(pair)
        return all_couple
    
    # Conditional sampling
    # data : 1 variable data that we know (uniformly distributed)
    # bicop : bicop
    # n : length of data
    # h1 : if h1 = True then compute hinv1 else : compute hinv2
    def sampling_cond(self, data, bicop, h1=True):
        n = self.n
        # unif_random_1 = np.random.uniform(n) # for u0 (resp. u1)
        unif_random = np.random.uniform(0,1,n) # for t1 (resp. t0)

        if h1 :
            u1 = bicop.hinv1(np.vstack((data, unif_random)).T)
            return u1
        else :
            u0 = bicop.hinv2(np.vstack((unif_random, data)).T)
            return u0
        
    # Create bicop from all posible couple
    def bicop(self, t, all_couple, data) :
        pairs = []
        
        # For tree > 0
        if t != 0:
            all_couple_init = self.all_couple_init
            new_all_couple = []
            for edge in all_couple :
                # We keep node where they have t variables in common and get its bicop
                if len(edge) == 2+t :
                    a = edge[0]
                    b = edge[1]
                    condition = (all_couple_init[:,0]==edge[0]) & (all_couple_init[:,1]==edge[1])
                    argcouple = np.where(condition)[0][0]
                    bicop = self.bicop_init[argcouple]
                    pairs.append(bicop)
                    new_all_couple.append(edge)
            all_couple = new_all_couple
            
        # For tree 0
        else :
            for edge in all_couple:
                a = edge[0]
                b = edge[1]
                bicop = pv.Bicop(data[:, [a-1,b-1]], self.controls)
                pairs.append(bicop)
            self.bicop_init = pairs
            self.all_couple_init = all_couple
            
        return all_couple, pairs
    
    # Create bicop from all posible couple
    def bicop_(self, t, preivous_tree, previous_tree_couple ,all_couple, data) :
        pairs = []
        
        # For tree > 0
        if t != 0:
            new_all_couple = []
            for edge in all_couple :
                # We keep node where they have t variables in common and get its bicop
                if len(edge) == 2+t :
                    a = edge[0]
                    b = edge[1]
                    bool1 = True
                    bool2 = True
                    i = 2
                    while bool1 or bool2 :
                        couple1 = np.sort([a,edge[i]])
                        couple2 = np.sort([b,edge[i]])
                        condition1 = (previous_tree_couple[:,0]==couple1[0]) & (previous_tree_couple[:,1]==couple1[1])
                        condition2 = (previous_tree_couple[:,0]==couple2[0]) & (previous_tree_couple[:,1]==couple2[1])
                        argcouple1 = np.where(condition1)[0]
                        argcouple2 = np.where(condition2)[0]
                        if len(argcouple1) != 0:
                            if couple1[0] == a :
                                h1 = False
                            else :
                                h1 = True
                            bicop = preivous_tree[argcouple1.item()]
                            sampling_data_1 = self.sampling_cond(data[:,edge[i]-1], bicop, h1)
                            bool1 = False
                        if len(argcouple2) != 0:
                            if couple1[0] == b :
                                h1 = False
                            else :
                                h1 = True
                            bicop = preivous_tree[argcouple2.item()]
                            sampling_data_2 = self.sampling_cond(data[:,edge[i]-1], bicop, h1)
                            bool2 = False
                        i+=1
                            
                    bicop = pv.Bicop(np.vstack((sampling_data_1, sampling_data_2)).T , self.controls)
                    pairs.append(bicop)
                    new_all_couple.append(edge)
            all_couple = new_all_couple
            
        # For tree 0
        else :
            for edge in all_couple:
                a = edge[0]
                b = edge[1]
                bicop = pv.Bicop(data[:, [a-1,b-1]], self.controls)
                pairs.append(bicop)
            self.bicop_init = pairs
            self.all_couple_init = all_couple
            
        return all_couple, pairs
    
    # Compute kendall value for each bicop
    def tau(self, pairs):
        tau = []
        for i in range(len(pairs)) :
                parameter = pairs[i].parameters
                tau.append(pairs[i].parameters_to_tau(parameter))
        return tau
    
    # Replace variable by its new naming if order different
    def replace(self, tab ):
        for i in range(len(tab)):
            tab[i] = np.where(self.order == tab[i])[0][0] + 1
        tab[0:2] = np.sort(tab[0:2])
        # tab[2:] = np.sort(tab[2:])
        return tab
    
    # Reorder tree_couple
    def reorder_tree_couple(self, tree_couple):
        for i in range(len(tree_couple)) :
            tree_couple[i] = self.replace(tree_couple[i])
        return tree_couple
    
    # For each pair copula generated for next tree, find 
    def all_couple(self, tree_couple, all_couple):
        new_all_couple = []
        for pair in all_couple:
            a = tree_couple[pair[0]-1]
            b = tree_couple[pair[1]-1]
            node = list((set(a) | set(b)) - (set(a) & set(b))) + list((set(a) & set(b)))
            node[0:2] = np.sort(node[0:2])
            new_all_couple.append(node) ## Find not matching variables + mathching one on side
        return new_all_couple
    
    # Build tree with no cycle
    def build_tree(self, t, diff, tree, all_couple):
        i = 1
        tree_couple = [all_couple[tree[0]]]
        node_visited = [all_couple[tree[0]][0]]
        while len(tree_couple) < self.d-t-1:
            rank = tree[i]
            edge = all_couple[rank]
            tmp = list((set(diff) | set(edge)) - (set(diff) & set(edge)))
            # if edge[0] in node_visited:
            #     print("edge : ",edge)
            #     edge[0:2] = np.flip(edge[0:2])
            if (len(tmp) > 0) and (edge[0] not in node_visited):
                tree_couple.append(edge)
                node_visited.append(edge[0])
                diff = tmp
                i+=1
            else :
                del tree[i]
                # i+=1
        tree = tree[:self.d-t]
        return tree, tree_couple
    
    # Fill structure for a tree
    def fill_structure(self, t, tree_couple):
        # (M[d - 1 - e, e], M[t, e]; M[t - 1, e], ..., M[0, e])  For all t = 1, ..., d - 2 and e = 0, ..., d - t - 1
        for e in range(self.d-t-1):
            couple = tree_couple[e]
            self.structure[self.d - 1 - e, e] = couple[0]
            self.structure[t, e] = couple[1]
            # for i in range(t-1):
            #     self.structure[i,e] = couple[2+t-1-i]
    
    # Define first tree
    def tree_init(self,data):
        self.n = len(data)
        
        t = 0
        all_couple = np.array(self.pairs(t))
    
        all_couple, pairs = self.bicop(t, all_couple, data)
        # all_couple, pairs = self.bicop_(t, None, None, all_couple, data)
        tau = self.tau(pairs)

        tree = np.argsort(-np.abs(tau))

        all_couple = np.array(all_couple)

        diff = all_couple[tree[0]].tolist()
        tree, tree_couple = self.build_tree(t, diff, tree.tolist(), all_couple)

        pairs = np.array(pairs)[tree].tolist()
        # Define order variable
        # order = np.concatenate(tree_couple)
        # _, idx = np.unique(order, return_index=True)
        # order = order[np.sort(idx)]
        # self.order = order
        
        
        # Reorder our couples
        # tree_couple = self.reorder_tree_couple(tree_couple)
        order_tree = sorted(range(len(tree_couple)), key=lambda k: tree_couple[k][0])
        pairs = np.array(pairs)[order_tree]
        
        # Add pair copula from first tree
        self.pair_copula.append(pairs)
        
        tree_couple = np.array(tree_couple)[order_tree]
        # Fill structure with first tree
        self.fill_structure(t,tree_couple)
        print(tree_couple)
        # tree_couple = np.array(tree_couple)[order_tree]
        return tree_couple
        
    # Define construction of tree above lvl 1
    def tree(self, t, tree_couple, data):
        all_couple = self.pairs(t)
        all_couple = self.all_couple(tree_couple, all_couple)
        all_couple, pairs = self.bicop(t, all_couple,data)
        # all_couple, pairs = self.bicop_(t, self.pair_copula[t-1], tree_couple, all_couple,data)
        tau = self.tau(pairs)
        tree = np.argsort(-np.abs(tau))
        
        all_couple = np.array(all_couple)
        diff = all_couple[tree[0]]
        tree, tree_couple = self.build_tree(t, diff, tree.tolist(), all_couple)
        pairs = np.array(pairs)[tree].tolist()
        # Reorder our couples
        tree_couple = self.reorder_tree_couple(tree_couple)
        order_tree = sorted(range(len(tree_couple)), key=lambda k: tree_couple[k][0])
        pairs = np.array(pairs)[order_tree]
        # Add pair copula from first tree
        self.pair_copula.append(pairs)
        
        tree_couple = np.array(tree_couple)[order_tree]
        # Fill structure with first tree
        self.fill_structure(t,tree_couple)
        print(tree_couple)
        return tree_couple
    
    def reset(self):
        d = self.d
        self.order = np.array(range(1,d+1))
        self.controls = pv.FitControlsBicop(num_threads = 32)
        self.pair_copula = []
        self.structure = np.zeros((d,d))
        self.bicop_init = []
        self.all_couple_init = []
        self.n = 0
        
    # Main function to build the entire copula
    def main(self, data):
        self.reset()
        for t in range(self.d-1):
            if t == 0:
                tree_couple = self.tree_init(data)
            else :
                tree_couple = self.tree(t, tree_couple, data)
        # finish filling structure
        d = self.d
        self.structure[0,d-1] = 0
        a = set(np.array([self.structure[d -1 -i, i] for i in range(d)]))
        b =set(range(0,d+1))
        element = list((a | b) - (a & b))
        self.structure[0,d-1] = element[0]
        return 0
    
    # Return vine copula object from a pre define structure and pair copula
    def create_model(self):
        return pv.Vinecop(self.structure, self.pair_copula)
    
    def save(self,model, filename):
        model.to_json(filename)
        return 0