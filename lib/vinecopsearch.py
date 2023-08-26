import itertools
import numpy as np
import pyvinecopulib as pv
import scipy.stats as stats
import json
import itertools
from collections import namedtuple
from sklearn.metrics import mutual_info_score
import os
import concurrent.futures
def read_paircopula(list_bicop, level):
    pair_copula = []
    if level == "all":
        if list_bicop != [] :
            t = len(list_bicop)
            pair_copula = []
            for i in range(t):
                pair_copula_t = []
                list_t_bicop = list_bicop[i]
                n = len(list_bicop[i])
                for j in range(n):
                    dict_bicop = list_t_bicop[j]
                    json_object = json.dumps(dict_bicop)
                    with open("./output/tmp.json", "w") as outfile:
                        outfile.write(json_object)
                    bicop = pv.Bicop("./output/tmp.json")
                    os.remove("./output/tmp.json")
                    pair_copula_t.append(bicop)

                pair_copula.append(pair_copula_t)
    else :
        indep = len(list_bicop)
        for i in range(indep):
            tree_indep = []
            tree = list_bicop[i]
            t = len(tree)
            for i in range(t):
                pair_copula_t = []
                list_t_bicop = tree[i]
                n = len(list_t_bicop)
                for j in range(n):
                    dict_bicop = list_t_bicop[j]
                    json_object = json.dumps(dict_bicop)
                    with open("./output/tmp.json", "w") as outfile:
                        outfile.write(json_object)
                    bicop = pv.Bicop("./output/tmp.json")
                    os.remove("./output/tmp.json")
                    pair_copula_t.append(bicop)

                tree_indep.append(pair_copula_t)
            pair_copula.append(tree_indep)
    return pair_copula
    
class VinecopSearch:

  #----------------------------------------#
  # This class is for building a Vine copula using Kendall criteria for constuction
  #----------------------------------------#

  # Initiate class with :
  # d : number of dimension
  # order : array variable order 
  # controls : controls for bicop fitting
  # pair_copula : pair copula of our copula
  # node_tree : list of tree nodes (list of variables)
  # couple_tree : the corresponding couple that indicates pair nodes from previous tree that became an edge
  # weight : list of weights of each node for each tree
  # structure : structure of our copula (R vine structure)
  # bicop_init : all bicop model each possible pairs from tree 0
  # metric_init : list of values of  the chosen metric
  # all_couple_init : its couple associated
  # t_max : max int value for the tree
  # metric : "tau" or "mi"
  # bins : int for bins for "mi"
    def __init__(self, d = 0, pair_copula = [], node_tree = [], couple_tree = [], weight = [],  metric_init = [], all_couple_init = [], structure = [], permutation = [], tree_indep = [],  metric = "tau", threshold = False, threshold_value = 0, level = 'all', bins = 1, t_max = 0, filename = ''):
        try : 
            self.read_json(filename)
        except :
            self.d = d
            self.pair_copula = read_paircopula(pair_copula, level)
            self.node_tree = node_tree
            self.couple_tree = couple_tree
            self.weight = weight
            self.structure = structure
            self.bicop_init = []
            self.metric_init = metric_init
            self.all_couple_init = all_couple_init
            self.t_max = t_max
            self.metric = metric
            self.bins = bins
            self.threshold = threshold
            self.threshold_value = threshold_value
            self.level = level
            self.permutation = permutation
            self.tree_indep = tree_indep
        
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
        controls = pv.FitControlsBicop(num_threads = 32)
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
                bicop = pv.Bicop(data[:, [a-1,b-1]], controls)
                pairs.append(bicop)
            self.bicop_init = pairs
            self.all_couple_init = all_couple
            
        return all_couple, pairs
    
    # Create bicop from all posible couple
    def bicop_(self, t, preivous_tree, previous_tree_couple ,all_couple, data) :
        pairs = []
        control = pv.FitControlsBicop(num_threads = 32)
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
                            
                    bicop = pv.Bicop(np.vstack((sampling_data_1, sampling_data_2)).T , control)
                    pairs.append(bicop)
                    new_all_couple.append(edge)
            all_couple = new_all_couple
            
        # For tree 0
        else :
            for edge in all_couple:
                a = edge[0]
                b = edge[1]
                bicop = pv.Bicop(data[:, [a-1,b-1]], control)
                pairs.append(bicop)
            self.bicop_init = pairs
            self.all_couple_init = all_couple
            
        return all_couple, pairs
    
    def bicop__(self, tree_couple, data) :
        pairs = []
        control = pv.FitControlsBicop(num_threads = 32)
        for edge in tree_couple:
            a = edge[0]
            b = edge[1]
            bicop = pv.Bicop(data[:, [a-1,b-1]], control)
            pairs.append(bicop)
            
        return pairs
    
    # Compute kendall tau value for each bicop
    def tau(self, pairs):
        tau = []
        for i in range(len(pairs)) :
                parameter = pairs[i].parameters
                tau.append(pairs[i].parameters_to_tau(parameter))
        return tau
    
    # Compute kendall tau or mi value from data
    def metric_data(self, x, y, metric, bins):
        if metric == "tau":
            tau, p_value = stats.kendalltau(x, y)
            return tau
        else :
            c_xy = np.histogram2d(x, y, bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            return mi
    
        
    # Compute  chosen metric for each pairs
    def all_metric(self, all_couple, data, metric, bins):
        metrics = []
        for pair in all_couple:
            x,y = pair
            metrics.append(self.metric_data(data[:,x-1], data[:,y-1], metric, bins))
        self.metric_init = metrics
        self.all_couple_init = all_couple
        return metrics
    
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
    def all_couple(self, t, tree_couple, all_couple):
        new_all_couple = []
        new_idx_couple = []
        for pair in all_couple:
            try : 
                a = tree_couple[pair[0]-1]
                b = tree_couple[pair[1]-1]
                node = list((set(a) | set(b)) - (set(a) & set(b))) + list((set(a) & set(b)))
                node[0:2] = np.sort(node[0:2])
                if len(node) <= 2 + t:
                    new_all_couple.append(node) ## Find not matching variables + mathching one on side
                    new_idx_couple.append(pair)
            except :
                None

        
        return new_all_couple, new_idx_couple
    
    # Build tree with no cycle
    def build_tree(self, t, diff, tree, metrics, all_couple, idx_all_couple, threshold, level):
        i = 1
        tree_couple = [all_couple[tree[0]]]
        node_visited = [all_couple[tree[0]][0]]
        idx_couple = [idx_all_couple[tree[0]]]
        tree_indep = [[all_couple[tree[0]][0], all_couple[tree[0]][1]]]
        if threshold == False : 
            condition = len(tree_couple) < self.d-t-1
        else :
            if level == "all":
                try :
                    condition = (len(tree_couple) < self.d-t-1) and (metrics[tree[i]] != 0)
                except :
                    condition = False
            elif level =="0":
                if t == 0:
                    try :
                        condition = (len(tree_couple) < self.d-t-1) and (metrics[tree[i]] != 0)
                    except :
                        condition = False
                else :
                    condition = len(tree_couple) < self.d-t-1
            else :
                raise AssertionError
        while condition:
            try :
                rank = tree[i]
                edge = all_couple[rank]
            
                tmp = list((set(diff) | set(edge)) - (set(diff) & set(edge)))
                if (len(tmp) > 0) and (edge[0] not in node_visited) and (len(node_visited) <= self.d - t):
                    if threshold == True:
                        if (metrics[rank] != 0) or (level == "0"):
                            tree_couple.append(edge)
                            idx_couple.append(idx_all_couple[rank])
                            node_visited.append(edge[0])
                            diff = tmp
                            i+=1
                        else :
                            del tree[i]
                    else :
                        tree_couple.append(edge)
                        idx_couple.append(idx_all_couple[rank])
                        node_visited.append(edge[0])
                        diff = tmp
                        i+=1
                else :
                    del tree[i]
            except :
                condition = False
            
            if threshold == True:
                if level == "all":
                    try:
                        condition = condition and (len(tree_couple) < self.d-t-1) and (metrics[tree[i]] != 0)
                    except :
                        condition = False
                elif level == '0':
                    if t == 0:
                        try:
                            condition = condition and (len(tree_couple) < self.d-t-1)  and (metrics[tree[i]] != 0)
                        except :
                            condition = False
                    else :
                        condition = condition and (len(tree_couple) < self.d-t-1) 
            else :
                condition = condition and (len(tree_couple) < self.d-t-1) 
        tree = tree[:self.d-t]
        return tree, tree_couple, idx_couple
    
    def fill_tree_indep(self, tree_couple):
        tree_couple = tree_couple.tolist()
        tree_indep = [[tree_couple[0][0], tree_couple[0][1]]]
        del tree_couple[0]
        i = 0
        while len(tree_couple) != 0:
            j = 0
            condition_tree = True
            while condition_tree:
                try :
                    edge = tree_couple[j]
                    current_tree = tree_indep[i]
                    if (edge[0] in current_tree):
                        if edge[1] not in current_tree :
                            tree_indep[i].append(edge[1])
                        del tree_couple[j]
                        j=0
                        
                    elif (edge[1] in current_tree):
                        if edge[0] not in current_tree :
                            tree_indep[i].append(edge[0])
                        del tree_couple[j]
                        j=0
                    else :
                        j+=1
                except:
                    try :
                        edge = tree_couple[0]
                        tree_indep.append([edge[0], edge[1]])
                        del tree_couple[0]
                        i+=1
                        condition_tree = False
                    except :
                        condition_tree = False
        self.tree_indep = tree_indep
        
    # Fill permutation matrix
    def fill_permutation(self, t, tree_couple):
        for e in range(len(tree_couple)):
            if t == 0:
                couple = tree_couple[e]
                self.permutation[couple[0] - 1, t] = couple[1]
            else :
                couple = tree_couple[e]
                try :
                    self.permutation[couple[0] -1, t] = couple[1]
                except : 
                    None
                
    # Fill structure for a tree
    def fill_structure(self, t, tree_couple):
        # (M[d - 1 - e, e], M[t, e]; M[t - 1, e], ..., M[0, e])  For all t = 1, ..., d - 2 and e = 0, ..., d - t - 1
        for e in range(self.d-t-1):
            try:
                couple = tree_couple[e]
                self.structure[self.d - 1 - e, e] = couple[0]
                self.structure[t, e] = couple[1]
            except:
                continue
    
        
    # Perform a threshold onchosen metric values for each pair
    def threshold_(self, value) :
        for i in range(len(self.metric_init)) :
            if abs(self.metric_init[i]) <= value :
                self.metric_init[i] = 0
        return 0
    
    
     # Define first tree
    def tree_init(self, data, bicop = True, threshold = False, threshold_value = 10**(-2), level = 'all', metric = "tau", bins = 1):
        self.n, d = data.shape
        
        t = 0
        all_couple = np.array(self.pairs(t))
    
        metrics = np.array(self.all_metric(all_couple, data, metric, bins))
        if threshold == True:
            tmp = self.metric_init.copy()
            self.threshold_(threshold_value)
            metrics = np.array(self.metric_init)
        tree = np.argsort(-np.abs(metrics))

        diff = all_couple[tree[0]].tolist()
        tree, tree_couple, idx_couple = self.build_tree(t, diff, tree.tolist(), metrics, all_couple, all_couple, threshold, level)

        if bicop and (level != '0') :
            pairs = self.bicop__(tree_couple, data)
        else :
            pairs = []

        order_tree = sorted(range(len(tree_couple)), key=lambda k: tree_couple[k][0])
        if bicop and (level != '0') :
            pairs = np.array(pairs)[order_tree]
        
            # Add pair copula from first tree
            self.pair_copula.append(pairs)
        tree_couple = np.array(tree_couple)[order_tree]
        tree = np.array(tree)[order_tree]
        
        if level == '0':
            self.fill_tree_indep(tree_couple)
                
        if (threshold == False):
            # Fill structure with first tree
            self.fill_structure(t,tree_couple)

        self.fill_permutation(t, tree_couple)
        self.couple_tree.append(np.array(idx_couple)[order_tree].tolist())
        self.node_tree.append(tree_couple.tolist())
        self.weight.append(metrics[tree])
        

        return tree_couple
        
    # Define construction of tree above lvl 1
    def tree(self, t, tree_couple, data, bicop = True, threshold = False, level ='all'):
        all_couple = self.pairs(t)
        all_couple, idx_all_couple = self.all_couple(t, tree_couple, all_couple)
        all_couple = np.array(all_couple, dtype=object)
        if len(all_couple) == 0:
            return []
        all_couple_init = np.array(self.all_couple_init)
        metrics = np.array(self.metric_init)[np.array([np.where( (all_couple_init[:,0] == all_couple[i][0]) & (all_couple_init[:,1]== all_couple[i][1]))[0][0] for i in range(len(all_couple)) ])]
        tree = np.argsort(-np.abs(metrics))
        diff = all_couple[tree[0]]
        tree, tree_couple, idx_couple = self.build_tree(t, diff, tree.tolist(), metrics, all_couple, idx_all_couple, threshold, level)
        if bicop and (level != '0'):
            pairs = self.bicop__(tree_couple, data)
        else :
            pairs = []
        # Reorder our couples
        tree_couple = self.reorder_tree_couple(tree_couple)
        order_tree = sorted(range(len(tree_couple)), key=lambda k: tree_couple[k][0])
        if bicop and (level != '0') :
            pairs = np.array(pairs)[order_tree]
            # Add pair copula from first tree
            self.pair_copula.append(pairs)
        
        tree_couple = np.array(tree_couple)[order_tree]
        tree = np.array(tree)[order_tree]
        if (threshold == False) or (level == '0'):
            # Fill structure with first tree
            self.fill_structure(t,tree_couple)

        self.fill_permutation(t, tree_couple)
        self.couple_tree.append(np.array(idx_couple)[order_tree].tolist())
        self.node_tree.append(tree_couple.tolist())
        self.weight.append(metrics[tree])
        return tree_couple
    
    def reset(self, metric, bins, level, threshold, threshold_value):
        d = self.d
        self.order = np.array(range(1,d+1))
        self.pair_copula = []
        self.structure = np.zeros((d,d))
        self.bicop_init = []
        self.all_couple_init = []
        self.metric_init = []
        self.metric = metric
        self.bins = bins
        self.level = level
        self.threshold = threshold
        self.threshold_value = threshold_value
        self.permutation = np.zeros((d,d), dtype = 'int')
        
    # Main function to build the entire copula
    # bicop : if True then compute the bicop for pairs else only search for the tree R vine structure
    # threshold : if True perform threshold on kendall's taus
    def main(self, data, bicop = True, threshold = False, threshold_value = 10**(-2), level = 'all' ,metric = "tau", bins = 1):
        self.reset(metric, bins, level, threshold, threshold_value)
        t = 0
        condition = True
        while condition:
            if t == 0:
                tree_couple = self.tree_init(data, bicop, threshold, threshold_value, level, metric, bins)
                # level = '0' : Do vinecop without threshold for each independant tree
                if level == '0':
                    self.structure = []
                    self.permuration = []
                    for indep in self.tree_indep:
                        vinecop_indep = VinecopSearch(len(indep))
                        vinecop_indep.main(data[:,np.sort(np.array(indep)-1) ], bicop = bicop, threshold=False, threshold_value= threshold_value, level = 'all')
                        self.structure.append(vinecop_indep.structure.copy())
                        self.permuration.append(vinecop_indep.permutation.copy())
                        self.pair_copula.append(vinecop_indep.pair_copula.copy())
                        self.couple_tree.append(vinecop_indep.couple_tree.copy())
                        self.node_tree.append(vinecop_indep.node_tree.copy())
                        self.weight.append(vinecop_indep.weight.copy())
                        t_max = vinecop_indep.t_max
                        self.t_max = max(self.t_max, t_max)
                    condition = False
            else :
                tree_couple = self.tree(t, tree_couple, data, bicop, threshold, level)
            t+=1
            condition = len(tree_couple) > 1
        if level != '0':
            self.t_max = t-1
        if threshold == False :
            # finish filling structure
            d = self.d
            self.structure[0,d-1] = 0
            a = set(np.array([self.structure[d -1 -i, i] for i in range(d)]))
            b =set(range(0,d+1))
            element = list((a | b) - (a & b))
            self.structure[0,d-1] = element[0]
        # change type of dataframe dependances
        return 0
    
    # Return vine copula object from a pre define structure and pair copula
    def create_model(self):
        return pv.Vinecop(self.structure, self.pair_copula)
    
    def save(self,model, filename):
        model.to_json(filename)
        return 0
    
    
    # convert our class object to json file
    def to_json(self, filename):
        json_string = json.dumps(self.__dict__, cls = NpEncoder)
        with open(filename, 'w') as outfile:
            outfile.write(json_string)
        return json.load(open(filename))
    
    def read_json(self, filename):
        model = json.load(open(filename))
        self.VinecopSearchDecoder(model)
    def VinecopSearchDecoder(self, obj):
        self.d = obj['d']
        self.pair_copula = read_paircopula(obj["pair_copula"], obj['level'])
        self.node_tree = obj["node_tree"]
        self.couple_tree = obj['couple_tree']
        self.weight = obj['weight']
        self.metric_init = obj['metric_init']
        self.all_couple_init = obj['all_couple_init']
        self.permutation = np.array(obj['permutation'])
        self.metric = obj['metric']
        self.threshold = obj['threshold']
        self.threshold_value = obj['threshold_value']
        self.level = obj['level']
        self.bins = obj['bins']
        self.t_max = obj['t_max']
        if self.level != '0':
            self.structure = np.array(obj['structure'])
        else :
            self.structure = obj['structure']
        self.tree_indep = obj['tree_indep']
        
    # pdf and cdf for no threshold and threshold first level
    def pdf(self, x):
        if self.threshold == False:
            m = pv.Vinecop(self.structure, self.pair_copula)
            return m.pdf(x)
        else : 
            if self.level == "0":
                p = 1
                tree_indep_list = self.tree_indep
                for i in range(0,len(tree_indep_list)):
                    d = len(tree_indep_list[i])
                    x_sub = x[np.array(tree_indep_list[i]) - 1]
                    sub_model = pv.Vinecop(self.structure[i], self.pair_copula[i])
                    p *= sub_model.pdf(x_sub)
                return p
            else :
                return 0
    # for level 0 : as mutual independant tree, cdf equals product of sub cdf ?
    def cdf(self, x):
        if self.threshold == False:
            m = pv.Vinecop(self.structure, self.pair_copula)
            return m.cdf(x)
        else : 
            if self.level == "0":
                c = 1
                tree_indep_list = self.tree_indep
                for i in range(0,len(tree_indep_list)):
                    d = len(tree_indep_list[i])
                    x_sub = x[np.array(tree_indep_list[i]) - 1]
                    sub_model = pv.Vinecop(self.structure[i], self.pair_copula[i])
                    c *= sub_model.cdf(x_sub)
                return c
            else :
                return 0
    def simulate(self, n):
        assert self.pair_copula != []
        if self.threshold == False:
            return pv.Vinecop(self.structure, self.pair_copula).simulate(n)
        if self.level == '0' :
            tree_indep = self.tree_indep
            d = self.d
            u = np.random.uniform(0,1,(n,d))
            def aux_tree_indep(i):
                vinecop = pv.Vinecop(np.array(self.structure[i]), self.pair_copula[i])
                simulated_data = vinecop.simulate(n)
                tree_indep[i].sort()
                list_variable = tree_indep[i]
                for j in range(len(list_variable)) :
                    u[:,list_variable[j]-1] = simulated_data[:,j]
            # Create a ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Apply the function to each item in parallel
                executor.map(aux_tree_indep, range(len(tree_indep)))
        else :
            def hinv1(bicop, value, A):
                return bicop.hinv1(np.column_stack((value, A)) )
            def hfunc2(bicop, value, B):
                return bicop.hfunc2(np.column_stack((value, B)) )
            def mc_integrate(func, bicop, a, b, value, dim, n = 1000):
                # Monte Carlo integration of given function over domain from a to b (for each parameter)
                # dim: dimensions of function
                res = b.copy()
                for i in range(dim):
                    b_ = b[i]
                    x_list = np.random.uniform(a, b_, (n, 1))
                    y = x_list.copy()
                    value_ = np.zeros((n,1))
                    value_[:,0] = value[i]

                    y = func(bicop, value_, x_list)

                    y_mean =  y.sum()/n
                    domain = np.power(b_-a, 1)
                        
                    res[i] = domain * y_mean
                    
                return res
            # Init 
            d = self.d
            A = np.zeros((d,d,n))
            B = np.zeros((d,d,n))
            w = np.random.uniform(0,1,(n,d))
            u = np.zeros((n,d))
            v = np.zeros((d,d,n))
            A[d-1,d-1,:] = w[:,d-1]
            B[d-1,d-1,:] = w[:,d-1]
            u[:,d-1] = w[:,d-1]
            v[d-1,d-1,:] = w[:,d-1]
            zeros = np.zeros(n)

            for i in range(1,d):
                try :
                    M = self.permutation[d-i-1,:]-1
                    M = M[np.where(M >= 0)[0]]
                except :
                    print("None")
                try :
                    n_ = len(M) # n = i
                    m = min(i,n_)
                    i_ = d - i - 1
                    v[i_, M[n_-1],:] = w[:,i_]
                    A[i_,d-1 -abs(i-n_),:] = w[:,i_]

                    for j in range(1,n_+1):
                        j_ = d - j - abs(i-n_) - 1

                        node_tree = np.array(self.node_tree[n_ - j])
                        pair_copula = np.array(self.pair_copula[n_ - j])
                        idx = np.where( (node_tree[:,0] == d-i) & (node_tree[:,1] == M[n_-j]+1) )
                        bicop = pair_copula[idx[0]][0]

                        try : 
                            idx = np.where((v[i_+1:,M[n_-j],:] != zeros).all(axis =1))[0][0]
                            value = v[i_+1:, M[n_-j],:][idx]
                            v[i_,M[n_-j-1],:]= bicop.hinv1(np.vstack((value, A[i_,j_+1,:])).T )               
                            A[i_,j_,:] = v[i_,M[n_-j-1],:]

                        except :
                            value = v[i_+1:, M[n_-j],:][np.where((v[i_+1:,M[n_-j],:] != zeros).all(axis =1))[0][0]]
                            A[i_,j_,:] = bicop.hinv1(np.vstack((value, A[i_,j_+1 ,:])).T)
                            
                    u[:,i_] = A[i_,i_,:]
                    v[i_,i_,:] = A[i_,i_,:]
                    B[i_,i_,:] = A[i_,i_,:]
                    
                    for j in range(n_-1, -1, -1):
                        j_ = d - j - abs(i- n_) -1
                        node_tree = np.array(self.node_tree[n_ - j - 1])
                        pair_copula = np.array(self.pair_copula[n_ - j - 1])
                        idx = np.where( (node_tree[:,0] == d-i) & (node_tree[:,1] == M[n_-j-1]+1) )
                        bicop = pair_copula[idx[0]][0]
                        try :
                            idx = np.where((v[M[n_-j-1], i_+1:,:] != zeros).all(axis =1))[0][0]
                            value = v[M[n_-j-1],i_+1:,:][idx]
                            v[M[n_-j-1],i_,:] = bicop.hfunc2(np.vstack((value, B[i_,j_ - 1,:])).T)
                            B[i_,j_,:] = v[M[n_-j-1],i_,:]
                        except :
                            value = v[M[n_-j-1], i_+1:,:][np.where((v[M[n_-j-1], i_+1:,:] != zeros).all(axis =1))[0][0]]
                            B[i_,j_,:] = bicop.hfunc2(np.vstack((value, B[i_,j_ - 1,:])).T)
    
                except:
                    u[:,d-i-1] = w[:,d-i-1]
                    v[d-i-1,d-i-1,:] =  w[:,d-i-1]
                    A[d-i-1,d-1,:] = w[:,d-i-1]
                    B[d-i-1,d-1,:] = w[:,d-i-1]
        return u 

### FUNCTIONS outside previous class

# Compute kendall tau or mi value from data
def metric_data(x, y, metric, bins):
    if metric == "tau":
        tau, p_value = stats.kendalltau(x, y)
        return tau
    else :
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi
    
        
# Compute  chosen metric for each pairs
def all_metric(all_couple, data, metric, bins):
    metrics = []
    for pair in all_couple:
        x,y = pair
        metrics.append(metric_data(data[:,x-1], data[:,y-1], metric, bins))
    return metrics

def all_metric_(nb_columns, data, metric, bins):
        all_couple = []
        for pair in itertools.combinations(range(1,nb_columns+1),2):
            all_couple.append(pair)
        return all_metric(all_couple, data, metric, bins)
    

# u to m space transformation

def F_j(data, t, j, n) :
        return (data[:,j] <= t).sum()/n
def F_(X):
    n,d = X.shape
    F = X.copy()
    for i in range(n) :
          for j in range(d) :
              t = X[i,j]
              F[i,j] = F_j(X, t, j,n)
    return F
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
# F is the table F_j(X_ij) value
# X is the table Xij
# U is the table Uij
def u_to_m(F,X ,U):
    
    n,d = U.shape
    M = U.copy()
    for i in range(n) :
          for j in range(d) :
                x= X[:,j]
                min_ = min(x)
                max_ = max(x)
                y= F[:,j]
                u, indices = np.unique(x, return_index=True)
                x=x[indices]
                y=y[indices]
                a = np.argsort(x)
                res = scipy.interpolate.PchipInterpolator(x[a], y[a]-(n+1)/n*U[i,j], axis=0, extrapolate=None)
                roots = res.roots()
                roots = roots[np.where((roots <= max_) & (min_ <= roots))]
                try :
                    random = np.random.randint(0, len(roots), size = 1)[0]
                    M[i,j] = roots[random]
                except :
                    value = find_nearest(F[:,j], (n+1)/n*U[i,j])
                    idx = np.where(F[:,j] == value)[0]
                    random = np.random.randint(0, len(idx), size = 1)[0]
                    M[i,j] = X[idx[random], j]
                        
    return M

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pv.Bicop):
            obj.to_json("./output/tmp.json")
            res = json.load(open("./output/tmp.json"))
            os.remove("./output/tmp.json")
            return res
        return super(NpEncoder, self).default(obj)
    
# plot structure as a graph from a vinecopsearch object
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pylab import rcParams
import pandas as pd

# model : vinecopsearch object
# t : int tree
def plot_structure(model, t):
    couple_tree = model.couple_tree[t]
    weight = model.weight[t]
    d = model.d
    # order = model.order per default the order is the natural one

    a = [] # first node of the edge
    b = [] # second node of the edge
    c = [] # weight for edge
    if t == 0 :
        node_tree = model.node_tree[0]
        for i in range(len(node_tree)):
            pair = node_tree[i]
            predecesor = pair[0]
            successor = pair[1]
            a.append(predecesor)
            b.append(successor)
            c.append(round(weight[i],2))
    else :
        node_tree = model.node_tree[t-1]
        for i in range(len(couple_tree)):
            predecesor, successor = couple_tree[i]
            predecesor_node = node_tree[predecesor-1]
            successor_node = node_tree[successor-1]
            a_node = str(predecesor_node[0])+","+str(predecesor_node[1])
            b_node = str(successor_node[0])+","+str(successor_node[1])
            if t != 1 :
                a_node+=";" +str(predecesor_node[2:])
                b_node+=";" +str(successor_node[2:])
            
            a.append(a_node)
            b.append(b_node)
            c.append(round(weight[i],2))
    structure_df = pd.DataFrame({"a" : a, "b" : b})
   
    paths = structure_df.loc[:,'a':].stack().groupby(level=0).agg(list).values.tolist()
    G = nx.DiGraph()
    for i in range(len(paths)):
        nx.add_path(G, paths[i], weight = c[i])
    return G,paths, c
