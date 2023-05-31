import numpy as np
import pandas as pd
import networkx as nx
from random import random
import streamlit as st
import matplotlib.pyplot as plt
import math
from collections import Counter
import time

st.set_page_config(layout="wide")


## Functions
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G))) 
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
def produce_eigenvalues(k,alpha=0,beta=64):
    delta=beta-alpha
    eigenvalues_list=[2*alpha-beta,alpha,beta,2*beta-alpha]
    for i in range(1,k+1):
        delta = delta/2
        eigenvalues_list[-1] += delta
        eigenvalues_list.append(eigenvalues_list[-1]+delta)
        eigenvalues_list.append(eigenvalues_list[0]-delta)
        eigenvalues_list.sort()
    return np.array(eigenvalues_list)
def next_diamater(T_0,T_1,edge_weight):

    d = nx.diameter(T_0)
    T = nx.union(T_0,T_1,rename=("", "T1{}".format(d))  )
    T.add_edge("0","T1{}0".format(d),weight=edge_weight)

    return T
def copy_graph(T,delta):
    T_delta = T.copy()
    T_delta.nodes["0"]["weight"]+=delta
    return(T_delta)
def recompute_edges(T,edge_weight):
    T["0"]['T1{}0'.format(max(0,nx.diameter(T)-2))]['weight'] = edge_weight
    return T

def get_ordered_vertices(T,root):
    
    lst=[[root]]
    computed_vertices=[]
    d = int(nx.diameter(T)/2)+1

    for i in range(d):
        new_lst=[]
        for parent in lst[-1]:
            computed_vertices.append(parent)
            

            for child in list(T.neighbors(parent)):
                if child not in computed_vertices:
                    new_lst.append(child)

        lst.append(new_lst)

    return lst

def compute_v0(T,_lambda_,root):

    a = get_ordered_vertices(T,root)
    eigenvalue = - _lambda_
    congruent_T = T.copy()
    d = math.ceil(nx.diameter(congruent_T)/2)

    for vertex in congruent_T.nodes:
        congruent_T.nodes.data()[vertex]['weight']+=eigenvalue

    for i in range(2,d+2):

        previous_level = a[-i+1]

        level = a[-i]

        for vertex in level:
            neighborhood = list(congruent_T.neighbors(vertex))

            if len(neighborhood) >= 1:
                children = list(set(previous_level) & set(neighborhood))
                weights = nx.get_node_attributes(congruent_T,'weight')
                summation = 0
                for child in children:
                    summation += (congruent_T[vertex][child]['weight'])**2/weights[child]
                congruent_T.nodes.data()[vertex]['weight'] -= summation 

    return nx.get_node_attributes(congruent_T,'weight')[root]

    

def get_subgraphs(T):
    leafs = []
    _ = [leafs.append(vertex[0]) for vertex in T.degree() if vertex[1]==1]

    children=[]
    _ = [children.append(child) for child in T.neighbors("0")]
    sub_root_vertex = []
    subgraph=[]

    d = int(nx.diameter(T)/2)+1
    _ = [sub_root_vertex.append([]) for i in range(d)]
    _ = [subgraph.append(["0"]) for i in range(d)]

    for distance in range(len(children)):
        sub_root_vertex[distance].extend([ children[distance] ])
        for leaf in leafs:
            path = nx.shortest_path(T,source=leaf, target=children[distance])
            if "0" not in path:
                subgraph[distance].extend(path)
        if distance>0:
            subgraph[distance].extend(subgraph[distance-1])
        subgraph[distance] = list(set(subgraph[distance]))
    return subgraph,sub_root_vertex

def compute_eigenvalues(T):
    node_weights = nx.get_node_attributes(T,'weight')
    weights=[]
    for i in node_weights:
        weights.append(node_weights[i])
    weights=np.array(weights)
    A = nx.to_scipy_sparse_array(T).toarray()
    np.fill_diagonal(A,weights)
    return A,np.round(np.linalg.eigh(A)[0],10)

def rearrange_edges(T,value,subgraph_vertices,free_edges,root=0):

    H =  T.subgraph(subgraph_vertices).copy()

    H.remove_edge(root,free_edges[0])
    S = [H.subgraph(subset).copy() for subset in nx.connected_components(H)]
    for i in range(2):
        if "0" in S[i].nodes:
            H_0 = S[i]
        elif free_edges[0] in S[i].nodes:
            H_1 = S[i]

    a_v0=compute_v0(H_0,value,"0")
    a_v1=compute_v0(H_1,value,free_edges[0])

    edge_weight = ((a_v0*a_v1))**(1/2)
    T["0"][free_edges[0]]['weight'] = edge_weight

    return T
def find_realization(alpha=0,beta=64,k=2):

    eigenvalues_list=produce_eigenvalues(k,alpha,beta)

    delta=(beta-alpha)/2

    T_beta = nx.Graph()
    T_beta.add_node(0,weight=beta)
    omega_beta = (((beta-alpha)*(beta-alpha)))**(1/2)
    T_beta = next_diamater(T_beta,T_beta,edge_weight=omega_beta)
    
    T_delta = copy_graph(T_beta,delta)
    omega_delta = ((beta+delta-alpha)*(beta-alpha))**(1/2)
    T_delta=recompute_edges(T_delta,edge_weight=omega_delta)

    T_alpha = nx.Graph()
    T_alpha.add_node(0,weight=alpha)
    omega_alpha = (((alpha-beta)*(alpha-beta)))**(1/2)
    T_alpha = next_diamater(T_alpha,T_alpha,edge_weight=omega_alpha)

    T_theta = copy_graph(T_alpha,delta)
    omega_theta = (((alpha+delta-beta)*(alpha-beta)))**(1/2)
    T_theta=recompute_edges(T_theta,edge_weight=omega_theta)

    for i in range(2,k+1):


        delta=delta/2

        a_v0=compute_v0(T_beta,eigenvalues_list[k+1-i],"0")
        a_v1=compute_v0(T_alpha,eigenvalues_list[k+1-i],"0")
        omega_alpha = ((a_v0*a_v1))**(1/2)   
        _T_alpha_ = next_diamater(T_beta,T_alpha,edge_weight=omega_alpha)
        _T_beta_ = next_diamater(T_theta,T_delta,edge_weight=omega_alpha)

        if i >2: 
            _T_alpha_aux_ = _T_alpha_.copy()
            _T_alpha_aux_.remove_node("0")
            _T_beta_aux_ = _T_beta_.copy()
            _T_beta_aux_.remove_node("0")
            eigenlist_alpha = sorted(list(set(list(np.unique(compute_eigenvalues(_T_alpha_)[1]))+list(np.unique(compute_eigenvalues(_T_alpha_aux_)[1])))))
            list_values[1].insert(0,max(eigenlist_alpha))
            eigenlist_beta = sorted(list(set(list(np.unique(compute_eigenvalues(_T_beta_)[1]))+list(np.unique(compute_eigenvalues(_T_beta_aux_)[1])))))
            list_values[0].insert(0,min(eigenlist_beta))


        if i==2:
            _T_alpha_aux_ = _T_alpha_.copy()
            _T_alpha_aux_.remove_node("0")
            _T_beta_aux_ = _T_beta_.copy()
            _T_beta_aux_.remove_node("0")
            if i==2:
                eigenlist1 = sorted(list(set(list(np.unique(compute_eigenvalues(_T_alpha_)[1]))+list(np.unique(compute_eigenvalues(_T_alpha_aux_)[1])))))
                eigenlist2 = sorted(list(set(list(np.unique(compute_eigenvalues(_T_beta_)[1]))+list(np.unique(compute_eigenvalues(_T_beta_aux_)[1])))))
                k_=int((len(eigenlist1)-1)/2)
                list_values = [[eigenlist2[2*i] for i in range(int(k_/2+1))],[eigenlist1[-2*i-1] for i in range(int(k_/2+1))]]


        list_values[0],list_values[1]=list_values[1],list_values[0]

        _T_theta_ = copy_graph(_T_alpha_,delta)
        values=list_values[0].copy()

        subgraph,sub_root_vertex = get_subgraphs(_T_theta_)

        for j,H in enumerate(subgraph):
            _T_theta_ = rearrange_edges(_T_theta_,values[-j-1],H,sub_root_vertex[j],root="0")


        _T_delta_ = copy_graph(_T_beta_,delta)

        values=list_values[1].copy()

        subgraph,sub_root_vertex = get_subgraphs(_T_delta_)
        for j,H in enumerate(subgraph):
            _T_delta_ = rearrange_edges(_T_delta_,values[-j-1],H,sub_root_vertex[j],root="0")


        T_alpha = _T_alpha_
        T_beta = _T_beta_
        T_theta = _T_theta_
        T_delta = _T_delta_
        
    return T_alpha,T_beta

def unfolding_weighted_tree(T,root_of_branch,number_of_copies=0):
    if number_of_copies>0:
        st.write()
        v = nx.shortest_path(T,root_of_branch,0)[1]
        s=number_of_copies
        edge_weight = T[v][root_of_branch]['weight']
        T_aux=T.copy()
        T_aux.remove_node(v)
        branch_nodes = list(nx.shortest_path(T_aux,root_of_branch).keys())
        branchies_copies = T_aux.subgraph(branch_nodes).copy()
        new_subroots = [root_of_branch*100000**l for l in range(s+1)]
        for l in range(s):
            branch_copy = T_aux.subgraph(branch_nodes).copy() 
            mapping_labels = {node:node*100000**(l+1) for node in branch_nodes}
            branch_copy = nx.relabel_nodes(branch_copy, mapping_labels)
            branchies_copies = nx.union(branchies_copies,branch_copy)
        T.remove_nodes_from(branch_nodes)
        T = nx.union(T,branchies_copies)
        for node in new_subroots:
            T.add_edge(v,node,weight=edge_weight/((s+1)**(1/2)))
        return T


#####################
tab1,tab2 = st.tabs(["Readme","Algorithm"])

with tab1:
    st.write('The goal of this application is to be a simulator of the work entitled "Generating acyclic symmetric matrices with the minimum number of distinct eigenvalues".')
    st.write('In this application, the user starts with a seed and can perform s-duplications at will, so that the generated tree $T_i$ will be displayed on the screen along with its spectrum.')
    st.write('Select the input parameters in the left menu of Algorithm and click on "Set Parameters". After that move to the tab "Algorithm".')
    st.write('Below the displayed tree, choose a vertex for its branch to be duplicated (the user can look up its number in the figure above) and the number of duplications for that branch. Click on "Perform s-duplication" in the desired branch.')
    st.write('Once some tree have been displayed, the user can choose the tree layout in the left menu and view the matrix of that tree.')
    st.write('You can access the full repository at https://github.com/Lucassib/ilas-diminimal-algorithm.')
    st.write('If you have any doubts, please contact us at lucas.siviero@ufrgs.br.')

with tab2:
    st.markdown("""
    <style>
    .big-font {
        font-size:250 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<p class="big-font">Inputs:</p>', unsafe_allow_html=True)

    alpha = st.sidebar.number_input("Alpha", min_value=None, max_value=None,value=0,step=1)
    beta = st.sidebar.number_input("Beta", min_value=None, max_value=None,value=64,step=1)
    diameter = st.sidebar.number_input("Diameter", min_value=1, max_value=19,step=1)

    run = st.sidebar.button("Set Parameters")
    if run:

        if 'sub_index' not in st.session_state:
            st.session_state['sub_index'] = 0
        else:
            st.session_state['sub_index'] = 0

        k = math.ceil((diameter)/2)
        M1,M2 = find_realization(alpha=alpha,beta=beta,k=k)
    
        M1 = nx.convert_node_labels_to_integers(M1,first_label=0)
        if diameter%2==0:
            M1 = unfolding_weighted_tree(M1,2**(k-1),number_of_copies=1)
            M1 = nx.convert_node_labels_to_integers(M1,first_label=0)

            st.session_state['paridade'] = 'par'
        else:
            st.session_state['paridade'] = 'impar'


        if 'graph' not in st.session_state:
            st.session_state['graph'] = M1
        else:
            st.session_state['graph'] = M1

    if 'graph' in st.session_state:

        M1 = st.session_state['graph']

        if st.session_state['sub_index'] == 0:
            st.title("Plotting Graph $S_{{{}}} = T_{}$ ".format(nx.diameter(M1),st.session_state['sub_index']))
        else:
            st.title("Plotting Graph $T_{}$ ".format(st.session_state['sub_index']) )
    
        st.sidebar.markdown('<p class="big-font">Display properties:</p>', unsafe_allow_html=True)

        graph_layout = st.sidebar.selectbox("Graph layout", ["Random","Hierarchy","Planar"])

        show_matrix = st.sidebar.selectbox("Matrix of Weights", ["Hide","Show"])

        if graph_layout == "Random":
            fig, ax = plt.subplots(figsize=(25,15))
            nx.draw(M1,with_labels=True, node_color='white', edgecolors='black',font_color="black",node_size=600)
            st.pyplot(fig)

        else:
            if graph_layout == "Hierarchy":
                pos = hierarchy_pos(M1,0)
            elif graph_layout == "Planar":
                pos = nx.planar_layout(M1)
            

            fig, ax = plt.subplots(figsize=(25,15))
            nx.draw(M1,pos=pos,with_labels=True, node_color='white', edgecolors='black',font_color="black",node_size=600)
            st.pyplot(fig)
        with st.form("unfolding",clear_on_submit=True):
            if st.session_state['paridade'] != 'par':
                no_possible_select_vertex = int(M1.number_of_nodes()/2 )
            else:
                pass

            central_edge = nx.center(M1)
            col1,col2 = st.columns(2)
            vertex = col1.number_input("Select the root of the branch you wish to s-duplicate.",min_value=1,max_value=M1.number_of_nodes()-1,step=1,value=1)
            number_of_copies = col2.number_input("Select s, the desired number of copies.",min_value=1,max_value=10,step=1,value=1)
            _,col = st.columns([0.28,0.72])
            submitted = col.form_submit_button("Perform s-duplication on the desired branch.")

            if submitted:

                if vertex in central_edge:
                    st.warning("The vertex cannot be chosen for duplication as it would increase the diameter.")
                else:
                    M1 = unfolding_weighted_tree(M1,vertex,number_of_copies=number_of_copies)
                    M1 = nx.convert_node_labels_to_integers(M1,first_label=0)
                    st.session_state['graph'] = M1  
                    st.session_state['sub_index']+=1

                    st.experimental_rerun()


        spec = compute_eigenvalues(M1)[1]
        dspec = np.unique(compute_eigenvalues(M1)[1])
        multiplicities = dict(Counter(spec))

        distinct_spectrum = ""
        for eigenvalue in np.unique(spec):
            if eigenvalue.is_integer():
                distinct_spectrum += "${}^{{[{}]}}$, ".format(int(eigenvalue),multiplicities[eigenvalue])
            else:
                distinct_spectrum += "${}^{{[{}]}}$, ".format(eigenvalue,multiplicities[eigenvalue])

        distinct_spectrum = distinct_spectrum[:-2]

        current_spec = "$Spec(T_{})=${{{}}}, $|DSpec(T_{})|={}$".format(st.session_state['sub_index'],distinct_spectrum,st.session_state['sub_index'],len(dspec))
        if 'current_spec' not in st.session_state:
            st.session_state['current_spec'] = [current_spec]
        else:
            if current_spec != st.session_state['current_spec'][-1]:
                st.session_state['current_spec'].append(current_spec)

        st.subheader(current_spec)

        for i in range(2,st.session_state['sub_index']+1+1):
            st.write(st.session_state['current_spec'][-i])
            time.sleep(0.5)




        if show_matrix=="Show":

            def prime_factors(n):
                i = 2
                factors = []
                while i * i <= n:
                    if n % i:
                        i += 1
                    else:
                        n //= i
                        factors.append(i)
                if n > 1:
                    factors.append(n)
                return factors
            def edge_label_as_latex(number):
                factors = dict(Counter(prime_factors(number)))

                outside_root=1
                inside_root=1
                for key in factors:
                    
                    key_repetitions = factors[key]
                    quocient = int(key_repetitions/2)
                    remainder = key_repetitions%2

                    outside_root *= key**quocient
                    inside_root *= key**remainder

                return "{}√{}".format(outside_root,inside_root) , outside_root, inside_root

            M1 = st.session_state['graph']

            matrix = compute_eigenvalues(M1)[0]

            dataframe_matrix = pd.DataFrame(matrix).astype(int)

            edges = list(M1.edges)
            edge_label = [ matrix[edge] for edge in edges ]
            possible_weight_dict = {i:(i)**(1/2) for i in range(math.ceil(max(edge_label))**2)}

            try:

                edge_weight_latex = []
                for edge_weight in edge_label:
                    edge_label_key = [i for i in possible_weight_dict if np.round(possible_weight_dict[i],10)==np.round(edge_weight,10)][0]

                    latex_string, outside_root, inside_root = edge_label_as_latex(edge_label_key)

                    validator = (outside_root**2*inside_root)**(1/2)

                    edge_weight_latex.append(latex_string)
                
                for i,edge in enumerate(edges):
                    dataframe_matrix.loc[edge] = edge_weight_latex[i]
                    dataframe_matrix.loc[(edge[1],edge[0])] = edge_weight_latex[i]

                st.write(dataframe_matrix.astype(str))

            except:
                st.write(matrix)

            



