import cv2
import numpy as np
import maxflow
import matplotlib.pyplot as plt

def potts_cost(wm, wn):
    '''
    Pairwise cost by Potts Model
    :param wm: Label1
    :param wn: Label2
    :return: Pairwise cost
    '''
    delta = 0
    if wm != wn:
        delta = 1
    return delta

def question_3(I,rho=0.7,pairwise_cost_same=0.005,pairwise_cost_diff=0.2):
    ### 1) Define Graph
    g = maxflow.Graph[float]()
    binary_img = I.copy()/255
    ### 2) Add pixels as nodes
    nodes = g.add_nodes(I.shape[0]*I.shape[1])

    ### 3) Compute Unary cost
    Un =  -np.log(np.array([rho,1-rho]))
    P_same = np.array([0,pairwise_cost_same])

    ### 4) Add terminal edges
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            edges = Un.copy() if binary_img[i,j] else np.roll(Un.copy(),1)
            pairwise_same = P_same.copy() if (i+j)%2 == 0 else  np.roll(P_same.copy(),1)
            total = edges+pairwise_same
            g.add_tedge(nodes[i*I.shape[1] + j],total[0],total[1])

    ### 5) Add Node edges
    ### Vertical Edges
    P_non_terminal = np.array([pairwise_cost_diff - 2 * pairwise_cost_same, pairwise_cost_diff])
    for i in range(I.shape[0]-1):
        for j in range(I.shape[1]):
            g.add_edge(nodes[i*I.shape[1] + j], (i+1)*I.shape[1] + j,P_non_terminal[0],P_non_terminal[1])
    ### Horizontal edges
    for i in range(I.shape[0]):
        for j in range(I.shape[1]-1):
            g.add_edge(nodes[i*I.shape[1] + j], i*I.shape[1] + j+1,P_non_terminal[0],P_non_terminal[1])
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow
    g.maxflow()
    Denoised_I = np.empty_like(I)
    for i in range(I.shape[0]-1):
        for j in range(I.shape[1]):
            Denoised_I[i,j] = g.get_segment(nodes[i*I.shape[1] + j])
    
    Denoised_I = Denoised_I*255

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I, rho=0.6):
    '''
    displays denoised image after Alpha expansion for each label
    :param I: Original Image
    :param rho: Bernoulli Parameter
    :return:
    '''
    infinite_cost = 1e15
    labels = np.unique(I).tolist()
    Denoised_I = np.zeros_like(I)

    label_indices = np.array([labels, np.roll(labels, -1), np.roll(labels, 1)])
    I_org = I.copy()
    ### Use Alpha expansion binary image for each label
    for k in range(10):
        for i in range(len(labels)):
            ### 1) Define Graph
            g = maxflow.Graph[float]()

            ### 2) Add pixels as nodes
            nodes = g.add_grid_nodes(I.shape)

            ### 3) Compute Unary cost
            unaryCost = np.ones_like(I)*(1 - rho)/2
            unaryCost[I == labels[i]] = rho

            ### define the labels
            alpha = label_indices[i, 0]
            beta = label_indices[i, 1]
            gamma = label_indices[i, 2]

            ### 4) Add terminal edges
            g.add_grid_tedges(nodes, unaryCost, 1 - unaryCost)

            ### 5) Add Node edges
            ### Vertical Edges
            for x in range(I.shape[0]-1):
                for y in range(I.shape[1]):
                    if I[x, y] == alpha and I[x+1, y] == alpha:
                        unaryCost[x, y] = rho
                    elif I[x, y] == alpha and I[x+1, y] == beta:
                        pairwiseCost = potts_cost(alpha, beta)
                        g.add_edge(nodes[x, y], nodes[x+1, y], pairwiseCost, pairwiseCost)
                    elif I[x, y] == beta and I[x+1, y] == beta:
                        pairwiseCost = potts_cost(alpha, beta)
                        g.add_edge(nodes[x, y], nodes[x+1, y], pairwiseCost, pairwiseCost)
                    elif I[x, y] == beta and I[x+1, y] == gamma:
                        new_node = g.add_nodes(1)[0]
                        pairwiseCost = potts_cost(gamma, beta)
                        g.add_edge(nodes[x, y], new_node, pairwiseCost, infinite_cost)
                        g.add_edge(new_node, nodes[x+1, y], pairwiseCost, infinite_cost)
                        g.add_tedge(new_node, 0, infinite_cost)

            ### Horizontal edges
            for x in range(I.shape[0]):
                for y in range(I.shape[1] - 1):
                    if I[x, y] == alpha and I[x, y+1] == alpha:
                        unaryCost[x, y] = rho
                    elif I[x, y] == alpha and I[x, y+1] == beta:
                        pairwiseCost = potts_cost(alpha, beta)
                        g.add_edge(nodes[x, y], nodes[x, y+1], pairwiseCost, pairwiseCost)

                    elif I[x, y] == beta and I[x, y+1] == beta:
                        pairwiseCost = potts_cost(alpha, beta)
                        g.add_edge(nodes[x, y], nodes[x, y+1], pairwiseCost, pairwiseCost)
                    elif I[x, y] == beta and I[x, y+1] == gamma:
                        new_node = g.add_nodes(1)[0]
                        pairwiseCost = potts_cost(gamma, beta)
                        g.add_edge(nodes[x, y], new_node, pairwiseCost, infinite_cost)
                        g.add_edge(new_node, nodes[x, y+1], pairwiseCost, infinite_cost)
                        g.add_tedge(new_node, 0, infinite_cost)


            ### 6) Maxflow
            g.maxflow()
            segmentedGraph = g.get_grid_segments(nodes)
            Denoised_I[segmentedGraph==False] = alpha
        I = Denoised_I
    cv2.imshow('Original Img', I_org), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    f, axarr = plt.subplots(1, 2, sharey=True)
    axarr[0].imshow(I_org, cmap='gray')
    axarr[0].set_title('Original Image')
    axarr[1].imshow(Denoised_I, cmap='gray')
    axarr[1].set_title('Denoised Image')
    plt.show()
    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4
    question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()


