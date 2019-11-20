import cv2
import numpy as np
import maxflow

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

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    ### Use Alpha expansion binary image for each label

    ### 1) Define Graph

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow


    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

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



