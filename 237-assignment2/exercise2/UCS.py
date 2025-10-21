'''
@author: Devangini Patel

@modified: Hannah Raju
@date: October 6, 2025
'''
from State import State
from Node import Node
import queue
from TreePlot import TreePlot
    

def performUCS():
    """
    This method performs uniform cost search
    """
    
    #create queue
    pqueue = queue.PriorityQueue()

    #create visited list
    visited = []
    
    #create root node
    initialState = State()
    root = Node(initialState, None)
    
    #show the search tree explored so far
    treeplot = TreePlot()
    treeplot.generateDiagram(root, root)
    
    #add to priority queue
    pqueue.put((root.heuristic, root))

    #check if there is something in priority queue to dequeue
    while not pqueue.empty(): 
        
        #dequeue nodes from the priority Queue
        _, currentNode = pqueue.get()
                
        #remove from the fringe
        currentNode.fringe = False
    
        # prints current node
        print ("-- dequeue --", currentNode.state.place)

        
        #check if this is goal state
        if currentNode.state.checkGoalState():
            print ("reached goal state")
            #print the path
            print ("----------------------")
            print ("Path")
            currentNode.printPath()
            break
            
        #get the child nodes if current node hasn't been marked 
        if(currentNode.state.place not in visited):

            visited.append(currentNode.state.place)
            childStates = currentNode.state.successorFunction()

            for childState in childStates:
                childNode = Node(State(childState), currentNode)
                
                if(childNode.state.place not in visited):
                               
                    # add child to pq if it hasnt been visited
                    pqueue.put((childNode.heuristic, childNode))

            #show the search tree explored so far
       

              
    #print tree
    treeplot = TreePlot()
    treeplot.generateDiagram(root, currentNode)
    print ("----------------------")
    print ("Tree")
    root.printTree()
    
performUCS()