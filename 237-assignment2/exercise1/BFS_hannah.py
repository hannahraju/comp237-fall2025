'''
@author: Hannah Raju

This function determines the shortest path between students according to their relationships

@parameters: graph_name, start student, target student 
@returns: shortest path between students as list

'''

from ClassData import classmates
from Node import Node
from State import State
from collections import deque

def BFS_hannah(graph_name, initial_student, target_student):

    if((graph_name or initial_student or target_student) is False):
        print("Argument doesn't exist")
        return 

    # create queue to track frontier
    queue = deque([])

    # create list to track visited nodes
    visited = []

    #create root node
    initialState = State(initial_student)
    root = Node(initialState)

    # add node to queue and visited list
    queue.append(root)
    visited.append(root.state.name)

    #check if there's something to dequeue:
    while len(queue) > 0:

        #get first item in queue
        currentNode = queue.popleft()
        print(("--dequeue--"), currentNode.state.name)

        #check if it's goal state
        if currentNode.state.checkGoalState(target_student):
            print("reached goal state")
            #prints the path
            print("------")
            print("Path")
            currentNode.printPath()
            path = currentNode.getPath()
            return path

        #get child nodes
        childStates = currentNode.state.successorFunction(graph_name)
        for childState in childStates:
            childNode = Node(State(childState))

            #make sure node not visited
            if childNode.state.name not in visited:
                 
                #add to visited
                visited.append(childNode.state.name)

                # add to tree and queue
                currentNode.addChild(childNode)
                queue.append(childNode)

        root.printTree()

    # if we make it here there is no path
    print("Sorry! No path exists between the two nodes")

    

path1 = BFS_hannah(classmates, "George", "Bob" )
#path2 = BFS_hannah(classmates, "Dolly", "Hannah")