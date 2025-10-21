'''
@author: Devangini Patel
@modified: Hannah Raju | October 6, 2025
'''

class State:
    '''
    This class retrieves state information for social connection feature
    '''
    
    def __init__(self, name = None):
        if name == None:
            #create initial state
            self.name = self.getInitialState()
        else:
            self.name = name
    
    def getInitialState(self):
        """
        This method returns initial state
        """
        initialState = "Hannah"
        return initialState


    def successorFunction(self, graph_name):
        """
        This is the successor function. It finds all the persons connected to the
        current person
        """
        return graph_name[self.name]
        
        
    def checkGoalState(self, goal):
        """
        This method checks whether the name is goal.
        """ 
        #check if the person's name is Jill
        return self.name == goal