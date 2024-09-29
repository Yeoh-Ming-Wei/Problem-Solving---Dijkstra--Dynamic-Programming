"""
        ---------------------
    --- FIT 2004 ASSIGNMENT 1 --- 
        ---------------------

These are the code for assignments 1. It includes all the function availables for question 1
and question 2. It will be labeled as a comment to specify which part belongs to the specific
questions. 

"""
__author__ = "Yeoh Ming Wei"

import math
"""
    __ Q U E S T I O N     1
"""
class Graph: 
    """ A Graph class which contains number of vertices. """
    
    def __init__(self): 
        """
        A constructor to initialize a list of vertices and the amount(length) of vertices available. 
        
        Attributes
            - vertices: A list contains Vertex object
            - length: the length of the vertices list (The amount of vertex available)
        """
        self.vertices = []
        self.length = 0
        
    def __str__(self): 
            """ A toString method when Graph object is called. """
            string = ""
            for v in self.vertices: string += "{}\n".format(str(v))
            return string
    
    def getVertex(self, id): 
        """ 
        A function that that returns the specific Vertex.
        
        Input
            - id: The vertex number

        Output: The Vertex object based on num input. 

        Time complexity: O(1), where it returns a specific vertex.
        """
        return self.vertices[id]
    
    def addVertex(self, id, edge): 
        """ 
        A function to add Vertex object inside the vertices list along with the edges.
        
        Input
            - id: The vertex number
            - edge: The edge object of the vertex

        Time complexity: O(V), where V is the number of vertices
            - O(V), when the worst case is where the id given is equal to the number of vertices
        Space complexity: O(V + E), where V is the number of vertices and E is the number of edge
            - O(V + E), when we append x amount of v and add one edge. 
        """
        if self.length == 0: 
            # If the length is 0, multiply the vertices list based on the number"
            self.vertices = [None] * (id + 1)
            self.length += (id + 1) 

        if (id + 1) > self.length: 
            # If the number given is more than the length of the list, 
            # increase the list size and append None along.
            for i in range((id + 1) - self.length):
                self.vertices.append(None) 
                self.length += 1
        
        if self.vertices[id] == None: 
            # Create the Vertex object based on its id and add an edge
            self.vertices[id] = Vertex(id, edge)
        else: 
            # Add the edges if the vertex exist
            self.vertices[id].addEdge(edge)
        
    
    def addRoad(self, tuple): 
        """
        Assignment related function, the road is a tuple that represents: 
        - Initial point (Vertex u)
        - End point (Vertex v)
        - Normal lane (The weight of the edge, w1)
        - Carpool lane (The weight of the edge, w2)

        We need to assign the edge to both u and v vertices.

        Input
            - tuple: A tuple represents the u, v, w1, w2

        Time complexity: O(V), where V is the number of vertices
            - O(V) as we are calling addVertex function
        """
        u, v, w1, w2 = tuple[0], tuple[1], tuple[2], tuple[3]
        self.addVertex(u, Edge(u, v, w1, w2))
        self.addVertex(v, Edge(u, v, w1, w2))
    
    def addPassenger(self, passengers): 
        """
        Assignment related function, if there is a passenger at specific
        road (vertex), assign the passenger attribute in vertex to true

        Input:
            passengers: List of integer that represents passenger available at the vertex

        TIme complexity: O(|P|), where P is the length of passenger
            - O(|P|) as we use for loop to lopo through the passenger list
            - O(1) when we perform assignment. 
        """

        for p in passengers: self.vertices[p].passenger = True

    def dijkstra(self, start : int, end : int): 
        """
        Assignment related function
        This is a modified dijkstra function which accomodate the assignment requirement. 

        - Instead of having one weight for each of the edges. We have two different or same
        weight. So, the heap created will have two same vertex but different road. 
        - The while loop will end when the discovered minHeap list becomes empty.
        - Inside the while loop, we serve the minimum value in the list. 
        - We loop through the edges of the vertex that we served using for loop. 
        - It will check the vertex passenger and road type.

        For the starting road u, 
        - if it has no passenger and uses normal road (1), 
            - total distance: init1 + weight1
            - update v: init1 and prevous1
            - There is no passenger so we use normal road and v will be updated with its normal road distance and road u. 

        - if it has passenger and uses normal road (1)
            - total distance: init1 + weight2  (2 = carpool)
            - update v: init2 and prevous2
            - There is passenger so we use carpool road and v will be updated with its carpool distance and road u.
            
        - If it has no passenger but uses carpool road (2)
            - total distance: init2 + weight2 
            - update v: init2 and prevous2
            - Which means that the person's car already have a passenger picked up from one of the vertex previously.
            - So, v will be updated with its carpool distance and road u.
        
        - If it has passenger and uses carpool road (2), 
            - total distance: init2 + weight2 
            - update v: init2 and prevous2
            - Which means that the person's car already have a passenger picked up from one of the vertex previously.
            - So, v will be updated with its carpool distance and road u.
        
        - The total distance will be compare with the v distance and perform update if the total distance is actually shorter.
        - After the process is finish, each of the vertex has its own minimum distance for normal road and carpool raod. 
        - To find the shortest distance, we need to compare both normal and carpool road.
        - Lets say if carpool road is the shortest, we will use the carpool road previous road to perform backtracking to find all
          passed road.
        - The output will be a list of road containing which road should I drive from start to end so that I have the shortest
          distance. 

        Input: 
            - start: The starting point of the road / vertex
            - end: The ending point of the road / vertex
        
        Output: A list of optimal route from start to end point

        Time complexity: O(|R| log |L|), where |R| is the number of road, |L| is the number of location
            - Similar to O(E log V) where E is the number of edge and V is the number of vertex. 
            - O(V log V) when inserting the vertices into min heap.
            - O(E log V) when performing the actions: 
                - O(V) as the "while" loop through vertices.
                - O(V^2) becomes O(E) as the "for loop" in "while loop" and it loops through edges. 
                  O(E) will represents the tightest bound of the complexity.
                - O(log V) when the update function in minHeap is called.
                - Final complexity is O(E) * O(log V) = O(E log V)
            - Backtracking list only uses O(V) as worst case complexity, same thing as the reverse list function. 
        
        Space complexity: O(|L|), where |L| is the number of location
            - The space is used at the discovered variable where we append all the locations inside the list. 
            
        Written at Week 4 Lecture Notes, modified by me. 
        """
        discovered = minHeap()      # A Minheap to manage the vertices
        
        # Assign starting vertice, the "1" representing normal road: 
        # distance1 = 0 
        # previous1 = (-1, -1) - To show that it is the initial
        self.getVertex(start).d1 = 0                     
        self.getVertex(start).previous1 = (-1, -1)

        # Insert all vertices, 2 different road (normal, 1 and carpool road, 2)
        # discovered length = total v * 2
        # Time complexity: O(V log V)
        for v in self.vertices:
            discovered.insert(v, 1, v.d1)
            discovered.insert(v, 2, v.d2)

        # Loop until the length of discovered is 0
        # Time complexity: O(V)
        while discovered.length != 0:

            # Peform serve to remove the shortest distance vertex from heap
            # Assigned as tuple u
            # Time complexity: O(V log V)
            uTuple = discovered.serve()
            u, uRoad = uTuple[0], uTuple[1]

            # Loop every edges in u
            # O(E) (if tightest bound), O(V^2) (if worst case)
            for edge in u.edges: 
                v = self.getVertex(edge.v)

                # If statement that compare and update the distance of v
                if (not u.passenger and uRoad == 1):
                    distance = u.d1 + edge.w1
                
                    if distance < v.d1: 
                        v.d1 = distance
                        v.previous1 = (u.id ,uRoad)
                        discovered.update(v.hIndex1)         # Update the heap every time a process is done, O(E log V)

                else: 
                    if not (u.passenger and uRoad == 1): 
                        distance = u.d2 + edge.w2
                    else: 
                        distance = u.d1 + edge.w2

                    if distance < v.d2: 
                        v.d2 = distance
                        v.previous2 = (u.id, uRoad)
                        discovered.update(v.hIndex2)         # Update the heap every time a process is done, O(E log V)

        # Compare normal road distance and carpool distance
        if (self.vertices[end].d1 < self.vertices[end].d2): 
            last = (self.vertices[end].previous1[0], self.vertices[end].previous1[1])
        else: 
            last = (self.vertices[end].previous2[0], self.vertices[end].previous2[1])

        # Perform backtracking
        # Time complexity: O(V)
        backtrackList = []
        while last[0] != -1: 
            backtrackList.append(last[0])
            if last[1] == 1: 
                last = (self.getVertex(last[0]).previous1[0], self.getVertex(last[0]).previous1[1])
            else: 
                last = (self.getVertex(last[0]).previous2[0], self.getVertex(last[0]).previous2[1])

        backtrackList.reverse()     # Time complexity, O(V)
        backtrackList.append(end)
        
        return backtrackList
            

class Vertex: 
    """A vertex class which contains a list of edges. """

    def __init__(self, id, edge):
        """
        A constructor to initialize vertex id, a list of edges, check passenger availability at a specific vertex. 
        It also contains every vertex information such as distance, heap index and previous vertex.
        It is to represents the normal and carpool roads.
        
        Attributes
            - id: A number to identify the vertex
            - passenger: true if there is passenger for specific vertex, otherwise false
            - edge: a list of edges in the specific vertex

            For 2 of the roads: 
            - d(1 or 2): The distance of the vertex
            - hIndex(1 or 2): The heap index of the vertex
            - previous(1 or 2): The previous vertex
        """
        self.id = id
        self.passenger = False
        self.edges = [edge]

        self.d1 = math.inf
        self.hIndex1 = -1
        self.previous1 = (-1, -1)

        self.d2 = math.inf
        self.hIndex2 = -1
        self.previous2 = (-1, -1)

    def __str__(self): 
        """ A toString method when Vertex object is called. """
        return "Vertex: {}".format(self.id)

        
    
    def addEdge(self, edge): 
        """
        A function to add edge into specific vertex

        Input: 
            - edge: The edge of the vertex
        Time complexity: O(1) as append doesn't go through the whole list. 
        """
        self.edges.append(edge)

        
class Edge: 
    """
     An edge class which contains provide the outgoing and ingoing vertex, 
     along with the weight of the edge. 
    """

    def __init__(self, u, v, w1, w2): 
        """
        A constructor to initialize the edge attributes.
        
        Attributes
            - u: the starting vertex index (int)
            - v: the ending vertex index (int)
            - w1: The normal road distance (int)
            - w2: The carpool road distance (int)
        """
        self.u = u 
        self.v = v
        self.w1 = w1
        self.w2 = w2

    def __str__(self): 
        """ A toString method when Edge object is called. """
        return "({}, {}, {}, {})".format(self.u, self.v, self.w1, self.w2)
    

class minHeap:
    """
     A Heap class that manage a list of vertex object, it can be sort by its minimum
     distance using the given function implemented inside the Heap class.
     Written by Daniel Anderson at FIT 2004 Course Note, Modified by me.
    """

    def __init__(self): 
        """
        A constructor to initialize the list and its length.
        
        Attributes
            - lst: The list for the heap
            - length: The length of the heap list 
        """
        self.lst = [None]
        self.length = 0

    def insert(self, vertex : Vertex, road : int,  distance : int): 
        """
        A function to insert a tuple containing given vertex, its road, distance and index into the heap list

        Input: 
            - vertex: A vertex class
            - road: The road of the vertex (Normal - 1, carpool - 2)
            - distance: The shortest distance of the vertex (It may be initialized or unknown)

        Time complexity: O(log V)
            - V is the vertex
            - O(log V) as we are using the rise function. 
        """

        # Append a tuple containing vertex, road, distance and its index
        self.lst.append((vertex, road, distance, self.length + 1))

        # Upadte the vertex's heap index based on the road
        if road == 1: vertex.hIndex1 = self.length + 1
        else: vertex.hIndex2 = self.length + 1

        # Update the length of the heap list
        self.length += 1

        # Perform rise function to rise the specific vertex to its correct position
        self.rise(self.length)

    def serve(self): 
        '''
        A function to swap the first indexent (shortest distance) with last indexent and pop
        the last indexent from the heap list. 

        Output: A tuple that is popped out from the heap list

        Time complexity: O(log V), where V is the number of elements in minHeap 
            - The time complexity is O(log V) as we perform sink function. 
        '''

        # Peform swapping with first and last indexent. 
        self.swap(1, self.length)

        # Reduce the length of the heap list
        self.length -= 1

        # Perform sink function to sink the specific vertex to its correct position
        self.sink(1)
        
        # Pop out the last index for output
        return self.lst.pop()
    
    def swap(self, a, b):
        """
        A swap function to swap the 2 specific indexent in the heap list. 

        Input: 
            - a: The indexent that you want to swap
            - b: The another indexent that you want to swap

        Time complexity: O(1) as we are only assigning new values. 
            - The updateIndex function has O(1) time complexity too. 
        """
        # Assignment for tuple and index so that it looks tidier.
        tupleA, tupleB = self.lst[a], self.lst[b]
        indexA, indexB = self.lst[a][3], self.lst[b][3]

        # Perform swapping with tuple a and b. 
        self.lst[a] = (tupleB[0], tupleB[1], tupleB[2], indexA)
        self.lst[b] = (tupleA[0], tupleA[1], tupleA[2], indexB)

        # Update the index both of the vertex 
        self.updateIndex(self.lst[a], self.lst[a][1])
        self.updateIndex(self.lst[b], self.lst[b][1])
        
    def updateIndex(self, tuple, road): 
        """
        A function to update the vertex heap index. 

        Input: 
            tuple: The tuple in the heap list
            road: The road of the vertex
        
        Time complexity: O(1) as we are only assigning new values. 
        """
        if road == 1: 
            # Update hIndex1 if the road is normal road (1)
            tuple[0].hIndex1 = tuple[3]
        else: 
            # Update hIndex2 if the road is carpool road (2)
            tuple[0].hIndex2 = tuple[3]

    def rise(self, index): 
        '''
        A function to perform rise to a tuple until it reaches the correct position. 
        This can be done by comparing the parent indexent and keep swaping until the 
        child is smaller than the parent or parent == 1.

        Input: 
            - index: The index of the heap list

        Time complexity: O(log V) where V is the number of element in minheap.
            - The function does not go through all the element in minHeap. 
            - Instead, it goes though index // 2 elements depends on how many loop
            - So, the final time complexity is O(log V). 
            - Swap function has time complexity of O(1). 
        '''

        # To determine which parent that we need to check
        parent = index // 2 
       
        if parent > 0 and self.lst[parent][2] > self.lst[index][2]: 
            # If the parent is more than 0, or the parent is bigger than the child
            # Perform swapping with parent and child. 
            self.swap(parent, index)

            # Perform recursion to correct the position of the parent. 
            self.rise(parent)


    def sink(self, index): 
        '''
        A function to perform sink to a tuple until it reaches the correct position. 
        This can be done by comparing the child indexent and keep swaping until the 
        parent is smaller than the child.
        
        Input: 
            - index: The index of the heap list

        Time complexity: O(log V) where V is the number of element in minheap.
            - The function does not go through all the element in minHeap. 
            - Instead, it goes though index * 2 elements depends on how many loop
            - So, the final time complexity is O(log V).
        '''
        # To determine which child that we need to check
        child = index * 2

        # The child must be smaller or equal to the length of list (or not it will be out of range)
        while child <= self.length:

            # If the child is smaller than length of list 
            # If the child at at the right is smaller than the left child
            if child < self.length and self.lst[child + 1][2] < self.lst[child][2]:
                # Use the left child 
                child += 1

            # Compare the parent if the number is bigger than the child
            if self.lst[index][2] > self.lst[child][2]: 
                # Perform swapping with the parent and child
                self.swap(child, index)

                # Assign the current child as its index for recheck
                index = child
                child = 2 * index
            else: 
                break
    
    def update(self, index):   
        """
        A function to update the tuple value and placed to its correct position. 

        Input: 
            - index: The index of the heap list

        Time complexity: O(log V) where V is the number of element in minheap.
            - The complexity is O(log V) as we are using rise function. 
            - The assignments only reach time complexity of O(1). 
        """
        # Update the tuple in the heap list based on its road after changes is done in dijkstra function
        if self.lst[index][1] == 1: 
            self.lst[index] = (self.lst[index][0], self.lst[index][1], self.lst[index][0].d1, self.lst[index][3])
        else: 
            self.lst[index] = (self.lst[index][0], self.lst[index][1], self.lst[index][0].d2, self.lst[index][3])
        
        # Perform rise function to rise the specific tuple to its correct position
        self.rise(index)


    def __str__(self): 
        """ A to string method when Heap class is called. """
        string = ""
        for v in self.lst: 
            if v != None: 
                string += "{}, \troad: {}, \td: {}, index: {}\n".format(str(v[0]), str(v[1]), str(v[2]), str(v[3]))
        return string

def optimalRoute(start, end, passengers, roads):
    """
    Assignment related function.
    This is a functon that contains a combination of different functions to find the optimal route
    for the question given. 

    1) Add the roads given into a graph class.
    2) Update the passengers attribute in vertex based on the vertex class
    3) Perform dijkstra by inputing the start point and end point, the result returns an
       optimal route 
    4) Return a list showing the optimal route.

    Input: 
        - start: The starting point of the road
        - end: The end point of the road
        - passengers: A list of passengers, the element inside represents which location has passenger
        - roads: A list of tuples containing the starting road, ending road, the distance between the road
                 the last two element represents normal road and carpool road

    Output: The list containing the location number. Shows that it is an optimal route from the start to end. 
    Time compexity: O(|R| log |L|) or O(E log V)
        - O(V^2) where every road, we perform addRoad function. 
        - O(|P|), where we call the addPassenger function.
        - O(|R| log |L|), where we call dijkstra function. 
        - The big-O notation will be O(|R| log |L|) or O(E log V).
    """
    
    # Create a new graph object
    g = Graph()     

    # Perform add road function provided in the Graph class for every road. 
    for r in roads: g.addRoad(r)

    # Update the vertex passenger based on the given passenger list
    g.addPassenger(passengers)

    # Perform dijkstra function by inputting the starting point and ending point
    # Return a list which is the result for optimal route
    res = g.dijkstra(start, end)

    return res

"""
    __ Q U E S T I O N     2
"""

def select_sections(lst): 
    """
    An assignment related function. 
    A function that takes a list as an input, and output the optimal sections to be
    removed for each row.

    The function starts by appending the first row into the memory list. 
    For other existing row, it will compare the value of the previous row which is the
    top left, up, and top right of the current position. 
    As for the last row, it will has an addition condition to find the minimum probability.

    When the process is complete, it will take the row and column with the minimum probability 
    and perform backtracking to back track the previous row and column until it reaches the first
    row.
    
    The output wil be the minimum probability with a list of one column that needs to remove for
    each row. 

    Input: 
        - lst: The list with size of n * m (n = row, m = column)
    
    Output: A list with the least probability as its first element and a list of 
            m that need to remove from each n as second element. 

    Time complexity: O(nm)
        - O(n) for the first for loop which loops through all n rows
        - O(m) for the second for loop which loops through all m columns
        - O(n) * O(m) = O(nm) as there is nested for loop.
    
    Aux space complexity: O(nm)
        - The memory list size is n * m, so O(nm)
    """

    # A memory list to update the value when going through the lst. 
    memory = []

    # Used for the last row of memory to track the minimum probability. 
    minProb = None
    
    # Looping thorugh all n rows
    # Time complexity: O(n)
    for n in range(len(lst)): 

        # Append an empty list before going through the column
        memory.append([])

        # Looping through all the m columns
        # Time complexity: O(nm)
        for m in range(len(lst[n])):  
            
            # First row is the base case, append a tuple consisting of the row number and an empty list for backtracking later
            if n == 0: 
                memory[n].append((lst[n][m], []))

            # Update the rest of the row
            else: 

                # The number above the row with same column will be append inside first.  
                lstNum = lst[n][m]
                memory[n].append((memory[n - 1][m][0] + lstNum, m))

                # Check the adjacent above the row if it is smaller than than the current value
                # If yes then, it will change the current value to a smaller value. 
                left = m - 1
                if left >= 0 and memory[n - 1][left][0] + lstNum < memory[n][m][0]:
                    memory[n][m] = (memory[n - 1][left][0] + lstNum, left)

                right = m + 1
                if right < len(lst[n]) and memory[n - 1][right][0] + lstNum < memory[n][m][0]:
                    memory[n][m] = (memory[n - 1][right][0] + lstNum, right)

            # The last row will check which row has the minimum probability. 
            if n == (len(lst) - 1):
                minProb = m if minProb == None or memory[n][m][0] < memory[n][minProb][0] else minProb

    res = []
    
    # Backtrack from the last one to obtain the other rows that needs to be removed. 
    for i in range(len(memory) - 1, -1, -1): 
        res.append((i, minProb))
        minProb = memory[i][minProb][1]

    res.reverse()

    return [memory[res[len(memory) - 1][0]][res[len(memory) - 1][1]][0], res]


if __name__ == "__main__": 
    # Example
    start = 0
    end = 4
    # The locations where there are potential passengers
    passengers = [2, 1]
    # The roads represented as a list of tuple
    roads = [(0, 3, 5, 3), (3, 4, 35, 15), (3, 2, 2, 2), (4, 0, 15, 10),
    (2, 4, 30, 25), (2, 0, 2, 2), (0, 1, 10, 10), (1, 4, 30, 20)]
    # Your function should return the optimal route (which takes 27 minutes).
    optimalRoute(start, end, passengers, roads)
    
    occupancy_probability = [
    [31, 54, 94, 34, 12],
    [26, 25, 24, 16, 87],
    [39, 74, 50, 13, 82],
    [42, 20, 81, 21, 52],
    [30, 43, 19, 5, 47],
    [37, 59, 70, 28, 15],
    [ 2, 16, 14, 57, 49],
    [22, 38, 9, 19, 99]]
    select_sections(occupancy_probability)
