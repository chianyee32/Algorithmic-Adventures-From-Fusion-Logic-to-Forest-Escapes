

#                               2004 Assignment 1
#                              Name: Chian Yee On
#                             Student ID: 33402302



### Question 1 ###

def fuse(fitmons):

    """
    Function description: 
        This function accepts a list of FITMONs, each with a left/right affinity and cuteness score, 
        and produces a maximum cuteness score after combining them all. To determine the maximum cuteness score, 
        this function use dynamic programming.

    Approach description: 
        This function is using a dynamic programming approach to consider all possible pairs of FITMONs,
        then fuse them at each step. For each pair of FITMONs, we calculate the resultant cuteness score,
        and keep track of the maximum score gotten. This function can enseres the final score is the maximum cuteness
        score after all fusions since it is building up from the simplest case (a single FITMON) to the entire list.

    Input: 
        fitmons: a list of lists, where each sublist represents a FITMON in the form of
        [affinity_left, cuteness_score, affinity_right]
    
    Output: 
        An integer representing the maximum cuteness score obtainable from fusing all the FITMONs.

    Time complexity: O(n^3), where n is the number of FITMONs in the input list.

    Time complexity analysis: 
        The function iterates over all subproblems of fusing FITMONs (which are O(n^2) in number), 
        and for each subproblem, it tries to fuse them at all possible positions (which is O(n)), 
        resulting in O(n^3) complexity overall.
    
    Space complexity: 
        O(n^2), where n is the number of FITMONs in the input list.

    Space complexity analysis: 
        The function uses a 2D list (dynamic programming table) to store the maximum cuteness scores 
        for each pair of FITMONs, which requires O(n^2) auxiliary space.
    """

    x = len(fitmons) # Stores the length of input list into a new variable for the code below

    # Initialize the dp table with zeros
    dp = []
    for _ in range(x):
        dp.append([0]*x)
    
    # Initialize the base case: individual FITMON's cuteness_score
    for i in range(x):
        dp[i][i] = fitmons[i][1]

    # Compute the maximum cuteness score for fusing FITMONs from i to j
    for num in range(2, x + 1):

        for i in range(x - num + 1):
            j = i + num - 1
            dp[i][j] = 0  # Reset maximum for this subproblem

            for k in range(i, j):

                # Compute the cuteness score for fusing at k
                if k + 1 <= j:
                    left_fitmon_cuteness = dp[i][k] * fitmons[k][2]
                else:
                    left_fitmon_cuteness = 0

                if k >= i:
                    right_fitmon_cuteness = dp[k + 1][j] * fitmons[k + 1][0]  
                
                else: 
                    right_fitmon_cuteness = 0

                # Calculate total cuteness score after the fusion
                total = int(left_fitmon_cuteness + right_fitmon_cuteness)

                # Choose the fusion that gives the maximum cuteness score
                if total > dp[i][j]:
                    dp[i][j] = total
    
    # Return the maximum cuteness score of the final fused FITMON
    return dp[0][x - 1]

### Question 2 ###

class MinHeap:

    """
    A class to represent a MinHeap data structure which is a specialized tree-based data structure
    that satisfies the heap property: in a min heap, for any given node C, if P is a parent node of C,
    then the key (the value) of P is less than or equal to the key of C. This implementation uses an array
    to efficiently represent the heap.

    Attributes:
        - heap (list): A dynamic list used to store the elements of the heap in a way that allows easy
                     manipulation to maintain the heap properties.

    Methods:
        - __init__(self): Initializes a new empty MinHeap.
        - push(self, item): Adds an item to the heap while maintaining the heap property.
        - pop(self): Removes and returns the smallest item from the heap, maintaining the heap property.
        - sift_up(self, index): Moves the item at the specified index up in the heap to maintain the heap property.
        - sift_down(self, index): Moves the item at the specified index down in the heap to maintain the heap property.
        - swap(self, i, j): Swaps two elements in the heap at indices i and j.
        - is_empty(self): Returns True if the heap is empty, False otherwise.

    """
    
    def __init__(self):
        """
        Function description: 
            Initializes a new instance of the MinHeap class. This method sets up the heap as an empty list,
            ready to store elements in a way that maintains the heap property.

        Approach description: 
            The MinHeap uses a dynamic array to store the elements. The root of the heap
            is at index 0, and for any element at index i, its children are at indices 2*i + 1 and 2*i + 2.

        Input: 
            None

        Output: 
            None.

        Time complexity: 
            O(1)

        Time complexity analysis: 
            The constructor only allocates space for an empty list, which is an O(1) operation.
            Constant time complexity because it only initializing an empty list.

        Space complexity: 
            O(1)

        Space complexity analysis: 
            The space is used to initialize the empty list (heap), which doesn't depend on the number of elements and is therefore constant.
        """
        self.heap = []

    def push(self, item):
        """
        Function description: 
            Adding an item to the heap preserves the min-heap attribute, which states that the parent node's size is always less 
            than or equal to that of its children. This is achieved by adding the item to the end of the heap and, in the event 
            that the heap property is broken, executing a sift-up operation to restore it.

        Approach description: 
            At first, the item is appended to the end of the list (heap). To make sure the min-heap property is preserved, 
            the sift-up process is then invoked. To do this, compare the new element with its parent and swap if the parent 
            is bigger than the child. This process is repeated until the heap's root is reached or there is no need for a swap.

        Input: 
            item: The item to be added to the heap. 

        Output: 
            None.

        Time complexity: 
            O(log n),  where 'n' is the number of elements in the heap prior to the addition.

        Time complexity analysis: 
            It takes O(1) to add an element to the end of the list. At most log(n) levels, the sift-up operation must travel 
            from the leaf to the root of the heap in the worst-case scenario.

        Space complexity: 
            O(1)

        Space complexity analysis: 
            The method used is constant because it utilises the same amount of space regardless of how many elements are in the heap. 
            The push operation takes a fixed amount of additional space and alters the heap that is already in place.

        """

        self.heap.append(item)
        self.sift_up(len(self.heap) - 1)


    def pop(self):
        """
        Function description: 
            Removes and returns the smallest item from the heap. This function ensures that the heap
            property is maintained after the removal of the root element, which is the smallest.

        Approach description: 
            By replacing the last element in the heap with the smallest element at the root, the method removes the last element,
            which is now the smallest, and then sifts down the new root element to restore the heap property.

        Input: 
            None

        Output: 
            Returns the smallest item from the heap. If the heap is empty, an IndexError is raised.

        Time complexity: 
            O(log n), Where 'n' is the number of elements in the heap.

        Time complexity analysis: 
            O(1) is the result of swapping the last item's root. In the worst scenario, the sift-down process 
            must descend down at most log(n) levels, traversing from the heap's root to its leaf.

        Space complexity: 
            O(1)

        Space complexity analysis: 
            The method maintains a consistent space utilisation by operating directly on the heap and not 
            allocating any additional substantial memory. The pop operation requires a fixed amount of space 
            and alters the heap that is already in place.
        """

        # Check if the heap contains only one element
        if len(self.heap) == 1:
            return self.heap.pop()

        # Ensure there are elements in the heap before proceeding
        if self.heap:
            # Swap the root with the last item and remove the smallest element
            self.swap(0, len(self.heap) - 1)
            item = self.heap.pop()
            # Restore the heap property by sifting down the new root
            self.sift_down(0)
            return item
        
        # Raise an error if the heap is empty and pop is attempted
        raise IndexError("pop from empty heap")


    def sift_up(self, index):
        """
        Function description: 
            Moves the item at the specified index up to its correct position in the heap to maintain
            the heap property. This method is typically called after insertion of a new element at the end
            of the heap.

        Approach description: 
            The method compares the item at the given index with its parent; if the item is less than its parent,
            they are swapped. This process continues recursively until the item is either at the root of the heap,
            or no further swaps are needed.

        Input: 
            index: The index of the element that needs to be moved up in the heap.

        Output: 
            None.

        Time complexity: 
            O(log n), Since in the worst case, the element needs to be moved up from the bottom of the heap to the root.

        Time complexity analysis: 
            Because the heap's height is log(n), the operation may involve at most log(n) swaps (where n is the current
            number of elements in the heap).

        Space complexity: 
            O(1), The operation uses constant space, modifying the heap in place.

        Space complexity analysis: 
            Only the temporary storage required for element swapping is occupied by the space; no additional structures are used.
        """

        # Calculate the parent index
        parent = (index - 1) // 2
        if parent < 0:
            return

        # If the current element is less than its parent, swap them and continue sifting up
        if self.heap[index] < self.heap[parent]:
            self.swap(index, parent)
            self.sift_up(parent)


    def sift_down(self, index):
        """
        Function description: 
            Moves the item at the given index down to its correct position within the heap to maintain
            the heap property. This method is used primarily after removal of the root element or when
            an element's value increases and may violate the heap property.

        Approach description: 
            Starting from the given index, the method checks the children of the current node. It swaps the
            current node with the smallest of its children and continues this process recursively until the
            node is in its correct position or it has no children smaller than itself.

        Input: 
            index: The index of the element that needs to be moved down in the heap.

        Output: 
            None.

        Time complexity: 
            O(log n), The procedure may continue from the given index down to the leaf nodes, thus in the worst case,
            it travels down the height of the heap.

        Time complexity analysis: 
            The depth of a heap is log(n) where n is the number of elements in the heap, and at each step of the sift down,
            the procedure could traverse one level down the tree.

        Space complexity: 
            O(1), The sifting operation modifies the heap in place and does not require additional significant space.

        Space complexity analysis: 
            Only a few extra variables are needed for the operation, such as child indices, so the space usage is constant.
        """

        # Get the index of the first child
        child = 2 * index + 1
        if child >= len(self.heap):
            return  # No children, end sifting

        # Check if there's a second child and if it's smaller
        if child + 1 < len(self.heap) and self.heap[child + 1] < self.heap[child]:
            child += 1

        # If the child is smaller than the current node, swap and continue
        if self.heap[child] < self.heap[index]:
            self.swap(index, child)
            self.sift_down(child)

    def swap(self, i, j):
        """
        Function description: 
            Swaps two elements in the heap. This method is a utility function used by other heap operations
            to maintain the heap properties during insertion and deletion.

        Approach description: 
            The method directly swaps the elements at the provided indices within the internal heap array.

        Input: 
            i: The index of the first element to swap.
            j: The index of the second element to swap.

        Output: 
            None.

        Time complexity: 
            O(1), Swapping two elements in an array is a constant time operation.

        Time complexity analysis: 
            The operation accesses the array at two indices and swaps the elements, all of which are constant time operations.

        Space complexity: 
            O(1)

        Space complexity analysis: 
            The method only modifies the existing array and does not allocate any new memory.
        """

        # Perform the swap of two elements in the heap
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def is_empty(self):
        """
        Function description: 
            Checks whether the heap is empty. This utility function allows other components or users
            of the MinHeap to verify the state of the heap before performing operations that require non-empty heaps.

        Approach description: 
            This method returns the result of evaluating whether the heap's internal list is empty.

        Input: 
            None

        Output: 
            Returns True if the heap is empty, False otherwise.

        Time complexity: 
            O(1), Checking if a list is empty is a constant time operation.

        Time complexity analysis: 
            The method simply checks the length of the list, which is a direct memory access operation in Python.

        Space complexity: 
            O(1).

        Space complexity analysis: 
            No new memory is allocated during the operation, thus it uses only constant space.
        """

        # Return whether the heap list is empty
        return len(self.heap) == 0

    
class TreeMap:
    """
    Fucntion description:
        A class to represent the forest graph that is filled with trees, solulu trees and roads to optimized in finding 
        the shortest path using the Dijkstra's algorithm. The class will handle different possibilities to find shortest 
        paths including a special node of 'solulus' that has a unique property including destruction time and teleportation.
    
    Approach description: 
        The class uses Dijkstra's algorithm to find the shortest path after meeting and including different conditions
        such as encountering a solulu and handling destruction time and teleportation. 

    Attributes:
        graph: a list of list of tuples that represents the forest as an adjacency list containing nodes with a list of 
        tuples(v,w), where 'v' is an adjacent node and 'w' is the weight of the edge from node(u) to node(v).
        solulu: a list of tuples that is represented as a node but with additional data including destruction time and 
        teleportation node.

    Methods:
        - __init__(self, roads, solulus): Initializes the TreeMap with roads and solulus.
        - dijkstra(self, start): Finding the shortest path from the start node by initializing the Dijkstra's algorithm.
        - find_minimum_route(self, shortest_road): Identifying the shortest travel time.
        - escape(self, start, exits): Determining the shortest path and shortest travel time from the start node to an 
        exit node with at least one solulu visited 
    """

    def __init__(self, roads, solulus):
        """
        Function description: 
            Initializes the TreeMap with roads and solulus as an adjacency list. This method stores the graph that 
            is represented as an adjacency list with roads and solulus containing teleporation node and destroy time.

        Approach description: 
            Determines the maximum tree index from list of roads inputted to initialize the adjacency list and intializes
            the graph with a list of empty list representing a node and the road that connects the nodes to other nodes 
            with the weight. The solulus are stored within the graph.

        Input: 
            roads: list of tuples (u,v,w) where 'u' is current node, 'v' is the target node and 'w' is the weight of the edge
            from node(u) to node(v).
            solulus: list of node with additional functionality such as teleport node and destroy time.

        Output: 
            None

        Time complexity: 
            O(|T| + |R|), where 'T' is the number tree nodes and 'R' is the number of roads.

        Time complexity analysis: 
            The method iterates through the length of roads list to initialize the graph which takes O(R) time and as the number 
            of nodes depends on the list of roads, the complexity becomes O(T + R).

        Space complexity: 
            O(T + R), where 'T' is the space for the nodes in the graph and 'R' is the space to store the connections between nodes.

        Space complexity analysis: 
            The node and its connections uses space equal to the number of nodes and connections to store within the graph. As worst case
            space complexity is calculated, when 'solulus' stored, it is determined by the size of input that makes the total space worst case
            complexity is O(T + R).
        """

        # Initializing the maximum index of a node depending on the list of roads
        tree_length = 0
        for u, v, w in roads:
            tree_length = max(tree_length, u, v)

        # Initialize the graph with empty lists 
        self.graph = []

        # Adding the zero index 
        tree_counts = tree_length + 1  
        for _ in range(tree_counts):
            self.graph.append([])

        # Initialize the graph to include properties of the road
        for u, v, w in roads:
            # Adding the direction to be one way
            self.graph[u].append((v, w))  

        # Store the solulus
        self.solulu = solulus

    def dijkstra(self, start):
        """
        Function description:
            Initializing the Dijkstra's algorithm through implementing priority queue with minheap to find the shortest path 
            to the next node and to all other nodes in the graph.

        Approach description:
            This function is using minheap to initialize the list of distances with infinity and the start node to zero to
            store nodes along with the distances to achieve the smallest distance. This function will process the node once
            and updates the distance to the adjacent nodes if exist a shorter path.
        
        Input:
            start: The starting node index.

        Output:
            distances: a list with the shortest distances to each adjacent nodes
            previous: a list with predecessors from the start node to each node

        Time complexity:
            O((V + E) log V), where 'V' is the number of vertices and 'E' is the number of edges 

        Time complexity analysis:
            Each vertex will be pushed using the priority queue (push and pop) that takes O(log V) time for each edge and vertex.

        Space complexity:
            O(V), where the complexity is for storing the 'distances', 'previous' and 'visited_node' lists that has size of V.

        Space complexity analysis:
            The complexity to store the 'distances', 'previous' and 'visited_node' is equal to the number of node that means the total space used to store is equal with the number of nodes.
        """
        l = len(self.graph)  # Initialize the length of the graph
        distances = [float('inf')] * l  # Initialize the distance from start to adjacent node to infinity
        previous = [None] * l  # Initialize previous node 
        distances[start] = 0  # Initialize the distance from start node to zero 
        minHeap = MinHeap()  # Initialize the minheap as priority queue

        # Initialize the min heap to have each nodes and the distances from one node to another
        for node in range(l):
            minHeap.push((distances[node], node))

        visited_node = [False] * l  # Initialize the visted node to track nodes that has been visited 

        # Initialize the minheap while it is empty
        while not minHeap.is_empty():
            min_dist, min_vertex = minHeap.pop()  # Initialize the node with the shortest distance 

            # If the node has been visited, continue
            if visited_node[min_vertex]:  
                continue
            
            visited_node[min_vertex] = True  # Initialize the node as visited 

            # Initialize process for adjacent nodes
            for n, w in self.graph[min_vertex]:
                # If the node has not been visited and the shortest distance with the weight is lesser
                if not visited_node[n] and min_dist + w < distances[n]:
                    distances[n] = min_dist + w  # Update the shortest distance
                    previous[n] = min_vertex  # Update the previous node
                    minHeap.push((distances[n], n))  # Push updated distance to the heap

        return distances, previous  # Return distances and previous  
    
    def find_minimum_route(self, shortest_road):
        """
        Function description:
            This function returns the shortest road with the shortest travel time from a list of route where the first
            element is the shortest travel time and the rest of the list is the path from start to exit.

        Approach description:
            This function will calculate and determine the shortest travel time after iterating through the list of path
            and determining the shortest travel time of each path with the current shortest travel time. 

        Input:
            shortest_road: list of tuples containing the travel time and the path.

        Output:
            Returns a tuple of the path with shortest travel time

        Time complexity:
            O(n), where 'n' is the number of routes.

        Time complexity analysis:
            An iteration will over the list of routes to identify the shortest travel time from each route to find the 
            best shortest travel time.

        Space complexity:
            O(1), the space complexity is constant as the method only uses a fixed amount of additional space
                (for storing the minimum route during the iteration).

        Space complexity analysis:
            There is no additional space used as the amount of space is constant for storing the best shortest travel time.
        """
        # If the current is not the shortes travel time
        if not shortest_road:
            return None  # Return None

        # Initialize the first route as the shortest travel time
        min_road = shortest_road[0]

        # Iterate through the paths to find the shortest travel time
        for route in shortest_road:
            # If current route has shorter travel time
            if route[0] < min_road[0]:
                min_road = route  # Initialize the current path to have the shortest travel time

        return min_road  # Return the path with shortest travel time

 
    def escape(self, start, exits):

        """
        Function description: 
            The function calculates the shortest path from the start to any exit after visiting at least one solulu where the 
            solulu has additional data such as destruction time and teleport node that will be calculated to find the shortest
            travel time.

        Approach description: 
            Shortest path from the start to each node using Dijkstra's algorithm is firstly calculated and when solulu node 
            is visited, then calculates the shortest path from the new teleported node to an exit. The function calculates the 
            total travel time of the path from start to exit after visiting one solulu node including the destruction time.

        Input: 
            start: starting node index.
            exits: list of possible exit node.

        Output:
            Returns a tuple with shortest travel time including with the destruction time and the path from start to exit 
            after visiting a solulu.
            
        Time complexity:
            O((V + E) log V * S), where 'V' is the number of nodes, 'E' is the number of edges, and 'S' is the number of solulu nodes. 
            This complexity is due to running Dijkstra's algorithm
            from the start node and again from each solulu's teleportation node.

        Time complexity analysis:
            The Dijkstra's algorithm will be initialized twice from the start and once again from the teleport node after 
            destroying the solulu. The initialization of the Dijkstra's algorithm is O((V + E) log V) and is dependent
            on the number of 'S' solulu.

        Space complexity:
            O(V + S), the space complexity is used to store the distances and paths after each initialization of the
            Dijkstra's algorithm to store into the list holding the path from start to exit. 

        Space complexity analysis:
            Initializations of the Dijkstra's algorithm and storing the outputs are calculated and using space.
        """
        start_dist, solulu_pred = self.dijkstra(start)  # Initialize start distance and solulu predecessors from the start node
        shortest_road = []  # Initialize a list to store the escape path

        inf = float('inf')  # Initialize the infinity 

        # Iterate each solulu to discover all escape paths from start to exit through at least one solulu
        for solulu in self.solulu:
            solulu_node, destroy_time, teleported_ID = solulu
            teleport_dist, solulu_pred2 = self.dijkstra(teleported_ID)  # Initialize the teleport distance from the solulu's teleported node ID

            min_exit_dist = inf  # Initialize the shortest travel time to keep track 
            # Iterate to see each exits
            for e in exits:
                if teleport_dist[e] < min_exit_dist:
                    min_exit_dist = teleport_dist[e]
                    min_exit = e  # Initialize to contain the shortest travel time to an exit

            # Check if the path reaches an exit 
            if min_exit_dist != inf:
                total_time = start_dist[solulu_node] + destroy_time + min_exit_dist  # Calculate the total time for the path

                # Initialize the path to the solulu node
                solulu_road = []
                pointer = solulu_node
                while pointer is not None:
                    solulu_road.append(pointer)
                    pointer = solulu_pred[pointer]
                solulu_road.reverse()  # Reverse to get the order to ensure it is from start to solulu 

                # Initialize the path from the teleported node to an exit 
                exit_road = []
                pointer = min_exit
                while pointer != teleported_ID:
                    exit_road.append(pointer)
                    pointer = solulu_pred2[pointer]
                if solulu_road[-1] != teleported_ID:
                    exit_road.append(teleported_ID)
                exit_road.reverse()  # Reverse to get the order to ensure it is from teleported node to exit

                # Append the complete route and its total time with the path
                shortest_road.append((total_time, solulu_road + exit_road))

        # Finding the shortest path from all the discovered paths 
        shortest_road = self.find_minimum_route(shortest_road)
        return shortest_road  # Return the shortest path