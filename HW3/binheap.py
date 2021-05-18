from typing import TypeVar, Generic, Union, List

T = TypeVar('T')

######## THIS VERSION HAS BEEN MODIFIED TO HANDLE COMPARISON BETWEEN VERTEX KEYS ##########

def min_order(a, b) -> bool:
    return a.key <= b.key          #changed to handle vertex keys

def max_order(a, b) -> bool:
    return a.key >= b.key          #changed to handle vertex keys

class binheap(Generic[T]):

    def __init__(self, A: Union[int, List[T]], total_order = None):

        if total_order is None:
            self._torder = min_order
        else:
            self._torder = max_order

        if isinstance(A, int):
            self._size = 0
            self._A = [None]*A
        else:
            self._size = len(A)
            self._A = A
        
        self._build_heap()

    # added static bc methods don't rely on the heap itself
    @staticmethod
    def parent(node: int) -> Union[int, None]:
        if node == 0:
            return None

        return (node-1)//2

    @staticmethod
    def child(node: int, side: int) -> int:
        return 2*node + 1 + side

    @staticmethod
    def left(node: int) -> int:
        return 2*node + 1

    @staticmethod
    def right(node: int) -> int:
        return 2*node + 2
    
    def is_empty(self) -> bool:
        return self._size == 0
 
    def __len__(self):
        return self._size

    def _swap_keys(self, node_a: int, node_b: int) -> None:
        tmp = self._A[node_a]
        self._A[node_a] = self._A[node_b]
        self._A[node_b] = tmp

        #additional swap needed to ensure node.heap_idx is also changed accordingly

        tmp = self._A[node_a].heap_idx
        self._A[node_a].heap_idx = self._A[node_b].heap_idx
        self._A[node_b].heap_idx = tmp

    # iterative version
    def _heapify(self, node: int) -> None:
        keep_fixing = True

        while keep_fixing:
            min_node = node
            for child_idx in [2*node + 1, 2*node + 2]:
                if child_idx < self._size and self._torder(self._A[child_idx], self._A[min_node]):
                    min_node = child_idx

            # min_node is the index of the minimum key 
            # among the keys of root and its children

            if min_node != node:
                self._swap_keys(min_node, node)
                node = min_node
            else:
                keep_fixing = False


    # returns min without removing it (not sure if needed)

    def return_min(self) -> T:
        if self.is_empty():
            raise RuntimeError('The heap is empty')
        
        return self._A[0]


    def remove_minimum(self) -> T:
        if self.is_empty():
            raise RuntimeError('The heap is empty')

        self._swap_keys(0, self._size-1)

        # self_A[0] = self._A[self._size-1]

        self._size = self._size-1

        self._heapify(0)

        
        return self._A[self._size]

    def _build_heap(self) -> None:
        for i in range(binheap.parent(self._size-1), -1, -1):
            self._heapify(i)
    
    def decrease_key(self, node: int, new_value: T) -> None:
        #if self._torder(self._A[node], new_value):
        #    raise RuntimeError(f'{new_value} is not smaller than {self._A[node]}')
    

        # In Dijkstra's Algorithm we are sure we can always decrease value without problem since check is made
        # beforehand.

        #directly assign value.

        self._A[node].key = new_value

        parent = binheap.parent(node)

        while node != 0 and not self._torder(self._A[parent], self._A[node]):
            self._swap_keys(node, parent)
            node = parent
            parent = binheap.parent(node)

    def insert(self, value: T) -> None: #not changed since not needed for Dijkstra's algorithm
        if self._size >= len(self._A):
            raise RuntimeError('The heap is full')
            
        if self.is_empty():
            self._A[0] = value
            self._size += 1
        else:
            parent = binheap.parent(self._size)
            if self._torder(self._A[parent], value):
                self._A[self._size] = value
            else:
                self._A[self._size] = self._A[parent]
                
            self._size += 1
            self.decrease_key(self._size - 1, value)
    
    # nice print 
    def __repr__(self) ->str:
        br_str = ''

        next_node = 1
        up_to = 2

        while next_node <= self._size:
            level = '\t'.join(f'{v}' for v in self._A[next_node-1: min(up_to-1, self._size)])

            if next_node == 1:
                bh_str = level
            else:
                bh_str += f'\n{level}'

            next_node = up_to
            up_to = 2*up_to

        return bh_str
