"""
Class for a bunch of random data structures
Includes:
FIFO - First In First Out

To Be Added:
LIFO - Last In First Out
LILO - First In Last Out
Queue - Version of FIFO with methods to operate relative to time/completion rather than length
Linked List - Because why not
"""


"""
First In First Out data structure class
"""
class FIFO:
    def __init__(self , max_len: int = None , initial_state = [],
                 reverse: bool = False):
        self.arr = initial_state
        self.max_len = max_len
        self.index = 0
        self.reverse = reverse

    def append(self , val):
        if self.max_len is None:
            if self.reverse is False:
                self.arr.append(val)
            else:
                self.arr.insert(0 , val)
            return

        if len(self.arr) < self.max_len:
            if self.reverse is False:
                self.arr.append(val)
            else:
                self.arr.insert(0 , val)
        else:
            self.arr = self.arr[len(self.arr) + 1 - self.max_len:]
            if self.reverse is False:
                self.arr.append(val)
            else:
                self.arr.insert(0 , val)

    def change_max(self , new_max):
        self.max_len = new_max

    def get_max(self):
        return self.max_len

    def str_concat(self , regex: str):
        returned = ''
        for val in self.arr:
            if val != '':
                returned += val + regex

        return returned[:len(returned) - len(regex)]

    def reset(self):
        self.arr.clear()

    def __list__(self):
        return self.arr.copy()

    def __str__(self):
        return f'{self.arr} <max_len: {self.max_len}>'

    def __len__(self):
        return len(self.arr)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.arr):
            current = self.arr[self.index]
            self.index += 1
            return current
        raise StopIteration

    def __getitem__(self, key):
        return self.arr[key]
