def indexInList(k, lst):
    for i in lst:
        try:
            return lst.index(k)
        except ValueError:
            return -1
