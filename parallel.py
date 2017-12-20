import multiprocessing
from six.moves import builtins, range


def parallelMap(f, items, nProcs=None):
    if nProcs == 1:
        return builtins.map(f, items)
    else:
        name = parallelMap.i
        parallelMap.scope[name] = (f, items)
        parallelMap.i += 1
        pool = multiprocessing.Pool(nProcs)
        result = pool.map(_parallel_func, ((name, i)
                                           for i in range(len(items))))
        pool.close()
        del parallelMap.scope[name]
        return result


parallelMap.i = 0
parallelMap.scope = {}


def _parallel_func(arg):
    name, index = arg
    f, items = parallelMap.scope[name]
    return f(items[index])


map = parallelMap