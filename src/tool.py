

def read_item_index_to_entity_id_file(item_index2entity_id, entity_id2item_index):
    file = '../chinese_medicine_data/item_index2entity_id_rehashed.txt'
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        item_index2entity_id[item_index] = i
        entity_id2item_index[i] = item_index
        i += 1

class Exercise(object):
    def __init__(self, id_number, scores, exercises_id):
        self.exercises_id = exercises_id
        self.id_number = id_number
        self.scores = scores


def get_top_k_exercises_id(args, scores, entity_id2item_index):
    scores_arr = dict()
    result_arr = dict()
    i = 0
    for score in scores:
        scores_arr[i] = Exercise(i, score, entity_id2item_index[i])
        i += 1
    quickSort(scores_arr, 0, len(scores_arr) - 1)
    for n in range(args.top_k):
        result_arr[n] = scores_arr[n]
    return result_arr

def quickSort(arr, s, e):
    if not arr:
        return None
    if s > e:
        return None
    p = arr[s].scores
    i = s
    j = e
    while i < j:
        while i < j and arr[j].scores <= p:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
        while i < j and arr[i].scores > p:
            i += 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]

    quickSort(arr, s, i - 1)
    quickSort(arr, i + 1, e)