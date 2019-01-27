import random

def get_random_percentage_lists(vec_list_A, vec_list_B, size):
    if not vec_list_A or not vec_list_B:
        return [], []

    list_A_shuffle = vec_list_A.copy()
    random.shuffle(list_A_shuffle)

    list_B_shuffle = vec_list_B.copy()
    random.shuffle(list_B_shuffle)

    # liste mit listen, um später mit all_lists[0] auf list_A_shuffle zuzugreifen und mit all_lists[1] auf list_B_shuffle
    all_lists = [list_A_shuffle, list_B_shuffle]

    # 0.0 - 1.0
    random_percent = size

    # anzahl aller elemente von list_A_new und list_B_new zusammen (bsp 20% bzw 0.2 -> len(list_A) + len(list_B) = 10; 10*0.2 -> 2; also 2 Elemente)
    len_lists_total = int((len(vec_list_A) + len(vec_list_B)) * random_percent)

    # liste mit listen, um später mit all_lists_new[0] auf list_A_new zuzugreifen und mit all_lists_new[1] auf list_B_new
    all_lists_new = [[], []]

    # für die anazahl aller Elemente, die wir hinzufügen:
    for _ in range(len_lists_total):
        # entweder 0 oder 1 -> all_lists[0] = list_A_shuffle und all_lists_new[0] = list_A_new
        random_list_index = random.getrandbits(1)
        # entweder list_A_shuffle oder list_B_shuffle
        random_list = all_lists[random_list_index]
        # wenn die liste leer ist, wir also alle elemente schon in die neue liste gepackt haben, die andere liste benutzen
        if len(random_list) == 0:
        # toggle index: 1 wird 0, 0 wird 1
            random_list_index = 1 - random_list_index
        # also wird die andere liste gewählt
            random_list = all_lists[random_list_index]
        # letztes element entfernen, wegen shuffle ist es ein zufälliger wert
        random_value = random_list.pop()
        # das entfernte element in neue liste packen. Wegen selben index kommen werte aus list_A_shuffle in list_A_new und werte aus list_B_shuffle in list_B_new
        all_lists_new[random_list_index].append(random_value + (True,))

    list_A_new, list_B_new = all_lists_new
    
    if not list_A_new:
        list_A_new.append(all_lists[0].pop() + (True,))
    if not list_B_new:
        list_B_new.append(all_lists[1].pop() + (True,))
    
    while(len(list_A_new) != len(vec_list_A)):
        list_A_new.append(all_lists[0].pop() + (False,))
    while(len(list_B_new) != len(vec_list_B)):
        list_B_new.append(all_lists[1].pop() + (False,))
    return list_A_new, list_B_new
