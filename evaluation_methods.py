import random
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sts_methods as sts 

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

def pos_filter_similarity(data_frame,emb, methods, edit_distance_dic=None, jaccard_distance_dic=None):
    pos_filter=['NN', 'VB', 'JJ']
    pos_list=[]
    pearson_score=[]
    spearman_score=[]
    method_list=[]
    for method in methods:
        for pos in pos_filter:
            data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2', 'pos_cov_avr_s1', 'pos_cov_avr_s2']]=data_frame.apply(
            lambda row: pd.Series(
                sts.rnd_sentence_similarity(emb ,row['sentence_1'],row['sentence_2'], 1, pos_filter=pos, percent_mode=True, method=method, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic)), axis=1)
            
            p_correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
            s_correlation, _ = spearmanr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
            method_list.append(method)
            pearson_score.append(round(p_correlation*100,2))
            spearman_score.append(round(s_correlation*100,2))
            if pos == 'NN':
                pos='Nomen'
            elif pos == 'VB':
                pos='Verben'
            elif pos == 'JJ':
                pos='Adjektive'
            pos_list.append(pos)

            avrg_cov = np.mean(data_frame[['cov_avrg_s1','cov_avrg_s2']].values)
            avrg_pos_cov = np.mean(data_frame[['pos_cov_avr_s1','pos_cov_avr_s2']].values)
            
            print(pos+' --------------------'+ ': ' + method )
            print('Pearson - Correlation: '+ str(round(p_correlation*100,2)))
            print('Spearman - Correlation: '+ str(round(s_correlation*100,2)))
            print('Avr_Coverage: '+str(avrg_cov))
            print('POS_Coverage: ' + str(avrg_pos_cov))
    df = pd.DataFrame({'Pearson': pearson_score, 'Spearman': spearman_score, 'Wortart':pos_list, 'Methoden': method_list})
    f = sns.barplot(x='Wortart', y='Pearson',data=df, hue='Methoden')
    plt.show()
    f = sns.barplot(x='Wortart', y='Spearman',data=df, hue='Methoden')
    print(df)
    plt.show()

def missing_words_similarity(data_frame,emb, edit_distance_dic=None, jaccard_distance_dic=None ,methods=['0-vector']):
    missing_sentence_count=set((data_frame['missing_emb'].tolist()))
    method_list =[]
    sentence_count = []
    aktual_miss=[]
    pearson_score=[]
    spearman_score=[]
    for method in methods:
        for word_miss in missing_sentence_count:
            miss_data = data_frame.query('missing_emb =='+str(word_miss))
            miss_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=miss_data.apply(
                lambda row: pd.Series(sts.rnd_sentence_similarity(emb,row['sentence_1'],row['sentence_2'], 1, percent_mode=True,method=method, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic)), axis=1)
            p_correlation, _ = pearsonr(miss_data['pred_value'].tolist(), miss_data['gold_value'].tolist())
            s_correlation , _ = spearmanr(miss_data['pred_value'].tolist(), miss_data['gold_value'].tolist())
            avrg_cov = np.mean(miss_data[['cov_avrg_s1','cov_avrg_s2']].values)
            if len(miss_data.index) > 100:
                sentence_count.append(len(miss_data.index))
                method_list.append(method)
                pearson_score.append(round(p_correlation*100,2))
                spearman_score.append(round(s_correlation*100,2))
                aktual_miss.append(word_miss)
            print('--------------- ' + str(word_miss)+ ': ' + str(method))
            print('Pearson - Correlation: '+ str(p_correlation*100))
            print('Spearman - Correlation: '+ str(s_correlation*100))
            print('Sentence Count: '+ str(len(miss_data.index)))
            print('Avr_Coverage: '+ str(avrg_cov))
    df = pd.DataFrame({'Pearson': pearson_score, 'Spearman': spearman_score, 'Anzahl der fehlenden Wörter':aktual_miss, 'Methoden': method_list, 'Anzahl der Satzpaare': sentence_count})
    print(df)
    g = sns.factorplot(x="Anzahl der fehlenden Wörter", y="Pearson", hue='Methoden', data=df)
    g.set(ylim=(0, 100))
    g = sns.factorplot(x="Anzahl der fehlenden Wörter", y="Spearman", hue='Methoden', data=df)
    g.set(ylim=(0, 100))
    plt.show()
    f = sns.barplot(x='Anzahl der fehlenden Wörter', y='Anzahl der Satzpaare',data=df)
    plt.show()

def method_evaluation(data_frame,emb, edit_distance_dic=None, jaccard_distance_dic=None ,methods=['0-vector']):
    result_dict={}
    for method in methods:
        data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=data_frame.apply(
        lambda row: pd.Series(sts.rnd_sentence_similarity(
        emb,row['sentence_1'],row['sentence_2'], size=1, percent_mode=True, method=method, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic)), axis=1)
        p_correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
        s_correlation, _ = spearmanr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
        result_dict[method+'_p'] = p_correlation
        result_dict[method+'_s'] = s_correlation
        print('--------------- : ' + str(method))
        print('Pearson - Correlation: '+ str(p_correlation*100))
        print('Spearman - Correlation: '+ str(s_correlation*100))
    return result_dict

def evaluate_rnd_coverage_emb(emb, data_frame, iterations, percent_mode, methods=None, edit_distance_dic=None, jaccard_distance_dic=None, correlationMethod='P'):
    if percent_mode is True:
        size=1
    else:
        size =1
    result_dict={}
    method_list = [ item for item in methods for _ in range(iterations) ] 
    for method in methods:
        for _ in range(10):
            result_list=[]
            coverage_list=[]
            i=0
            while i < iterations:
                data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=data_frame.apply(
                    lambda row: pd.Series(sts.rnd_sentence_similarity(
                        emb,row['sentence_1'],row['sentence_2'], size=size, percent_mode=percent_mode, method=method, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic)), axis=1)
                if correlationMethod == 'P':
                    correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
                elif correlationMethod == 'S':
                    correlation, _ = spearmanr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
                result_list.append(round(correlation*100,2))
                avr_cov = np.mean(data_frame[['cov_avrg_s1','cov_avrg_s2']].values)
                coverage_list.append(avr_cov)
                i+=1
            avr_cov = int(np.mean(coverage_list)*100)
            avr_correlation = np.mean(result_list)
            min_correlation = np.min(result_list)
            max_correlation = np.max(result_list)
            print('--------------- ' + str(size)+ ': '+ method)
            print('Min: '+str(min_correlation), 'Max: '+ str(max_correlation))
            print('Avrg: '+ str(avr_correlation))
            print('Avr_Coverage: '+ str(avr_cov)+"%")

            if avr_cov in result_dict:
                result_dict[avr_cov] = result_dict[avr_cov] + result_list
            else:
                result_dict[avr_cov]=result_list
            if percent_mode is True:
                size=round(size-0.1,1)
            else:
                size+=1
        size=1
    result_dict['Methoden'] = method_list
    df = pd.DataFrame.from_dict(result_dict)
    if correlationMethod == 'P':
        df = df.melt('Methoden', var_name='Wortabdeckung in %', value_name='Pearson')
        ax = sns.pointplot(x='Wortabdeckung in %', y='Pearson', hue='Methoden', data=df)
    elif correlationMethod == 'S':
        df = df.melt('Methoden', var_name='Wortabdeckung in %', value_name='Spearman')
        ax = sns.pointplot(x='Wortabdeckung in %', y='Spearman', hue='Methoden', data=df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(methods)], labels[:len(methods)]).set_title('Methoden')
    ax.set(ylim=(0, 100))
    plt.gca().invert_xaxis()
    plt.show()
    return None