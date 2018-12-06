import numpy as np

def evaluationA(tokens_l1, tokens_l2, embeddings_l1, embeddings_l2, agregation='avg'):
    return None

def evaluationB(predicted_path, gold_path):
    correct_pred=0
    nearly_correct=0
    count=0
    meh_correct=0
    wrong_pred=0
    with open(predicted_path) as p, open(gold_path) as g:
        for pred_line, gold_value in zip(p.read().splitlines(), g.read().splitlines()):
            pred_line = pred_line.split('\t')
            pred_value = float(pred_line[0])
            gold_value = float(gold_value)
            count+=1
            if pred_value - gold_value == 0:
                correct_pred+=1
            if abs(pred_value-gold_value) <= 0.3:
                nearly_correct+=1
            if round(pred_value)==round(gold_value):
                meh_correct+=1
            else:
                wrong_pred+=1
    if count != 0:
        avr_correct=correct_pred/count
        avr_nearly_correct= nearly_correct/count
        avr_meh_correct= meh_correct/count
        avr_wrong= wrong_pred/count
    return avr_correct,avr_nearly_correct,avr_meh_correct,avr_wrong