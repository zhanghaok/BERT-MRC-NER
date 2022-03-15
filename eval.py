import torch
from transformers import BertTokenizer
from model  import MRCModel
from load_data import TEST_DATA_PATH,tempalte,MAX_LEN
from logger import logger


tokenizer = BertTokenizer.from_pretrained('/home/zhk/workstation/bert-crf/bert-base-chinese')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MRCModel.from_pretrained('./saved_model')
model.to(device)
model.eval()

def extract(chars,tags):
    """抽取真实的实体以及对应的标签 gold entity"""
    result = []
    pre = ''
    w = []
    for idx, tag in enumerate(tags):
        if not pre:
            if tag.startswith('B'):
                pre = tag.split('-')[1]
                w.append(chars[idx])
        else:
            if tag == f'I-{pre}':
                w.append(chars[idx])
            else:
                result.append([w,pre])
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
            
    return [[''.join(x[0]), x[1]] for x in result]

def mrc_decode(start_pred,end_pred,raw_text):
    pred_entities = []
    for i,s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j,e_type in enumerate(end_pred[i:]):
            if s_type == e_type:#找离start最近的一个span，所以下面有break语句
                tmp_ent = raw_text[i:i+j+1]
                pred_entities.append(tmp_ent)
                break

    return pred_entities

gold_num = 0
predict_num = 0
correct_num = 0
logger.info("*"*15+"Start Testing"+"*"*15)
with open(TEST_DATA_PATH,encoding='utf-8') as rf:
    chars = []
    labels = []
    origin_labels = []
    for line in rf:
        if line != '\n':
            char,label = line.strip().split()
            chars.append(char)
            origin_labels.append(label)
            if '-' in label:
                label = label.split('-')[1]
            labels.append(label)
        else:
            sent = ''.join(chars)
            logger.info("Sent:%s"%sent)
            entities = extract(chars,origin_labels)
            gold_num += len(entities)
            logger.info("NRE:%s"%entities)

            pred_entities = []
            for predix, target in tempalte:
                input_ids_1 =[tokenizer.convert_tokens_to_ids(c) for c in predix]
                input_ids_1 = [tokenizer.cls_token_id] + input_ids_1 + [tokenizer.sep_token_id]
                token_type_ids_1 = [0]*len(input_ids_1)
                if len(chars)+1+len(input_ids_1) > MAX_LEN:
                    chars = chars[:MAX_LEN-1-len(input_ids_1)]

                input_ids_2 = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                input_ids_2 = input_ids_2 + [tokenizer.sep_token_id]
                token_type_ids_2 = [1]*len(input_ids_2)

                input_ids = input_ids_1 + input_ids_2
                input_ids =torch.tensor(input_ids,dtype=torch.long).unsqueeze(0).to(device)#模型的输入需要3个维度，增加一个batch维度为1
                token_type_ids = token_type_ids_1 + token_type_ids_2
                token_type_ids = torch.tensor(token_type_ids,dtype=torch.long).unsqueeze(0).to(device)#模型的输入需要3个维度，增加一个batch维度为1
                attention_mask = [1]*len(input_ids)
                attention_mask = torch.tensor(attention_mask,dtype=torch.long).unsqueeze(0).to(device)#模型的输入需要3个维度，增加一个batch维度为1

                start_pred, end_pred = model(input_ids,attention_mask,token_type_ids)
                start_pred = start_pred.squeeze(0)[len(input_ids_1):-1]#因为模型的输出两个句子拼接，只要第二个句子预测的结果
                end_pred = end_pred.squeeze(0)[len(input_ids_1):-1]
                predict_entities = mrc_decode(start_pred,end_pred,sent)
                for pred in pred_entities:
                    pred_entities.append([pred,target])
            chars = []
            labels = []
            origin_labels = []


            predict_num += len(predict_entities)
            logger.info("Predict NER:%s"%predict_entities)
            logger.info("-"*15+"\n")
            for pred in pred_entities:
                if pred in entities:
                    correct_num += 1


logger.info("gold_num = %d"%gold_num)
logger.info("predict_num = %d"%predict_num)
logger.info("correct_num = %d"%correct_num)
P = correct_num/predict_num
logger.info("P = %.4f"%P)
R = correct_num/gold_num
logger.info("R = %.4f"%R)
F1 = 2*P*R/(P+R)
logger.info("F1 = %.4f"%F1)




                
