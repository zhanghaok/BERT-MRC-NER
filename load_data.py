import torch
import torch.utils.data as tud
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = './data/train_data.txt'
TEST_DATA_PATH = './data/test_data.txt'
TOKENIZER_PATH = '/home/zhk/workstation/bert-crf/bert-base-chinese'
BATCH_SIZE = 16
MAX_LEN = 128
tempalte = [("请找出句子中提及的机构","ORG"),("请找出句子中提及的地名","LOC"),("请找出句子中提及的人名","PER")]

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, token_type_ids_list, attention_mask_list, start_ids_list, end_ids_list = [], [], [], [], []
    for instance in batch_data:
        #按照batch中的最大数据长度，对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        attention_mask_temp = instance["attention_mask"]
        start_ids_temp = instance["start_ids"]
        end_ids_tmp = instance["end_ids"]
        #添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp,dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp,dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp,dtype=torch.long))
        start_ids_list.append(torch.tensor(start_ids_temp,dtype=torch.long))
        end_ids_list.append(torch.tensor(end_ids_tmp,dtype=torch.long))
    #使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids":pad_sequence(input_ids_list,batch_first=True,padding_value=0),
            "token_type_ids":pad_sequence(token_type_ids_list,batch_first=True,padding_value=1),
            "attention_mask":pad_sequence(attention_mask_list,batch_first=True,padding_value=0),
            "start_ids":pad_sequence(start_ids_list,batch_first=True,padding_value=-100),
            "end_ids":pad_sequence(end_ids_list,batch_first=True,padding_value=-100)}
        
    pass

class NERDataset(tud.Dataset):
    def __init__(self,data_path,tokenizer_path,max_len) -> None:
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        chars = []
        labels = []
        self.data_set = []
        with open(data_path,encoding='utf-8') as rf:
            for line in rf:
                if line != '\n':
                    char , label = line.strip().split()
                    chars.append(char)
                    if '-' in label:#说明不是非实体，而是三种实体类型中的一种
                        label = label.split('-')[1] #label是个列表如['B','LOC']这种,但是只取最后一个元素LOC，以此类推
                    labels.append(label)
                else:
                    for predix,target in tempalte:
                        # Predix是问题的前缀
                        # target是目标类型
                        input_ids_1 = [tokenizer.convert_tokens_to_ids(c) for c in predix]#inputs_ids_1表示输入模板的token id
                        input_ids_1 = [tokenizer.cls_token_id] + input_ids_1 + [tokenizer.sep_token_id]
                        #Bert两个句子进行拼接时候，token_type_ids:在第一个句子的位置为0，第二个句子为1
                        token_type_ids_1 = [0]*len(input_ids_1)
                        start_ids_1 = end_ids_1 = [-100]*len(input_ids_1)
                        if len(chars)+1+len(input_ids_1)>max_len:
                            chars = chars[:max_len-1-len(input_ids_1)]
                            labels = labels[:max_len-1-len(input_ids_1)]
                        input_ids_2 = [tokenizer.convert_tokens_to_ids(c) for c in chars]#inputs_ids_2表示数据集中每句话的token id
                        input_ids_2 = input_ids_2 + [tokenizer.sep_token_id]
                        token_type_ids_2 = [1]*len(input_ids_2)
                        labels_ = labels + ["O"]#加的O是[seq]的类型
                        start_ids_2,end_ids_2 = self.get_ids(target,labels_)
                        start_ids_2[-1] = -100
                        end_ids_2[-1] = -100
                        input_ids = input_ids_1 + input_ids_2
                        token_type_ids = token_type_ids_1 + token_type_ids_2
                        start_ids = start_ids_1 + start_ids_2
                        end_ids = end_ids_1 + end_ids_2
                        assert len(input_ids) == len(token_type_ids) == len(start_ids) == len(end_ids)
                        self.data_set.append({"input_ids": input_ids, 
                                              "token_type_ids": token_type_ids,
                                              "attention_mask":[1]*len(input_ids), 
                                              "start_ids":start_ids, 
                                              "end_ids":end_ids})
                    chars =[]
                    labels = []


    @staticmethod
    def get_ids(target,data):
        # data表示一句话的真实的标签但是这些标签去除了B和I，只有类型
        # target是目标的标签比如 LOC
        #返回真是标签的位置索引1，其他位置为0
        start_ids = [0]*len(data)
        end_ids = [0]*len(data)
        flag = 0
        for ind, t in enumerate(data):
            if not flag:
                if t == target:
                    start_ids[ind] = 1
                    flag = 1
            else:
                if t != target:
                    end_ids[ind-1] = 1
                    flag = 0
        if flag:
            end_ids[ind] = 1
        return(start_ids,end_ids)

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, index):
        return self.data_set[index]

traindataset = NERDataset(TRAIN_DATA_PATH,TOKENIZER_PATH,MAX_LEN)
traindataloader = tud.DataLoader(traindataset,BATCH_SIZE,shuffle=True,collate_fn=collate_fn,num_workers=10)
# for idx,data in enumerate(traindataloader):
#     print(data["start_ids"].shape)#torch.Size([16, 128])
#     print(data["start_ids"].view(-1).shape)#torch.Size([16, 128])
#     break
# print(traindataset.__getitem__(0))