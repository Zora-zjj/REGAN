import random
import numpy as np
import torch

# from main import BATCH_SIZE, VOCAB_SIZE, g_sequence_len

class DataLoader:
    
    def __init__(self, file_path, batch_size=16):
        
        self.batch_size = batch_size
        self.char_to_ix = {
            'x': 0,
            '+': 1,
            '-': 2,
            '*': 3,
            '/': 4,
            '_': 5,
            #'\n': 6
        }
        self.ix_to_char = {v:k for (k,v) in self.char_to_ix.items()}                         #建立字典
        self.readFile(file_path)    # readFile后面函数：按行阅读
        self.idx = 0
        
    def __len__(self):
        pass
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()                        #next后面函数
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.lines)                #行行打散
            
    def next(self):                               #next是迭代   #返回某一个batch的all_input_data, all_target_data  
        
        # iterator edge case
        if self.idx >= self.total_lines:          #total_lines : 行数
            raise StopIteration
    
        # figure out end_index based on what is left in the list   根据表中剩下的内容计算end_index
        if(self.idx + self.batch_size < self.total_lines):  #将lines分成一个个batch，  [每个batch的第一个为idx,end_index]
            end_index = self.idx + self.batch_size
        else:
            end_index = self.total_lines
            
        #包含字符串列表(长度为batch_size，该列表中的每个元素都是math eq字符串)
        batch_lines = self.lines[self.idx : end_index]   #lines中一部分数据即某个batch的数据，长度是一个batch_size，行列数据
        
        #increment idx (bookeeping for iterator)
        self.idx += self.batch_size                      #下一个batch的idx
        
        # contains input data to be returned
        all_input_data = []
        # contains target data to be returned
        all_target_data = []
        
        for i,line in enumerate(batch_lines):    #i从1到batch_size，取某个batch内的数据
            # convert char to index (do this for input data and target data)
            # here input data and target data are staggered by one position  #stagger交错
            input_data = [self.char_to_ix[c] for c in line]  #c是单词？？？，返回某个batch某一行的单词id列表
            # target doesn't contain the first char, add 6 (maps to '\n') to end
            if i == end_index-1:
                print('break here')
            target_data = input_data[1:]   #往后移一位单词的单词id列表
            target_data.append(random.choice([1,2,3,4]))      #random.choice从给定的1维数组中随机采样，取值+—*/的id号，作用保持长度相等？？？

            # print(f"line {i}. input_data = {input_data}, target_data = {target_data}")

            all_input_data.append(input_data)         #一个batch的所有行
            all_target_data.append(target_data)

        # convert to torch long tensor (ready to be used by nn.Embedding)
        all_input_data = torch.from_numpy(np.asarray(all_input_data)).long()
        all_target_data = torch.from_numpy(np.asarray(all_target_data)).long()

        return all_input_data, all_target_data  #维度[batch_size,seq_len]，数据：单词的id号
    
    def readFile(self, file_path):         #按行阅读
        with open(file_path, 'r') as f:
            self.lines = f.read().split('\n')
        self.total_lines = len(self.lines)

    def frequency(self, file_path, vocab_size=5, seq_len=15):                            
        freq_arr = np.zeros((vocab_size,vocab_size))
        with open(file_path, 'r') as f:
            self.lines = f.read().split('\n')
            chars = list(self.lines)    #输出：列表[“第一句话”，“第一句话”，…]
    
            for i in range(1,len(chars)):  # i表示第几句话
                freq_arr[self.char_to_ix.get(chars[i-1]),self.char_to_ix.get(chars[i])]+=1      # 频数矩阵？？？这样说chars应该是单词而不是句子
                                                               # dict.get(key, default=None)，例如dict.get('Sex', "Never")可多个key，返回字典指定键的值
        if seq_len == 15:    #？？？
            np.save('freq_array.npy',freq_arr/np.sum(freq_arr))   #每个单词占比？？？
        elif seq_len == 3:
            np.save('freq_array_3.npy', freq_arr / np.sum(freq_arr))


        self.total_lines = len(self.lines)


    def convert_to_char(self, data):          #data形式----batch个句子的单词id的序列？？？例如上面的all_input_data, all_target_data？？？
        string_arr = []
        for each_tensor in data:    #each_tensor是一个句子的单词id号序列
            string = ''.join([self.ix_to_char[i] for i in each_tensor.data.numpy()])  #i是某个id，再转成单词
            string_arr.append(string)
        return string_arr   #由id序列转化成单词序列
